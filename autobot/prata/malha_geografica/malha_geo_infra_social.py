import os
import boto3
import duckdb
import polars as pl
import polars.selectors as cs
import h3
import unicodedata
import zipfile
import json
import glob
import re
import hashlib
from datetime import datetime
from botocore.config import Config

# ==========================================
# ARQUITETURA SAFEDRIVER: CAMADA PRATA (POLARS EXTREME + OSM)
# ==========================================
class ArquitetoSafeDriverPrata:
    def __init__(self):
        self.H3_RES = 9 
        self.bronze_dir = "./data_raw"
        self.prata_dir = "./data_prata"
        os.makedirs(self.bronze_dir, exist_ok=True)
        os.makedirs(self.prata_dir, exist_ok=True)
        
        # Conexão Boto3 Blindada
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        if endpoint.endswith(f"/{self.bucket}"):
            endpoint = endpoint[: -len(f"/{self.bucket}")]
            
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint,
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4')
        )

        self.pepper = os.getenv("LGPD_PEPPER", "safedriver_seguranca_padrao_2026").strip()
        
        # Conexão DuckDB (Usada apenas para ler a geometria espacial do SP_Faces e do OSM PBF)
        self.con = duckdb.connect(database=':memory:')
        self.con.execute("INSTALL spatial; LOAD spatial;")
        
        self.auditoria = {"DATA_EXECUCAO": str(datetime.now()), "CAMADAS": {}}

    # ==========================================
    # MOTORES DE PERFORMANCE E LIMPEZA
    # ==========================================
    def aplicar_lgpd(self, df: pl.DataFrame, colunas_sensiveis: list) -> pl.DataFrame:
        """Aplica Hash SHA-256 irreversível com Pepper em colunas sensíveis (ex: Razões Sociais com CPF)."""
        colunas_presentes = [c for c in colunas_sensiveis if c in df.columns]
        if not colunas_presentes: return df
            
        print(f"   🛡️ [LGPD] Anonimizando em massa: {colunas_presentes}...", flush=True)
        def hash_seguro(valor):
            if valor is None or str(valor).strip() in ["", "NULL", "NAN", "."]: return "NAO_INFORMADO"
            return hashlib.sha256((str(valor).upper().strip() + self.pepper).encode('utf-8')).hexdigest()

        exprs = [pl.col(c).map_elements(hash_seguro, return_dtype=pl.Utf8).alias(c) for c in colunas_presentes]
        return df.with_columns(exprs)

    def _normalizar(self, valor):
        """Motor base que arranca acentos e deixa maiúsculo."""
        if not valor or str(valor).upper() in ["NULL", "NAN", "."]: return "NAO INFORMADO"
        t = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^a-zA-Z0-9\s]', '', t).upper().strip()

    def normalizar_strings_global(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Encontra TODAS as colunas de texto (Bairro, Logradouro, Município, etc.) e aplica a limpeza.
        Usa "Unique Join" para não estourar o limite de tempo do GitHub Actions.
        """
        string_cols = df.select(cs.string()).columns
        if not string_cols: return df
        
        print(f"   🧹 [Limpeza] Normalizando textos para Maiúsculas e sem acento: {string_cols}", flush=True)
        
        for col in string_cols:
            unique_df = df.select(pl.col(col).unique().alias(col)).drop_nulls()
            mapped_df = unique_df.with_columns(
                pl.col(col).map_elements(self._normalizar, return_dtype=pl.Utf8).alias(f"{col}_norm")
            )
            df = df.join(mapped_df, on=col, how="left")
            df = df.with_columns(pl.col(f"{col}_norm").fill_null("NAO INFORMADO").alias(col)).drop(f"{col}_norm")
            df = df.with_columns(pl.col(col).cast(pl.Categorical)) 
            
        return df

    def aplicar_h3_otimizado(self, df: pl.DataFrame, lat_col="lat", lon_col="lon") -> pl.DataFrame:
        """Calcula o H3 utilizando a mesma técnica de Otimização de Coordenadas Únicas."""
        print("   📍 [H3] Calculando malha hexagonal (Modo Otimizado)...", flush=True)
        coords_unicas = df.select([lat_col, lon_col]).drop_nulls().unique()
        
        coords_unicas = coords_unicas.with_columns(
            pl.struct([lat_col, lon_col]).map_batches(
                lambda s: pl.Series([h3.latlng_to_cell(x[lat_col], x[lon_col], self.H3_RES) for x in s])
            ).alias("H3_INDEX").cast(pl.Categorical)
        )
        return df.join(coords_unicas, on=[lat_col, lon_col], how="left")

    # ==========================================
    # WORKFLOW DE EXECUÇÃO
    # ==========================================
    def download_r2(self):
        print("📥 Buscando arquivos na Bronze via Boto3...", flush=True)
        paginator = self.s3.get_paginator('list_objects_v2')
        # OSM PBF adicionado à lista de downloads
        targets = ["SP_Faces_2022.zip", "sp-latest.osm.pbf", "Agregados_por_setores_basico", "CNPJ_SP_HISTORICO_LOTE_"]
        
        encontrados = 0
        for page in paginator.paginate(Bucket=self.bucket):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if any(t in key for t in targets):
                    filename = key.split('/')[-1]
                    dest = os.path.join(self.bronze_dir, filename)
                    print(f"   ⬇️ Baixando: {filename}", flush=True)
                    self.s3.download_file(self.bucket, key, dest)
                    encontrados += 1
        print(f"✅ Download finalizado: {encontrados} arquivos.", flush=True)

    def upload_r2(self):
        projeto = os.getenv("BQ_PROJECT_ID", "safedriver").strip()
        print(f"📤 Subindo arquivos processados para o R2...", flush=True)
        for file in os.listdir(self.prata_dir):
            if file.endswith(".parquet") or file.endswith(".json"):
                local_path = os.path.join(self.prata_dir, file)
                r2_path = f"{projeto}/datalake/prata/malhas/{file}"
                print(f"   ⬆️ Enviando: {file}", flush=True)
                self.s3.upload_file(local_path, self.bucket, r2_path)
        print("✅ Pipeline concluído com sucesso!", flush=True)

    def processar(self):
        # 1. GEO (IBGE + OSM)
        print("🗺️ Processando Malha Geográfica (IBGE + OSM)...", flush=True)
        try:
            zip_f = glob.glob(f"{self.bronze_dir}/**/SP_Faces_2022.zip", recursive=True)[0]
            with zipfile.ZipFile(zip_f, 'r') as z: z.extractall(self.bronze_dir)
            
            osm_pbf = glob.glob(f"{self.bronze_dir}/**/sp-latest.osm.pbf", recursive=True)[0]
            
            # União das duas bases usando DuckDB puro para leitura espacial
            query = f"""
                SELECT 
                    CAST(osm_id AS VARCHAR) as CD_FACE,
                    name as RUA, 
                    CAST(ST_Y(ST_Centroid(geom)) AS FLOAT) as LAT, 
                    CAST(ST_X(ST_Centroid(geom)) AS FLOAT) as LON
                FROM ST_Read('{osm_pbf}', layer='lines')
                WHERE highway IS NOT NULL AND name IS NOT NULL
                
                UNION ALL
                
                SELECT 
                    CD_FACE,
                    trim(NM_TIP_LOG || ' ' || NM_LOG) as RUA, 
                    CAST(ST_Y(ST_Centroid(geom)) AS FLOAT) as LAT, 
                    CAST(ST_X(ST_Centroid(geom)) AS FLOAT) as LON
                FROM ST_Read('{self.bronze_dir}/**/*.json')
                WHERE NM_LOG IS NOT NULL
            """
            df = self.con.execute(query).pl()
            
            # Limpeza Global e H3 via Polars
            df = self.normalizar_strings_global(df)
            df = self.aplicar_h3_otimizado(df, lat_col="LAT", lon_col="LON")
            
            # Remoção de Duplicatas (Garante que a mesma rua no mesmo H3 não conte em dobro)
            df = df.unique(subset=["RUA", "H3_INDEX"])
            
            df.write_parquet(f"{self.prata_dir}/PRATA_MALHA_GEOGRAFICA_VIAS.parquet", compression="zstd")
            self.auditoria["CAMADAS"]["GEO"] = {"STATUS": "OK", "LINHAS": df.height}
            print(f"   ✅ Geo Concluído: {df.height} linhas.", flush=True)
        except Exception as e: 
            print(f"❌ Erro Geo: {e}", flush=True)
            self.auditoria["CAMADAS"]["GEO"] = {"STATUS": "ERRO", "MSG": str(e)}

        # 2. SOCIAL
        print("👥 Processando Malha Social...", flush=True)
        try:
            csv_f = glob.glob(f"{self.bronze_dir}/**/Agregados_por_setores_basico*.csv", recursive=True)[0]
            df = pl.read_csv(csv_f, separator=";", null_values=["."], infer_schema_length=10000, schema_overrides={"CD_SETOR": pl.Utf8, "CD_MUN": pl.Utf8}).filter(pl.col("CD_SETOR").str.starts_with("35"))
            
            cols = ["CD_SETOR", "NM_MUN", "NM_BAIRRO", "AREA_KM2", "CD_TIPO", "v0001", "v0002"]
            df = df.select([c for c in cols if c in df.columns]).with_columns([
                pl.col("AREA_KM2").str.replace(",", ".").cast(pl.Float32, strict=False),
                pl.col("v0001").cast(pl.UInt32, strict=False).fill_null(0).alias("POPULACAO"),
                pl.col("v0002").cast(pl.UInt32, strict=False).fill_null(0).alias("DOMICILIOS")
            ])
            
            df = self.normalizar_strings_global(df)
            
            df.write_parquet(f"{self.prata_dir}/PRATA_MALHA_SOCIAL.parquet", compression="zstd")
            self.auditoria["CAMADAS"]["SOCIAL"] = {"STATUS": "OK", "LINHAS": df.height}
            print(f"   ✅ Social Concluído: {df.height} linhas.", flush=True)
        except Exception as e: 
            self.auditoria["CAMADAS"]["SOCIAL"] = {"STATUS": "ERRO", "MSG": str(e)}

        # 3. INFRA (CNAE / POI)
        print("🏗️ Processando Malha Infra/Comércio...", flush=True)
        try:
            pqs = glob.glob(f"{self.bronze_dir}/**/CNPJ_SP_HISTORICO_LOTE_*.parquet", recursive=True)
            df = pl.read_parquet(pqs)
            df = df.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
            
            colunas_perigosas = ["cnpj", "cnpj_basico", "razao_social", "nome_fantasia", "cpf_responsavel", "nome_socio"]
            df = self.aplicar_lgpd(df, colunas_perigosas)
            
            df = self.normalizar_strings_global(df)
            df = self.aplicar_h3_otimizado(df, lat_col="lat", lon_col="lon")
            
            df.write_parquet(f"{self.prata_dir}/PRATA_MALHA_INFRA.parquet", compression="zstd")
            self.auditoria["CAMADAS"]["INFRA"] = {"STATUS": "OK", "LINHAS": df.height}
            print(f"   ✅ Infra Concluído: {df.height} linhas.", flush=True)
        except Exception as e: 
            self.auditoria["CAMADAS"]["INFRA"] = {"STATUS": "ERRO", "MSG": str(e)}

    def finalizar(self):
        erros = [v for k, v in self.auditoria["CAMADAS"].items() if v.get("STATUS") == "ERRO"]
        if erros: self.auditoria["QUALIDADE_GERAL"] = "FALHA_CRITICA"
        else: self.auditoria["QUALIDADE_GERAL"] = "SUCESSO_TOTAL"
            
        with open(f"{self.prata_dir}/AUDITORIA_PRATA.json", "w") as f:
            json.dump(self.auditoria, f, indent=4)
        print("\n📊 AUDITORIA:\n", json.dumps(self.auditoria, indent=4), flush=True)

if __name__ == "__main__":
    app = ArquitetoSafeDriverPrata()
    app.download_r2()
    app.processar()
    app.finalizar()
    app.upload_r2()
