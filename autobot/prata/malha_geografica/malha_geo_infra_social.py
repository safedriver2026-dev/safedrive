import os
import boto3
import duckdb
import polars as pl
import unicodedata
import zipfile
import json
import glob
import re
import hashlib
from datetime import datetime
from botocore.config import Config

# ==========================================
# ARQUITETURA SAFEDRIVER: CAMADA PRATA (TURBO ENGINE + LGPD)
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

        # Chave Secreta LGPD
        self.pepper = os.getenv("LGPD_PEPPER", "safedriver_seguranca_padrao_2026").strip()
        
        # Motor C++ para cálculos espaciais ultra-rápidos
        self.con = duckdb.connect(database=':memory:')
        self.con.execute("INSTALL spatial; LOAD spatial;")
        self.con.execute("INSTALL h3; LOAD h3;") 
        
        self.auditoria = {"DATA_EXECUCAO": str(datetime.now()), "CAMADAS": {}}

    def aplicar_lgpd(self, df: pl.DataFrame, colunas_sensiveis: list) -> pl.DataFrame:
        """
        Aplica Hash SHA-256 irreversível com Pepper em colunas sensíveis (ex: Razão Social MEI que contém CPF).
        Garante conformidade com a LGPD sem destruir a capacidade de cruzamento (JOIN) no futuro.
        """
        colunas_presentes = [c for c in colunas_sensiveis if c in df.columns]
        if not colunas_presentes:
            return df
            
        print(f"   🛡️ [LGPD] Anonimizando em massa: {colunas_presentes}...", flush=True)
        
        def hash_seguro(valor):
            if valor is None or str(valor).strip() in ["", "NULL", "NAN", "."]: 
                return "NAO_INFORMADO"
            texto_base = str(valor).upper().strip() + self.pepper
            return hashlib.sha256(texto_base.encode('utf-8')).hexdigest()

        # Polars processa a transformação criptográfica de forma otimizada
        exprs = [
            pl.col(c).map_elements(hash_seguro, return_dtype=pl.Utf8).alias(c)
            for c in colunas_presentes
        ]
        return df.with_columns(exprs)

    def _normalizar(self, valor):
        if not valor or str(valor).upper() in ["NULL", "NAN", "."]: return "NAO INFORMADO"
        t = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^a-zA-Z0-9\s]', '', t).upper().strip()

    def download_r2(self):
        print("📥 Buscando arquivos na Bronze via Boto3...", flush=True)
        paginator = self.s3.get_paginator('list_objects_v2')
        targets = ["SP_Faces_2022.zip", "Agregados_por_setores_basico", "CNPJ_SP_HISTORICO_LOTE_"]
        
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
        
        if encontrados == 0:
            raise FileNotFoundError("Nenhum ficheiro encontrado no R2!")
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
        # ==========================================
        # 1. GEO (Cálculo H3 e Extração via DuckDB)
        # ==========================================
        print("🗺️ Processando Malha Geográfica...", flush=True)
        try:
            zip_f = glob.glob(f"{self.bronze_dir}/**/SP_Faces_2022.zip", recursive=True)[0]
            with zipfile.ZipFile(zip_f, 'r') as z: z.extractall(self.bronze_dir)
            
            query = f"""
                SELECT 
                    CD_FACE,
                    trim(NM_TIP_LOG || ' ' || NM_LOG) as RUA, 
                    CAST(ST_Y(ST_Centroid(geom)) AS FLOAT) as LAT, 
                    CAST(ST_X(ST_Centroid(geom)) AS FLOAT) as LON,
                    h3_cell_to_string(h3_latlng_to_cell(ST_Y(ST_Centroid(geom)), ST_X(ST_Centroid(geom)), {self.H3_RES})) as H3_INDEX
                FROM ST_Read('{self.bronze_dir}/**/*.json')
                WHERE NM_LOG IS NOT NULL
            """
            df = self.con.execute(query).pl()
            
            df = df.with_columns([
                pl.col("RUA").map_elements(self._normalizar, return_dtype=pl.Utf8).cast(pl.Categorical),
                pl.col("H3_INDEX").cast(pl.Categorical)
            ]).unique(subset=["CD_FACE"])
            
            df.write_parquet(f"{self.prata_dir}/PRATA_MALHA_GEOGRAFICA_VIAS.parquet", compression="zstd")
            self.auditoria["CAMADAS"]["GEO"] = {"STATUS": "OK", "LINHAS": df.height}
            print(f"   ✅ Geo Concluído: {df.height} linhas.", flush=True)
        except Exception as e: 
            print(f"❌ Erro Geo: {e}", flush=True)
            self.auditoria["CAMADAS"]["GEO"] = {"STATUS": "ERRO", "MSG": str(e)}

        # ==========================================
        # 2. SOCIAL (Polars Lazy Execution)
        # ==========================================
        print("👥 Processando Malha Social...", flush=True)
        try:
            csv_f = glob.glob(f"{self.bronze_dir}/**/Agregados_por_setores_basico*.csv", recursive=True)[0]
            
            df_lazy = pl.scan_csv(csv_f, separator=";", null_values=["."], infer_schema_length=10000, schema_overrides={"CD_SETOR": pl.Utf8, "CD_MUN": pl.Utf8})
            df_lazy = df_lazy.filter(pl.col("CD_SETOR").str.starts_with("35"))
            
            cols = ["CD_SETOR", "NM_MUN", "NM_BAIRRO", "AREA_KM2", "CD_TIPO", "v0001", "v0002"]
            df = df_lazy.select([c for c in cols]).with_columns([
                pl.col("AREA_KM2").str.replace(",", ".").cast(pl.Float32, strict=False),
                pl.col("v0001").cast(pl.UInt32, strict=False).fill_null(0).alias("POPULACAO"),
                pl.col("v0002").cast(pl.UInt32, strict=False).fill_null(0).alias("DOMICILIOS")
            ]).collect()
            
            df.write_parquet(f"{self.prata_dir}/PRATA_MALHA_SOCIAL.parquet", compression="zstd")
            self.auditoria["CAMADAS"]["SOCIAL"] = {"STATUS": "OK", "LINHAS": df.height}
            print(f"   ✅ Social Concluído: {df.height} linhas.", flush=True)
        except Exception as e: 
            print(f"❌ Erro Social: {e}", flush=True)
            self.auditoria["CAMADAS"]["SOCIAL"] = {"STATUS": "ERRO", "MSG": str(e)}

        # ==========================================
        # 3. INFRA (H3 C++ e Proteção LGPD)
        # ==========================================
        print("🏗️ Processando Malha Infra/Comércio...", flush=True)
        try:
            query_infra = f"""
                SELECT 
                    *,
                    h3_cell_to_string(h3_latlng_to_cell(lat, lon, {self.H3_RES})) as H3_INDEX
                FROM read_parquet('{self.bronze_dir}/**/CNPJ_SP_HISTORICO_LOTE_*.parquet')
                WHERE lat IS NOT NULL AND lon IS NOT NULL
            """
            df = self.con.execute(query_infra).pl()
            
            # --- BLINDAGEM LGPD APLICADA AOS CNPJS ---
            # Esconde CPFs vazados em nomes de empresas MEI e o próprio número de registo exato
            colunas_perigosas = ["cnpj", "cnpj_basico", "razao_social", "nome_fantasia", "cpf_responsavel", "nome_socio"]
            df = self.aplicar_lgpd(df, colunas_perigosas)
            
            cat_cols = ["H3_INDEX", "cnae_fiscal_principal", "cep", "municipio", "bairro", "situacao_cadastral"]
            df = df.with_columns(
                [pl.col(c).cast(pl.Categorical) for c in cat_cols if c in df.columns] + 
                [pl.col("lat").cast(pl.Float32), pl.col("lon").cast(pl.Float32)]
            )
            
            df.write_parquet(f"{self.prata_dir}/PRATA_MALHA_INFRA.parquet", compression="zstd")
            self.auditoria["CAMADAS"]["INFRA"] = {"STATUS": "OK", "LINHAS": df.height}
            print(f"   ✅ Infra Concluído: {df.height} linhas.", flush=True)
        except Exception as e: 
            print(f"❌ Erro Infra: {e}", flush=True)
            self.auditoria["CAMADAS"]["INFRA"] = {"STATUS": "ERRO", "MSG": str(e)}

    def finalizar(self):
        with open(f"{self.prata_dir}/AUDITORIA_PRATA.json", "w") as f:
            json.dump(self.auditoria, f, indent=4)
        print("\n📊 AUDITORIA:\n", json.dumps(self.auditoria, indent=4), flush=True)

        # Validador de Saída para o GitHub Actions
        erros = [v for k, v in self.auditoria["CAMADAS"].items() if v.get("STATUS") == "ERRO"]
        if erros:
            self.auditoria["QUALIDADE_GERAL"] = "FALHA_CRITICA"
            with open(f"{self.prata_dir}/AUDITORIA_PRATA.json", "w") as f:
                json.dump(self.auditoria, f, indent=4)

if __name__ == "__main__":
    app = ArquitetoSafeDriverPrata()
    app.download_r2()
    app.processar()
    app.finalizar()
    app.upload_r2()
