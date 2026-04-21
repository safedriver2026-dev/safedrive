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
import csv
import urllib.request
from datetime import datetime
from botocore.config import Config

# ==========================================
# ARQUITETURA SAFEDRIVER: CAMADA PRATA (TURBO BLINDADA)
# ==========================================
class ArquitetoSafeDriverPrata:
    def __init__(self):
        # Configurações de Resolução e Caminhos
        self.H3_RES = 9 
        self.bronze_dir = "./data_raw"
        self.prata_dir = "./data_prata"
        os.makedirs(self.bronze_dir, exist_ok=True)
        os.makedirs(self.prata_dir, exist_ok=True)
        
        # Conexão Boto3 Resiliente
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        if endpoint.endswith(f"/{self.bucket}"):
            endpoint = endpoint[: -len(f"/{self.bucket}")]
            
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint,
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4', retries={'max_attempts': 3})
        )

        self.pepper = os.getenv("LGPD_PEPPER", "safedriver_seguranca_padrao_2026").strip()
        
        # Motor DuckDB com Limite de Memória para GitHub Actions (7GB limite)
        self.con = duckdb.connect(database=':memory:')
        self.con.execute("PRAGMA memory_limit='6GB';")
        self.con.execute("PRAGMA temp_directory='./temp_duckdb';")
        os.makedirs("./temp_duckdb", exist_ok=True)
        
        try:
            self.con.execute("INSTALL spatial; LOAD spatial;")
        except Exception as e:
            raise RuntimeError(f"Falha crítica no DuckDB Spatial: {e}")
        
        self.auditoria = {"DATA_EXECUCAO": str(datetime.now()), "CAMADAS": {}}

    # ==========================================
    # MOTORES DE AUTO-DETECÇÃO E SEGURANÇA
    # ==========================================
    def _buscar_arquivo_seguro(self, padrao, nome_amigavel):
        arquivos = glob.glob(f"{self.bronze_dir}/**/{padrao}", recursive=True)
        if not arquivos:
            raise FileNotFoundError(f"⚠️ Arquivo ausente: {nome_amigavel} ({padrao})")
        return arquivos[0]

    def _analisar_csv_dinamicamente(self, file_path):
        """Sniffer: Descobre Encoding e Separador sem intervenção humana."""
        print(f"   🔎 [Auto-Detect] Analisando estrutura do Censo...", flush=True)
        encodings = ['utf-8', 'windows-1252', 'iso-8859-1', 'latin1']
        
        with open(file_path, 'rb') as f:
            amostra_bytes = f.read(100000)
            
        encoding_detectado = 'utf-8'
        for enc in encodings:
            try:
                amostra_bytes.decode(enc)
                encoding_detectado = enc
                break
            except UnicodeDecodeError:
                continue
                
        separador = ";"
        try:
            amostra_texto = amostra_bytes.decode(encoding_detectado)
            sniffer = csv.Sniffer()
            dialect = sniffer.sniff(amostra_texto, delimiters=";,|\t")
            separador = dialect.delimiter
        except:
            pass

        print(f"   ✅ [Auto-Detect] Mapeado: {encoding_detectado.upper()} | Separador: '{separador}'", flush=True)
        return encoding_detectado, separador

    def _garantir_osmconf_offline(self):
        """Garante as regras de tradução do PBF sem depender da internet."""
        osmconf_path = os.path.join(self.bronze_dir, "osmconf.ini")
        if not os.path.exists(osmconf_path):
            config = "[lines]\nosm_id=yes\nattributes=name,highway,waterway,aerialway,barrier,man_made,z_order\n"
            with open(osmconf_path, "w") as f: f.write(config)
        return osmconf_path

    # ==========================================
    # MÓDULOS DE PROCESSAMENTO (IA & LGPD)
    # ==========================================
    def aplicar_lgpd(self, df: pl.DataFrame, colunas: list) -> pl.DataFrame:
        presentes = [c for c in colunas if c in df.columns]
        if not presentes: return df
        print(f"   🛡️ [LGPD] Anonimizando: {presentes}...", flush=True)
        
        def hash_seguro(v):
            if v is None or str(v).strip() in ["", "NULL", "NAN", "."]: return "NAO_INFORMADO"
            return hashlib.sha256((str(v).upper().strip() + self.pepper).encode('utf-8')).hexdigest()

        return df.with_columns([pl.col(c).map_elements(hash_seguro, return_dtype=pl.Utf8) for c in presentes])

    def _normalizar(self, valor):
        if not valor or str(valor).upper() in ["NULL", "NAN", "."]: return "NAO INFORMADO"
        t = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^a-zA-Z0-9\s]', '', t).upper().strip()

    def normalizar_strings_global(self, df: pl.DataFrame) -> pl.DataFrame:
        string_cols = df.select(cs.string()).columns
        if not string_cols: return df
        print(f"   🧹 [Limpeza] Padronizando textos: {string_cols}", flush=True)
        for col in string_cols:
            unique_df = df.select(pl.col(col).unique().alias(col)).drop_nulls()
            if unique_df.height > 0:
                mapped = unique_df.with_columns(pl.col(col).map_elements(self._normalizar, return_dtype=pl.Utf8).alias("_n"))
                df = df.join(mapped, on=col, how="left").with_columns(pl.col("_n").fill_null("NAO INFORMADO").alias(col)).drop("_n")
            df = df.with_columns(pl.col(col).cast(pl.Categorical))
        return df

    def aplicar_h3_otimizado(self, df: pl.DataFrame, lat_col="lat", lon_col="lon") -> pl.DataFrame:
        print(f"   📍 [H3] Indexando coordenadas...", flush=True)
        unicas = df.select([lat_col, lon_col]).drop_nulls().unique()
        if unicas.height > 0:
            unicas = unicas.with_columns(pl.struct([lat_col, lon_col]).map_batches(
                lambda s: pl.Series([h3.latlng_to_cell(x[lat_col], x[lon_col], self.H3_RES) for x in s])
            ).alias("H3_INDEX").cast(pl.Categorical))
            return df.join(unicas, on=[lat_col, lon_col], how="left")
        return df.with_columns(pl.lit(None).alias("H3_INDEX").cast(pl.Categorical))

    def completar_por_vizinhanca_h3(self, df: pl.DataFrame, alvos: list) -> pl.DataFrame:
        cols = [c for c in alvos if c in df.columns]
        if not cols or "H3_INDEX" not in df.columns: return df
        print(f"   🪄 [IA Espacial] Imputando vazios via H3: {cols}...", flush=True)
        
        aggs = [pl.col(c).filter((pl.col(c).is_not_null()) & (pl.col(c) != "NAO INFORMADO")).mode().first().alias(f"{c}_INF") for c in cols]
        dict_h3 = df.group_by("H3_INDEX").agg(aggs)
        df = df.join(dict_h3, on="H3_INDEX", how="left")
        
        return df.with_columns([
            pl.when((pl.col(c).is_null()) | (pl.col(c) == "NAO INFORMADO")).then(pl.col(f"{c}_INF")).otherwise(pl.col(c)).fill_null("NAO INFORMADO").alias(c)
            for c in cols
        ]).drop([f"{c}_INF" for c in cols])

    # ==========================================
    # WORKFLOW PRINCIPAL
    # ==========================================
    def download_r2(self):
        print("📥 Sincronizando Bronze do R2...", flush=True)
        targets = ["SP_Faces_2022.zip", "sp-latest.osm.pbf", "Agregados_por_setores_basico", "CNPJ_SP_HISTORICO_LOTE_"]
        paginator = self.s3.get_paginator('list_objects_v2')
        encontrados = 0
        for page in paginator.paginate(Bucket=self.bucket):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if any(t in key for t in targets):
                    filename = key.split('/')[-1]
                    self.s3.download_file(self.bucket, key, os.path.join(self.bronze_dir, filename))
                    encontrados += 1
        print(f"✅ Download finalizado: {encontrados} arquivos.", flush=True)

    def processar(self):
        # 1. GEO (IBGE JSONs + OSM PBF)
        print("🗺️ Processando Malha Geográfica (IBGE + OSM)...", flush=True)
        try:
            zip_f = self._buscar_arquivo_seguro("SP_Faces_2022.zip", "IBGE Faces")
            if not glob.glob(f"{self.bronze_dir}/**/*.json", recursive=True):
                with zipfile.ZipFile(zip_f, 'r') as z: z.extractall(self.bronze_dir)
            
            json_files = glob.glob(f"{self.bronze_dir}/**/*.json", recursive=True)
            osm_pbf = self._buscar_arquivo_seguro("sp-latest.osm.pbf", "OpenStreetMap")
            osmconf = self._garantir_osmconf_offline()
            
            # UNION Dinâmico para suportar centenas de arquivos JSON (ST_Read rígido)
            ibge_queries = [f"SELECT CAST(CD_FACE AS VARCHAR) as CD_FACE, trim(NM_TIP_LOG || ' ' || NM_LOG) as RUA, CAST(NULL AS VARCHAR) as NM_MUN, CAST(NULL AS VARCHAR) as NM_BAIRRO, CAST(ST_Y(ST_Centroid(geom)) AS FLOAT) as LAT, CAST(ST_X(ST_Centroid(geom)) AS FLOAT) as LON FROM ST_Read('{f.replace('\\','/')}') WHERE NM_LOG IS NOT NULL" for f in json_files]
            
            # Query Híbrida: OSM (Leitura Sequencial para evitar OOM) + IBGE
            query = f"""
                SELECT CAST(osm_id AS VARCHAR) as CD_FACE, name as RUA, CAST(NULL AS VARCHAR) as NM_MUN, CAST(NULL AS VARCHAR) as NM_BAIRRO, CAST(ST_Y(ST_Centroid(geom)) AS FLOAT) as LAT, CAST(ST_X(ST_Centroid(geom)) AS FLOAT) as LON
                FROM ST_Read('{osm_pbf}', layer='lines', open_options=['CONFIG_FILE={osmconf}', 'INTERLEAVED_READING=YES']) 
                WHERE highway IS NOT NULL AND name IS NOT NULL
                UNION ALL {" UNION ALL ".join(ibge_queries)}
            """
            df = self.con.execute(query).pl()
            
            df = self.normalizar_strings_global(df)
            df = self.aplicar_h3_otimizado(df, lat_col="LAT", lon_col="LON")
            df = self.completar_por_vizinhanca_h3(df, ["NM_MUN", "NM_BAIRRO"])
            
            df.unique(subset=["RUA", "H3_INDEX"]).write_parquet(f"{self.prata_dir}/PRATA_MALHA_GEOGRAFICA_VIAS.parquet", compression="zstd")
            self.auditoria["CAMADAS"]["GEO"] = {"STATUS": "OK", "LINHAS": df.height}
            print(f"   ✅ Geo Concluído: {df.height} registros.", flush=True)
        except Exception as e: self.auditoria["CAMADAS"]["GEO"] = {"STATUS": "ERRO", "MSG": str(e)}

        # 2. SOCIAL (Censo Dinâmico)
        print("👥 Processando Malha Social...", flush=True)
        try:
            csv_f = self._buscar_arquivo_seguro("Agregados_por_setores_basico*.csv", "Censo")
            enc, sep = self._analisar_csv_dinamicamente(csv_f)
            df = pl.read_csv(csv_f, separator=sep, null_values=["."], encoding=enc, infer_schema_length=10000, schema_overrides={"CD_SETOR": pl.Utf8, "CD_MUN": pl.Utf8}).filter(pl.col("CD_SETOR").str.starts_with("35"))
            
            df = df.select([c for c in ["CD_SETOR", "NM_MUN", "NM_BAIRRO", "AREA_KM2", "v0001", "v0002"] if c in df.columns]).with_columns([
                pl.col("AREA_KM2").str.replace(",", ".").cast(pl.Float32, strict=False),
                pl.col("v0001").cast(pl.UInt32, strict=False).fill_null(0).alias("POPULACAO"),
                pl.col("v0002").cast(pl.UInt32, strict=False).fill_null(0).alias("DOMICILIOS")
            ])
            
            df = self.normalizar_strings_global(df)
            df.write_parquet(f"{self.prata_dir}/PRATA_MALHA_SOCIAL.parquet", compression="zstd")
            self.auditoria["CAMADAS"]["SOCIAL"] = {"STATUS": "OK", "LINHAS": df.height}
            print(f"   ✅ Social Concluído.", flush=True)
        except Exception as e: self.auditoria["CAMADAS"]["SOCIAL"] = {"STATUS": "ERRO", "MSG": str(e)}

        # 3. INFRA (CNPJ + IA Imputação)
        print("🏗️ Processando Malha Infra...", flush=True)
        try:
            pqs = glob.glob(f"{self.bronze_dir}/**/CNPJ_SP_HISTORICO_LOTE_*.parquet", recursive=True)
            df = pl.read_parquet(pqs).filter(pl.col("lat").is_not_null())
            df = self.aplicar_lgpd(df, ["cnpj", "razao_social", "nome_fantasia", "cpf_responsavel"])
            df = self.normalizar_strings_global(df)
            df = self.aplicar_h3_otimizado(df, lat_col="lat", lon_col="lon")
            df = self.completar_por_vizinhanca_h3(df, ["municipio", "bairro"])
            
            df.write_parquet(f"{self.prata_dir}/PRATA_MALHA_INFRA.parquet", compression="zstd")
            self.auditoria["CAMADAS"]["INFRA"] = {"STATUS": "OK", "LINHAS": df.height}
            print(f"   ✅ Infra Concluída.", flush=True)
        except Exception as e: self.auditoria["CAMADAS"]["INFRA"] = {"STATUS": "ERRO", "MSG": str(e)}

    def finalizar(self):
        projeto = os.getenv("BQ_PROJECT_ID", "safedriver").strip()
        erros = [v for k, v in self.auditoria["CAMADAS"].items() if v.get("STATUS") == "ERRO"]
        self.auditoria["QUALIDADE_GERAL"] = "FALHA_CRITICA" if erros else "SUCESSO_TOTAL"
        
        with open(f"{self.prata_dir}/AUDITORIA_PRATA.json", "w") as f: json.dump(self.auditoria, f, indent=4)
        for file in os.listdir(self.prata_dir):
            if file.endswith(".parquet") or file.endswith(".json"):
                self.s3.upload_file(os.path.join(self.prata_dir, file), self.bucket, f"{projeto}/datalake/prata/malhas/{file}")
        print("\n📊 AUDITORIA FINALIZADA:\n", json.dumps(self.auditoria, indent=4), flush=True)

if __name__ == "__main__":
    app = ArquitetoSafeDriverPrata()
    app.download_r2()
    app.processar()
    app.finalizar()
