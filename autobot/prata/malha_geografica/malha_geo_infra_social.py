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
from pathlib import Path

# ==========================================
# ARQUITETURA SAFEDRIVER: CAMADA PRATA (STREAMING ENGINE)
# ==========================================
class ArquitetoSafeDriverPrata:
    def __init__(self):
        self.H3_RES = 9 
        self.bronze_dir = "./data_raw"
        self.prata_dir = "./data_prata"
        self.temp_dir = "./temp_duckdb"
        
        os.makedirs(self.bronze_dir, exist_ok=True)
        os.makedirs(self.prata_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
        # Conexão R2
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
        
        # DuckDB (Limite rigoroso de RAM para evitar Crash)
        self.con = duckdb.connect(database=':memory:')
        self.con.execute(f"PRAGMA memory_limit='4GB';")
        self.con.execute(f"PRAGMA temp_directory='{self.temp_dir}';")
        
        try:
            self.con.execute("INSTALL spatial; LOAD spatial;")
        except:
            pass # Fallback se já estiver instalado
        
        self.auditoria = {"DATA_EXECUCAO": str(datetime.now()), "CAMADAS": {}}

    # ==========================================
    # UTILITÁRIOS
    # ==========================================
    def _buscar_arquivo_seguro(self, padrao, nome):
        arqs = glob.glob(f"{self.bronze_dir}/**/{padrao}", recursive=True)
        if not arqs: raise FileNotFoundError(f"⚠️ {nome} não encontrado!")
        return arqs[0]

    def _analisar_csv_dinamicamente(self, path):
        encs = ['utf-8', 'windows-1252', 'iso-8859-1', 'latin1']
        with open(path, 'rb') as f: sample = f.read(100000)
        enc_final = 'utf-8'
        for e in encs:
            try:
                sample.decode(e)
                enc_final = e
                break
            except: continue
        sep = ";"
        try:
            sniffer = csv.Sniffer()
            sep = sniffer.sniff(sample.decode(enc_final), delimiters=";,|").delimiter
        except: pass
        return enc_final, sep

    def _normalizar(self, v):
        if v is None or str(v).upper() in ["NULL", "NAN", ".", "", "NONE"]: return "NAO INFORMADO"
        s = "".join(c for c in unicodedata.normalize('NFKD', str(v)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^a-zA-Z0-9\s]', '', s).upper().strip()

    def normalizar_df(self, df: pl.DataFrame) -> pl.DataFrame:
        cols = df.select(cs.string()).columns
        for c in cols:
            uniques = df.select(pl.col(c).unique()).drop_nulls()
            if uniques.height > 0:
                map_df = uniques.with_columns(pl.col(c).map_elements(self._normalizar, return_dtype=pl.Utf8).alias("_n"))
                df = df.join(map_df, on=c, how="left").with_columns(pl.col("_n").fill_null("NAO INFORMADO").alias(c)).drop("_n")
            df = df.with_columns(pl.col(c).cast(pl.Categorical))
        return df

    # ==========================================
    # CORE DE PERFORMANCE
    # ==========================================
    def aplicar_h3_batch(self, df: pl.DataFrame, lat="lat", lon="lon") -> pl.DataFrame:
        """Calcula H3 apenas para coordenadas únicas (economiza 90% de processamento)"""
        coords = df.select([lat, lon]).drop_nulls().unique()
        if coords.height > 0:
            coords = coords.with_columns(pl.struct([lat, lon]).map_batches(
                lambda s: pl.Series([h3.latlng_to_cell(x[lat], x[lon], self.H3_RES) for x in s])
            ).alias("H3_INDEX"))
            return df.join(coords, on=[lat, lon], how="left")
        return df.with_columns(pl.lit(None).alias("H3_INDEX"))

    def imputar_h3_infra(self, df: pl.DataFrame, colunas: list) -> pl.DataFrame:
        """Imputação de Bairro/Município usando dicionário de referência leve."""
        print(f"   🪄 IA Espacial: Recuperando localizações via H3...", flush=True)
        for c in colunas:
            # Cria um guia H3 -> Valor baseado apenas em quem tem o dado
            guia = df.filter((pl.col(c).is_not_null()) & (pl.col(c) != "NAO INFORMADO")) \
                     .group_by("H3_INDEX").agg(pl.col(c).mode().first().alias("_inf"))
            
            df = df.join(guia, on="H3_INDEX", how="left") \
                   .with_columns(pl.when(pl.col(c).is_null() | (pl.col(c) == "NAO INFORMADO"))
                                   .then(pl.col("_inf")).otherwise(pl.col(c)).fill_null("NAO INFORMADO").alias(c)) \
                   .drop("_inf")
        return df

    def aplicar_lgpd_infra(self, df: pl.DataFrame) -> pl.DataFrame:
        """Hash SHA-256 para colunas sensíveis (CNPJ/Razão Social)"""
        def _h(v):
            if v is None or str(v).strip() == "": return "ANONIMIZADO"
            return hashlib.sha256((str(v).upper().strip() + self.pepper).encode()).hexdigest()
        
        cols = [c for c in ["cnpj", "razao_social", "nome_fantasia"] if c in df.columns]
        return df.with_columns([pl.col(c).map_elements(_h, return_dtype=pl.Utf8) for c in cols])

    # ==========================================
    # WORKFLOW
    # ==========================================
    def download_r2(self):
        print("📥 Sincronizando Bronze...", flush=True)
        targets = ["SP_Faces_2022.zip", "sp-latest.osm.pbf", "Agregados_por_setores_basico", "CNPJ_SP_HISTORICO_LOTE_"]
        pag = self.s3.get_paginator('list_objects_v2')
        count = 0
        for p in pag.paginate(Bucket=self.bucket):
            for obj in p.get('Contents', []):
                key = obj['Key']
                if any(t in key for t in targets):
                    dest = os.path.join(self.bronze_dir, key.split('/')[-1])
                    self.s3.download_file(self.bucket, key, dest)
                    count += 1
        print(f"✅ {count} arquivos prontos.", flush=True)

    def processar(self):
        # 1. GEO (OSM + IBGE)
        print("🗺️ Malha Geográfica...", flush=True)
        try:
            zip_f = self._buscar_arquivo_seguro("SP_Faces_2022.zip", "IBGE Faces")
            if not glob.glob(f"{self.bronze_dir}/**/*.json", recursive=True):
                with zipfile.ZipFile(zip_f, 'r') as z: z.extractall(self.bronze_dir)
            
            json_files = glob.glob(f"{self.bronze_dir}/**/*.json", recursive=True)
            osm_pbf = self._buscar_arquivo_seguro("sp-latest.osm.pbf", "OSM PBF")
            
            sqls = []
            for f in json_files:
                f_path = str(f).replace("\\", "/")
                sqls.append(f"SELECT CAST(CD_FACE AS VARCHAR) as CD_FACE, trim(COALESCE(NM_TIP_LOG, '') || ' ' || COALESCE(NM_LOG, '')) as RUA, CAST(NULL AS VARCHAR) as NM_MUN, CAST(NULL AS VARCHAR) as NM_BAIRRO, CAST(ST_Y(ST_Centroid(geom)) AS FLOAT) as LAT, CAST(ST_X(ST_Centroid(geom)) AS FLOAT) as LON FROM ST_Read('{f_path}')")
            
            query = f"""
                SELECT CAST(osm_id AS VARCHAR) as CD_FACE, name as RUA, CAST(NULL AS VARCHAR) as NM_MUN, CAST(NULL AS VARCHAR) as NM_BAIRRO, CAST(ST_Y(ST_Centroid(geom)) AS FLOAT) as LAT, CAST(ST_X(ST_Centroid(geom)) AS FLOAT) as LON
                FROM ST_Read('{osm_pbf}', layer='lines', open_options=['INTERLEAVED_READING=YES']) 
                WHERE highway IS NOT NULL AND name IS NOT NULL
                UNION ALL {" UNION ALL ".join(sqls)}
            """
            df = self.con.execute(query).pl()
            df = self.normalizar_df(df)
            df = self.aplicar_h3_batch(df, "LAT", "LON")
            df = self.imputar_h3_infra(df, ["RUA", "NM_MUN", "NM_BAIRRO"])
            
            df.unique(subset=["RUA", "H3_INDEX"]).write_parquet(f"{self.prata_dir}/PRATA_MALHA_GEOGRAFICA_VIAS.parquet", compression="zstd")
            self.auditoria["GEO"] = {"STATUS": "OK", "COUNT": df.height}
        except Exception as e: self.auditoria["GEO"] = {"STATUS": "ERRO", "MSG": str(e)}

        # 2. SOCIAL
        print("👥 Malha Social...", flush=True)
        try:
            csv_f = self._buscar_arquivo_seguro("Agregados_por_setores_basico*.csv", "Censo")
            enc, sep = self._analisar_csv_dinamicamente(csv_f)
            df = pl.read_csv(csv_f, separator=sep, encoding=enc, null_values=["."], infer_schema_length=10000).filter(pl.col("CD_SETOR").cast(pl.Utf8).str.starts_with("35"))
            df = df.with_columns([
                pl.col("AREA_KM2").cast(pl.Utf8).str.replace(",", ".").cast(pl.Float32, strict=False),
                pl.col("v0001").cast(pl.UInt32, strict=False).fill_null(0).alias("POPULACAO"),
                pl.col("v0002").cast(pl.UInt32, strict=False).fill_null(0).alias("DOMICILIOS")
            ])
            df = self.normalizar_df(df)
            df.write_parquet(f"{self.prata_dir}/PRATA_MALHA_SOCIAL.parquet", compression="zstd")
            self.auditoria["SOCIAL"] = {"STATUS": "OK", "COUNT": df.height}
        except Exception as e: self.auditoria["SOCIAL"] = {"STATUS": "ERRO", "MSG": str(e)}

        # 3. INFRA (O Exterminador de Crash de Memória)
        print("🏗️ Malha Infra (Processamento Otimizado)...", flush=True)
        try:
            pqs = glob.glob(f"{self.bronze_dir}/**/CNPJ_SP_HISTORICO_LOTE_*.parquet", recursive=True)
            
            # SCAN (Lazy) em vez de READ (Eager) - Não carrega o arquivo todo na RAM
            lf = pl.scan_parquet(pqs)
            
            # Seleção de Colunas Precoce (Economiza RAM)
            lf = lf.select([
                "cnpj", "razao_social", "nome_fantasia", "lat", "lon", 
                "municipio", "bairro", "cnae_fiscal_principal", "situacao_cadastral"
            ]).filter(pl.col("lat").is_not_null())

            # Coletamos o DataFrame (O Polars vai usar streaming aqui se possível)
            df = lf.collect(streaming=True)
            
            # Processamento em memória otimizado
            df = self.aplicar_h3_batch(df, "lat", "lon")
            df = self.imputar_h3_infra(df, ["municipio", "bairro"])
            df = self.aplicar_lgpd_infra(df)
            df = self.normalizar_df(df)
            
            df.write_parquet(f"{self.prata_dir}/PRATA_MALHA_INFRA.parquet", compression="zstd")
            self.auditoria["INFRA"] = {"STATUS": "OK", "COUNT": df.height}
            print(f"   ✅ Infra Concluída: {df.height} empresas processadas.", flush=True)
        except Exception as e: self.auditoria["INFRA"] = {"STATUS": "ERRO", "MSG": str(e)}

    def finalizar(self):
        proj = os.getenv("BQ_PROJECT_ID", "safedriver").strip()
        with open(f"{self.prata_dir}/AUDITORIA_PRATA.json", "w") as f: json.dump(self.auditoria, f, indent=4)
        for f in os.listdir(self.prata_dir):
            if f.endswith((".parquet", ".json")):
                self.s3.upload_file(os.path.join(self.prata_dir, f), self.bucket, f"{proj}/datalake/prata/malhas/{f}")
        print("\n📊 PIPELINE FINALIZADO:", json.dumps(self.auditoria, indent=4), flush=True)

if __name__ == "__main__":
    app = ArquitetoSafeDriverPrata()
    app.download_r2()
    app.processar()
    app.finalizar()
