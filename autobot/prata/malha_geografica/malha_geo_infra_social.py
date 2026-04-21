import os
import boto3
import duckdb
import polars as pl
import h3
import unicodedata
import zipfile
import json
import glob
import re
from datetime import datetime
from botocore.config import Config

class ArquitetoSafeDriverPrata:
    def __init__(self):
        # Configurações de Ambiente
        self.H3_RES = 9 
        self.bronze_dir = "./data_raw"
        self.prata_dir = "./data_prata"
        os.makedirs(self.bronze_dir, exist_ok=True)
        os.makedirs(self.prata_dir, exist_ok=True)
        
        # Conexão Boto3 (Limpando espaços e quebras de linha preventivamente)
        self.bucket = os.getenv("R2_BUCKET_NAME").strip()
        endpoint = os.getenv("R2_ENDPOINT_URL").strip().rstrip('/')
        if endpoint.endswith(f"/{self.bucket}"):
            endpoint = endpoint[: -len(f"/{self.bucket}")]
            
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint,
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY").strip(),
            config=Config(signature_version='s3v4')
        )
        
        self.con = duckdb.connect(database=':memory:')
        self.con.execute("INSTALL spatial; LOAD spatial;")
        
        self.auditoria = {"DATA_EXECUCAO": str(datetime.now()), "CAMADAS": {}}

    def download_r2(self):
        """Busca e baixa os arquivos necessários ignorando pastas fantasmas."""
        print("📥 Buscando arquivos na Bronze via Boto3...")
        paginator = self.s3.get_paginator('list_objects_v2')
        # Filtramos para não listar o bucket inteiro se for muito grande
        targets = ["SP_Faces_2022.zip", "Agregados_por_setores_basico", "CNPJ_SP_HISTORICO_LOTE_"]
        
        for page in paginator.paginate(Bucket=self.bucket):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if any(t in key for t in targets):
                    filename = key.split('/')[-1]
                    dest = os.path.join(self.bronze_dir, filename)
                    print(f"   ⬇️  Baixando: {key} -> {dest}")
                    self.s3.download_file(self.bucket, key, dest)

    def upload_r2(self):
        """Sobe os resultados para a pasta do projeto."""
        projeto = os.getenv("BQ_PROJECT_ID", "safedriver").strip()
        print(f"📤 Subindo arquivos para datalake/prata/ no projeto {projeto}...")
        for file in os.listdir(self.prata_dir):
            local_path = os.path.join(self.prata_dir, file)
            r2_path = f"{projeto}/datalake/prata/malhas/{file}"
            self.s3.upload_file(local_path, self.bucket, r2_path)
            print(f"   ✅ Enviado: {file}")

    def _normalizar(self, valor):
        if not valor or str(valor).upper() in ["NULL", "NAN", "."]: return "NAO INFORMADO"
        t = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^a-zA-Z0-9\s]', '', t).upper().strip()

    def processar(self):
        # 1. GEO
        print("🗺️ Processando Malha Geográfica...")
        try:
            zip_f = glob.glob(f"{self.bronze_dir}/**/SP_Faces_2022.zip", recursive=True)[0]
            with zipfile.ZipFile(zip_f, 'r') as z: z.extractall(self.bronze_dir)
            query = f"SELECT CD_FACE, trim(NM_TIP_LOG || ' ' || NM_LOG) as RUA, ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON FROM ST_Read('{self.bronze_dir}/**/*.json') WHERE NM_LOG IS NOT NULL"
            df = self.con.execute(query).pl().with_columns([
                pl.col("RUA").map_elements(self._normalizar, return_dtype=pl.Utf8).cast(pl.Categorical),
                pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], self.H3_RES) for x in s])).alias("H3_INDEX").cast(pl.Categorical),
                pl.col("LAT").cast(pl.Float32), pl.col("LON").cast(pl.Float32)
            ]).unique(subset=["CD_FACE"])
            df.write_parquet(f"{self.prata_dir}/PRATA_MALHA_GEOGRAFICA_VIAS.parquet", compression="zstd")
            self.auditoria["CAMADAS"]["GEO"] = {"STATUS": "OK", "LINHAS": df.height}
        except Exception as e: self.auditoria["CAMADAS"]["GEO"] = {"STATUS": "ERRO", "MSG": str(e)}

        # 2. SOCIAL
        print("👥 Processando Malha Social...")
        try:
            csv_f = glob.glob(f"{self.bronze_dir}/**/Agregados_por_setores_basico*.csv", recursive=True)[0]
            df = pl.read_csv(csv_f, separator=";", null_values=["."], infer_schema_length=10000, schema_overrides={"CD_SETOR": pl.Utf8, "CD_MUN": pl.Utf8}).filter(pl.col("CD_SETOR").str.starts_with("35"))
            # Seleção de colunas baseada no seu dicionário
            cols = ["CD_SETOR", "NM_MUN", "NM_BAIRRO", "AREA_KM2", "CD_TIPO", "v0001", "v0002"]
            df = df.select([c for c in cols if c in df.columns]).with_columns([
                pl.col("AREA_KM2").str.replace(",", ".").cast(pl.Float32, strict=False),
                pl.col("v0001").cast(pl.UInt32, strict=False).fill_null(0).alias("POPULACAO"),
                pl.col("v0002").cast(pl.UInt32, strict=False).fill_null(0).alias("DOMICILIOS")
            ])
            df.write_parquet(f"{self.prata_dir}/PRATA_MALHA_SOCIAL.parquet", compression="zstd")
            self.auditoria["CAMADAS"]["SOCIAL"] = {"STATUS": "OK", "LINHAS": df.height}
        except Exception as e: self.auditoria["CAMADAS"]["SOCIAL"] = {"STATUS": "ERRO", "MSG": str(e)}

        # 3. INFRA
        print("🏗️ Processando Malha Infra/Comércio...")
        try:
            pqs = glob.glob(f"{self.bronze_dir}/**/CNPJ_SP_HISTORICO_LOTE_*.parquet", recursive=True)
            df = pl.read_parquet(pqs).drop_nulls(subset=["lat", "lon"]).with_columns([
                pl.struct(["lat", "lon"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["lat"], x["lon"], self.H3_RES) for x in s])).alias("H3_INDEX").cast(pl.Categorical),
                pl.col("lat").cast(pl.Float32), pl.col("lon").cast(pl.Float32)
            ])
            df.write_parquet(f"{self.prata_dir}/PRATA_MALHA_INFRA.parquet", compression="zstd")
            self.auditoria["CAMADAS"]["INFRA"] = {"STATUS": "OK", "LINHAS": df.height}
        except Exception as e: self.auditoria["CAMADAS"]["INFRA"] = {"STATUS": "ERRO", "MSG": str(e)}

    def finalizar(self):
        with open(f"{self.prata_dir}/AUDITORIA_PRATA.json", "w") as f:
            json.dump(self.auditoria, f, indent=4)
        print("\n📊 AUDITORIA:\n", json.dumps(self.auditoria, indent=4))

if __name__ == "__main__":
    app = ArquitetoSafeDriverPrata()
    app.download_r2()
    app.processar()
    app.finalizar()
    app.upload_r2()
