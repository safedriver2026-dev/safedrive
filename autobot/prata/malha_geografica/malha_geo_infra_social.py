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
import csv
from datetime import datetime
from botocore.config import Config

class ArquitetoSafeDriverPrata:
    def __init__(self):
        self.H3_RES = 9 
        self.bronze_dir = "./data_raw"
        self.prata_dir = "./data_prata"
        self.temp_dir = "./temp_duckdb"
        
        os.makedirs(self.bronze_dir, exist_ok=True)
        os.makedirs(self.prata_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)
        
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

        self.con = duckdb.connect(database=':memory:')
        self.con.execute(f"PRAGMA memory_limit='4GB';")
        self.con.execute(f"PRAGMA temp_directory='{self.temp_dir}';")
        try: self.con.execute("INSTALL spatial; LOAD spatial;")
        except: pass 
        
        self.auditoria = {"DATA_EXECUCAO": str(datetime.now())}

    def _normalizar_texto(self, valor):
        if valor is None or str(valor).upper() in ["NULL", "NAN", ".", "", "NONE"]: 
            return "NAO INFORMADO"
        texto = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^a-zA-Z0-9\s]', '', texto).upper().strip()

    def _buscar_arquivo_seguro(self, padrao, nome):
        arqs = glob.glob(f"{self.bronze_dir}/**/{padrao}", recursive=True)
        if not arqs: raise FileNotFoundError(f"{nome} ({padrao}) ausente!")
        return arqs[0]

    def processar(self):
        print("Processando Malha Geográfica Hierárquica...", flush=True)
        try:
            # 1. Processar Malha Social Primeiro (Referência de Nomes)
            print("--- Criando Referência de Setores (Social) ---")
            csv_f = self._buscar_arquivo_seguro("Agregados_por_setores_basico*.csv", "Censo")
            df_social = pl.read_csv(csv_f, separator=";", encoding="latin1", infer_schema_length=10000)
            
            # Criamos o mapa CD_SETOR -> NM_MUN, NM_BAIRRO
            df_ref_nomes = df_social.select([
                pl.col("CD_SETOR").cast(pl.Utf8),
                pl.col("NM_MUN").alias("CIDADE"),
                pl.col("NM_BAIRRO").alias("BAIRRO")
            ]).with_columns([
                pl.col("CIDADE").map_elements(self._normalizar_texto, return_dtype=pl.Utf8),
                pl.col("BAIRRO").map_elements(self._normalizar_texto, return_dtype=pl.Utf8)
            ])

            # 2. Processar Faces de Logradouros (RUA + CD_SETOR + GEO)
            print("--- Extraindo Ruas e Geometrias ---")
            zip_f = self._buscar_arquivo_seguro("SP_Faces_2022.zip", "IBGE Faces")
            if not glob.glob(f"{self.bronze_dir}/**/*.json", recursive=True):
                with zipfile.ZipFile(zip_f, 'r') as z: z.extractall(self.bronze_dir)
            
            json_files = glob.glob(f"{self.bronze_dir}/**/*.json", recursive=True)
            sqls = []
            for f in json_files:
                f_path = str(f).replace("\\", "/")
                # Note que aqui pegamos o CD_SETOR para fazer o join depois
                sqls.append(f"""
                    SELECT 
                        CAST(CD_SETOR AS VARCHAR) as CD_SETOR,
                        trim(COALESCE(NM_TIP_LOG, '') || ' ' || COALESCE(NM_LOG, '')) as RUA, 
                        CAST(ST_Y(ST_Centroid(geom)) AS FLOAT) as LAT, 
                        CAST(ST_X(ST_Centroid(geom)) AS FLOAT) as LON, 
                        TRY_CAST(TOT_RES AS FLOAT) as TOT_RES 
                    FROM ST_Read('{f_path}')
                """)
            
            df_faces = self.con.execute(" UNION ALL ".join(sqls)).pl()
            
            # 3. Join para obter nomes de Cidade e Bairro
            df_vias_completo = df_faces.join(df_ref_nomes, on="CD_SETOR", how="left").with_columns([
                pl.col("CIDADE").fill_null("NAO INFORMADO"),
                pl.col("BAIRRO").fill_null("NAO INFORMADO"),
                pl.col("RUA").map_elements(self._normalizar_texto, return_dtype=pl.Utf8)
            ])

            # 4. Geração de H3
            df_vias_completo = df_vias_completo.with_columns(
                pl.struct(["LAT", "LON"]).map_batches(
                    lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], self.H3_RES) for x in s])
                ).alias("H3_INDEX")
            )

            # 5. Construção da Hierarquia
            df_hierarquico = (
                df_vias_completo.group_by(["CIDADE", "BAIRRO", "RUA"])
                .agg(pl.col("H3_INDEX").unique().alias("H3_LIST"))
                .group_by(["CIDADE", "BAIRRO"])
                .agg(pl.struct([pl.col("RUA"), pl.col("H3_LIST")]).alias("LOGRADOUROS"))
                .group_by("CIDADE")
                .agg(pl.struct([pl.col("BAIRRO"), pl.col("LOGRADOUROS")]).alias("BAIRROS"))
            )

            # Salvar Malha de Vias
            df_hierarquico.write_parquet(f"{self.prata_dir}/PRATA_MALHA_GEOGRAFICA_VIAS.parquet", compression="zstd")

            # Salvar Micro-População (Flat)
            df_micro_pop = df_vias_completo.group_by("H3_INDEX").agg(pl.sum("TOT_RES").fill_null(0).alias("MICRO_POPULACAO_H3"))
            df_micro_pop.write_parquet(f"{self.prata_dir}/PRATA_MALHA_MICRO_POPULACAO.parquet", compression="zstd")

            # 6. Salvar Social Macro (Garantindo persistência)
            df_social_final = df_social.select([
                pl.col("CD_SETOR").cast(pl.Utf8),
                pl.col("NM_MUN"), pl.col("NM_BAIRRO"),
                pl.col("v0001").alias("v0001"), pl.col("v0002").alias("v0002")
            ])
            df_social_final.write_parquet(f"{self.prata_dir}/PRATA_MALHA_SOCIAL.parquet", compression="zstd")

            self.auditoria["MALHA_GEOGRAFICA"] = "OK"
        except Exception as e:
            self.auditoria["MALHA_GEOGRAFICA"] = str(e)
            print(f"Erro Fatal: {e}")

    def finalizar(self):
        r2_dest_path = "datalake/prata/malha_trusted"
        with open(f"{self.prata_dir}/AUDITORIA_PRATA.json", "w") as f: 
            json.dump(self.auditoria, f, indent=4)
        for f in os.listdir(self.prata_dir):
            if f.endswith((".parquet", ".json")):
                self.s3.upload_file(os.path.join(self.prata_dir, f), self.bucket, f"{r2_dest_path}/{f}")
        print("Pipeline Prata Finalizado!")

if __name__ == "__main__":
    app = ArquitetoSafeDriverPrata()
    app.processar()
    app.finalizar()
