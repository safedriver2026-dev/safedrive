import os
import boto3
import duckdb
import polars as pl
import h3
import zipfile
import json
import glob
import re
import csv
import io
import shutil
import time
import requests
import unicodedata
from datetime import datetime
from botocore.config import Config

class ArquitetoSafeDriverPrata:
    def __init__(self):
        self.H3_RES = 9 
        self.bronze_dir = "./data_raw"
        self.prata_dir = "./data_prata"
        self.temp_extract_dir = "./data_raw/extracted_json"
        
        os.makedirs(self.bronze_dir, exist_ok=True)
        os.makedirs(self.prata_dir, exist_ok=True)
        os.makedirs(self.temp_extract_dir, exist_ok=True)
        
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
        
        self.webhook_url = os.getenv("DISCORD_SUCESSO")
        self.con = duckdb.connect(database=':memory:')
        self.con.execute("PRAGMA memory_limit='5GB';")
        try:
            self.con.execute("INSTALL spatial; LOAD spatial;")
        except:
            pass

    def _limpar_tabela_toda(self, df):
        """Limpeza bruta em todas as colunas de texto da tabela."""
        cols_texto = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Utf8]
        if not cols_texto:
            return df
        
        return df.with_columns([
            pl.col(c)
            .str.to_uppercase()
            .str.strip_chars()
            .str.replace_all(r"[ÁÀÂÃÄ]", "A")
            .str.replace_all(r"[ÉÈÊË]", "E")
            .str.replace_all(r"[ÍÌÎÏ]", "I")
            .str.replace_all(r"[ÓÒÔÕÖ]", "O")
            .str.replace_all(r"[ÚÙÛÜ]", "U")
            .str.replace_all(r"[Ç]", "C")
            .str.replace_all(r"[^A-Z0-9\s_]", " ")
            .str.replace_all(r"\s+", " ")
            .fill_null("DESCONHECIDO")
            .alias(c)
            for c in cols_texto
        ])

    def _limpar_cd_setor(self, coluna):
        return pl.col(coluna).cast(pl.Utf8).str.replace(r"\.0$", "").str.replace_all(r"\D", "").str.slice(0, 15)

    def download_r2(self):
        print("Sincronizando arquivos...")
        targets = ["SP_Faces_2022.zip", "SP_bairros_CD2022", "Agregados_por_setores_basico", "CNPJ_SP_HISTORICO_LOTE_"]
        pag = self.s3.get_paginator('list_objects_v2')
        for p in pag.paginate(Bucket=self.bucket):
            for obj in p.get('Contents', []):
                key = obj['Key']
                if any(t in key for t in targets):
                    dest = os.path.join(self.bronze_dir, key.split('/')[-1])
                    if not os.path.exists(dest):
                        self.s3.download_file(self.bucket, key, dest)

    def upload_r2(self):
        print("Enviando para o R2...")
        arquivos = glob.glob(f"{self.prata_dir}/*")
        for filepath in arquivos:
            filename = os.path.basename(filepath)
            s3_key = f"datalake/prata/malha_trusted/{filename}"
            self.s3.upload_file(filepath, self.bucket, s3_key)

    def processar(self):
        tempo_inicio = time.time()
        try:
            # 1. Censo (Corrigido Separador e Encoding)
            csv_f = glob.glob(f"**/Agregados_por_setores_basico*.csv", recursive=True)[0]
            # O SEGREDO ESTÁ AQUI: separator=";" e encoding="iso-8859-1"
            df_censo = pl.read_csv(csv_f, separator=";", encoding="iso-8859-1", infer_schema_length=0, ignore_errors=True)
            df_censo.columns = [c.strip().upper() for c in df_censo.columns]
            
            df_censo = df_censo.select([
                self._limpar_cd_setor("CD_SETOR").alias("CD_SETOR"),
                pl.col("NM_MUN").alias("CID_CENSO") if "NM_MUN" in df_censo.columns else pl.lit("DESCONHECIDO").alias("CID_CENSO"),
                pl.col("V0001").cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0).alias("CENSO_POPULACAO"),
                pl.col("V0002").cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0).alias("CENSO_RENDA")
            ])
            df_censo = self._limpar_tabela_toda(df_censo)

            # 2. Ruas (Faces JSON)
            zip_f = glob.glob(f"**/SP_Faces_2022.zip", recursive=True)[0]
            vias_list = []
            with zipfile.ZipFile(zip_f, 'r') as z:
                json_files = [f for f in z.namelist() if f.endswith('.json')]
                for f in json_files:
                    z.extract(f, self.temp_extract_dir)
                    path_json = os.path.join(self.temp_extract_dir, f).replace('\\','/')
                    sql = f"SELECT CD_SETOR, trim(COALESCE(NM_TIP_LOG, '') || ' ' || COALESCE(NM_LOG, '')) as RUA, ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON, TRY_CAST(TOT_RES AS FLOAT) as TOT_RES FROM ST_Read('{path_json}')"
                    vias_list.append(self.con.execute(sql).pl())
                    shutil.rmtree(os.path.join(self.temp_extract_dir, f.split('/')[0]), ignore_errors=True)

            df_ruas = pl.concat(vias_list)
            df_ruas = df_ruas.with_columns(self._limpar_cd_setor("CD_SETOR").alias("CD_SETOR"))
            df_ruas = self._limpar_tabela_toda(df_ruas)

            # 3. Cruzamento e H3
            df_ruas_h3 = df_ruas.join(df_censo, on="CD_SETOR", how="left").fill_null(0)
            df_ruas_h3 = df_ruas_h3.with_columns(
                pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], self.H3_RES) for x in s])).alias("H3_INDEX")
            )

            # 4. Join Espacial (Bairros SHP)
            df_h3_coords = df_ruas_h3.group_by("H3_INDEX").agg([
                pl.col("LAT").first().alias("LAT"), pl.col("LON").first().alias("LON"),
                pl.col("CID_CENSO").first().alias("CID_CENSO")
            ])
            self.con.register("tabela_h3", df_h3_coords.to_arrow())
            shp_path = glob.glob(f"**/*bairros_CD2022*.shp", recursive=True)[0].replace("\\", "/")
            sql_spatial = f"""
                SELECT h3.H3_INDEX, 
                COALESCE(ibge.NM_MUN, h3.CID_CENSO, 'DESCONHECIDO') AS CIDADE,
                COALESCE(ibge.NM_BAIRRO, 'DESCONHECIDO') AS BAIRRO
                FROM tabela_h3 h3 LEFT JOIN ST_Read('{shp_path}') ibge ON ST_Contains(ibge.geom, ST_Point(h3.LON, h3.LAT))
            """
            df_geo = self.con.execute(sql_spatial).pl()
            df_geo = self._limpar_tabela_toda(df_geo)

            # 5. Exportação (TUDO LIMPO)
            df_vias_final = df_ruas_h3.drop(["CID_CENSO"]).join(df_geo, on="H3_INDEX", how="left")
            df_vias_final = self._limpar_tabela_toda(df_vias_final)
            df_vias_final.write_parquet(f"{self.prata_dir}/PRATA_MALHA_SOCIAL_H3.parquet")

            # 6. Infraestrutura (CNAEs)
            infra_files = glob.glob(f"**/CNPJ_SP_HISTORICO_LOTE_*.parquet", recursive=True)
            if infra_files:
                df_infra = pl.scan_parquet(infra_files).filter(pl.col("lat").is_not_null()) \
                    .with_columns(pl.col("cnae_fiscal_principal").cast(pl.Utf8).str.slice(0, 2).alias("CNAE_DIV")) \
                    .collect()
                df_infra = df_infra.with_columns(
                    pl.struct(["lat", "lon"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["lat"], x["lon"], self.H3_RES) for x in s])).alias("H3_INDEX")
                )
                df_pivot = df_infra.group_by(["H3_INDEX", "CNAE_DIV"]).len() \
                    .pivot(values="len", index="H3_INDEX", on="CNAE_DIV").fill_null(0)
                df_pivot = df_pivot.rename({c: f"INFRA_DIV_{c}" for c in df_pivot.columns if c != "H3_INDEX"})
                df_infra_final = df_pivot.join(df_geo, on="H3_INDEX", how="left")
                df_infra_final = self._limpar_tabela_toda(df_infra_final)
                df_infra_final.write_parquet(f"{self.prata_dir}/PRATA_MALHA_INFRA_AGREGADA.parquet")

            self.upload_r2()
            print(f"Malha Prata OK em {time.time() - tempo_inicio:.2f}s")

        finally:
            self.con.close()

if __name__ == "__main__":
    app = ArquitetoSafeDriverPrata()
    app.download_r2()
    app.processar()
