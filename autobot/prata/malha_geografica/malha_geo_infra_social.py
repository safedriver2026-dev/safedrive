import os
import boto3
import duckdb
import polars as pl
import h3
import zipfile
import glob
import re
import shutil
import time
import requests
import io
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
        self.con.execute("PRAGMA memory_limit='5GB'; PRAGMA threads=8;")
        try:
            self.con.execute("INSTALL spatial; LOAD spatial;")
        except:
            pass

    def _limpar_tabela_toda(self, df):
        """Passa o trator em todas as colunas de texto: sem acento, caixa alta, sem lixo."""
        cols_texto = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Utf8]
        if not cols_texto: return df
        
        return df.with_columns([
            pl.col(c)
            .str.to_uppercase()
            .str.strip_chars()
            .str.replace_all(r"[ÁÀÂÃÄ]", "A")
            .str.replace_all(r"[ÉÈÊË]", "E")
            .str.replace_all(r"[ÍÌÎÏ]", "I")
            .str.replace_all(r"[ÓÒÔÕÖ]", "O")
            .str.replace_all(r"[ÚÙÛÜ]", "U")
            .str.replace_all(r"Ç", "C")
            .str.replace_all(r"[^A-Z0-9\s_]", " ")
            .str.replace_all(r"\s+", " ")
            .fill_null("DESCONHECIDO")
            .alias(c)
            for c in cols_texto
        ])

    def _limpar_cd_setor(self, coluna):
        return pl.col(coluna).cast(pl.Utf8).str.replace(r"\.0$", "").str.replace_all(r"\D", "").str.slice(0, 15)

    def download_r2(self):
        print("Sincronizando Bronze...")
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
        print("Enviando Camada Prata para o R2...")
        for f in glob.glob(f"{self.prata_dir}/*"):
            self.s3.upload_file(f, self.bucket, f"datalake/prata/malha_trusted/{os.path.basename(f)}")

    def processar(self):
        tempo_inicio = time.time()
        print("Iniciando Processamento Otimizado...")
        try:
            # 1. CENSO (Correção de Encoding e Separador)
            csv_f = glob.glob(f"**/Agregados_por_setores_basico*.csv", recursive=True)[0]
            df_censo = pl.read_csv(csv_f, separator=";", encoding="iso-8859-1", infer_schema_length=0)
            df_censo.columns = [c.strip().upper() for c in df_censo.columns]
            
            df_censo = df_censo.select([
                self._limpar_cd_setor("CD_SETOR").alias("CD_SETOR"),
                pl.col("NM_MUN").alias("CID_CENSO") if "NM_MUN" in df_censo.columns else pl.lit("DESCONHECIDO").alias("CID_CENSO"),
                pl.col("V0001").cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0).alias("CENSO_POPULACAO"),
                pl.col("V0002").cast(pl.Utf8).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0).alias("CENSO_RENDA")
            ]).pipe(self._limpar_tabela_toda)

            # 2. RUAS (Extração em Lote)
            zip_f = glob.glob(f"**/SP_Faces_2022.zip", recursive=True)[0]
            vias_list = []
            with zipfile.ZipFile(zip_f, 'r') as z:
                json_files = [f for f in z.namelist() if f.endswith('.json')]
                for i in range(0, len(json_files), 100):
                    batch = json_files[i:i+100]
                    for f in batch: z.extract(f, self.temp_extract_dir)
                    sql = f"SELECT CD_SETOR, trim(COALESCE(NM_TIP_LOG, '') || ' ' || COALESCE(NM_LOG, '')) as RUA, ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON, TRY_CAST(TOT_RES AS FLOAT) as TOT_RES FROM ST_Read('{self.temp_extract_dir}/*/*.json')"
                    vias_list.append(self.con.execute(sql).pl())
                    for f in batch: shutil.rmtree(os.path.join(self.temp_extract_dir, f.split('/')[0]), ignore_errors=True)

            df_ruas = pl.concat(vias_list).with_columns(self._limpar_cd_setor("CD_SETOR")).pipe(self._limpar_tabela_toda)

            # 3. H3 E SPATIAL JOIN OTIMIZADO
            # Aqui esta o segredo: calculamos o bairro apenas para H3 UNICOS, nao para milhoes de ruas.
            df_ruas_h3 = df_ruas.join(df_censo, on="CD_SETOR", how="left").fill_null(0)
            df_ruas_h3 = df_ruas_h3.with_columns(
                pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], self.H3_RES) for x in s])).alias("H3_INDEX")
            )

            df_h3_unicos = df_ruas_h3.group_by("H3_INDEX").agg([
                pl.col("LAT").first().alias("LAT"), pl.col("LON").first().alias("LON"),
                pl.col("CID_CENSO").first().alias("CID_CENSO")
            ])
            
            self.con.register("tabela_h3", df_h3_unicos.to_arrow())
            shp_path = glob.glob(f"**/*bairros_CD2022*.shp", recursive=True)[0].replace("\\", "/")
            
            df_geo_ref = self.con.execute(f"""
                SELECT h3.H3_INDEX, 
                COALESCE(ibge.NM_MUN, h3.CID_CENSO, 'DESCONHECIDO') AS CIDADE,
                COALESCE(ibge.NM_BAIRRO, 'DESCONHECIDO') AS BAIRRO
                FROM tabela_h3 h3 LEFT JOIN ST_Read('{shp_path}') ibge ON ST_Contains(ibge.geom, ST_Point(h3.LON, h3.LAT))
            """).pl().pipe(self._limpar_tabela_toda)

            # 4. EXPORTAÇÕES (Sempre limpando no final)
            # Social H3
            df_vias_final = df_ruas_h3.drop("CID_CENSO").join(df_geo_ref, on="H3_INDEX", how="left").pipe(self._limpar_tabela_toda)
            df_vias_final.group_by("H3_INDEX").agg([
                pl.sum("TOT_RES").alias("MICRO_POPULACAO_FACES"),
                pl.mean("CENSO_POPULACAO").alias("CENSO_MEDIA_V0001"),
                pl.mean("CENSO_RENDA").alias("CENSO_MEDIA_V0002")
            ]).write_parquet(f"{self.prata_dir}/PRATA_MALHA_SOCIAL_H3.parquet")

            # Infraestrutura (CNAE)
            infra_files = glob.glob(f"**/CNPJ_SP_HISTORICO_LOTE_*.parquet", recursive=True)
            if infra_files:
                df_infra = pl.scan_parquet(infra_files).filter(pl.col("lat").is_not_null()) \
                    .with_columns(pl.col("cnae_fiscal_principal").cast(pl.Utf8).str.slice(0, 2).alias("CNAE_DIV")) \
                    .collect() \
                    .with_columns(pl.struct(["lat", "lon"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["lat"], x["lon"], self.H3_RES) for x in s])).alias("H3_INDEX")) \
                    .group_by(["H3_INDEX", "CNAE_DIV"]).len() \
                    .pivot(values="len", index="H3_INDEX", on="CNAE_DIV").fill_null(0)
                
                df_infra = df_infra.rename({c: f"INFRA_DIV_{c}" for c in df_infra.columns if c != "H3_INDEX"}) \
                    .join(df_geo_ref, on="H3_INDEX", how="left").pipe(self._limpar_tabela_toda)
                df_infra.write_parquet(f"{self.prata_dir}/PRATA_MALHA_INFRA_AGREGADA.parquet")

            self.upload_r2()
            print(f"✅ Malha Prata Otimizada OK em {time.time() - tempo_inicio:.2f}s")

        finally:
            self.con.close()
            if os.path.exists(self.temp_extract_dir): shutil.rmtree(self.temp_extract_dir)

if __name__ == "__main__":
    app = ArquitetoSafeDriverPrata()
    app.download_r2()
    app.processar()
