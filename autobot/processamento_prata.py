import polars as pl
import boto3
import io
import os
import h3
import logging
from datetime import datetime
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        self.h3_resolution = 8

    def executar_prata(self, ano):
        path_bronze = f"datalake/bronze/ssp_raw_{ano}.xlsx"
        path_geo = "datalake/base_geografica/safedriver_geo_base_sp_h3_8.parquet"
        path_prata = f"datalake/prata/ssp_consolidada_{ano}.parquet"

        try:
            try:
                self.s3.head_object(Bucket=self.bucket, Key=path_prata)
                logger.info(f"PRATA: O ano {ano} ja esta consolidado.")
                return True
            except:
                pass

            logger.info(f"PRATA: Lendo dados brutos e base geografica...")
            
            resp_ssp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            lf_ssp = pl.read_excel(io.BytesIO(resp_ssp['Body'].read()))

            resp_geo = self.s3.get_object(Bucket=self.bucket, Key=path_geo)
            lf_geo = pl.read_parquet(io.BytesIO(resp_geo['Body'].read()))

            lf_ssp = lf_ssp.filter(
                (pl.col("LATITUDE").is_not_null()) & 
                (pl.col("LONGITUDE").is_not_null()) &
                (pl.col("LATITUDE") < -20) & (pl.col("LATITUDE") > -26)
            )

            df_pandas = lf_ssp.to_pandas()
            df_pandas['h3_index'] = df_pandas.apply(
                lambda row: h3.latlng_to_cell(row['LATITUDE'], row['LONGITUDE'], self.h3_resolution), 
                axis=1
            )
            lf_ssp = pl.from_pandas(df_pandas)

            lf_crimes = self._agregar_crimes(lf_ssp)

            lf_final = lf_crimes.join(lf_geo, on="h3_index", how="inner")

            lf_final = self._processar_indicadores(lf_final, ano)

            buffer = io.BytesIO()
            lf_final.write_parquet(buffer)
            buffer.seek(0)

            self.s3.put_object(
                Bucket=self.bucket,
                Key=path_prata,
                Body=buffer.getvalue()
            )

            logger.info(f"PRATA: Processamento concluido para {ano}.")
            return True

        except Exception as e:
            logger.error(f"Erro na Camada Prata ({ano}): {e}")
            raise e

    def _agregar_crimes(self, lf):
        return lf.group_by("h3_index").agg([
            pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Veículo|Automóvel")).count().alias("TOTAL_CRIMES_MOTORISTA"),
            pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Celular|Transeunte")).count().alias("TOTAL_CRIMES_PEDESTRE"),
            pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Motocicleta")).count().alias("TOTAL_CRIMES_MOTOCICLISTA")
        ])

    def _processar_indicadores(self, lf, ano):
        return lf.with_columns([
            pl.lit(ano).alias("ANO_REFERENCIA"),
            (pl.col("TOTAL_GERAL") - pl.col("TOTAL_RESIDENCIAL")).alias("TOTAL_NAO_RESIDENCIAL"),
            (pl.col("TOTAL_RESIDENCIAL") / pl.col("TOTAL_GERAL")).alias("INDICE_RESIDENCIAL"),
            (pl.col("TOTAL_GERAL") / pl.col("AREA_HEXAGONO")).alias("DENSIDADE_ENDERECOS")
        ]).fill_null(0)

if __name__ == "__main__":
    prata = ProcessamentoPrata()
    ano_inicio = 2022
    for a in range(ano_inicio, datetime.now().year + 1):
        prata.executar_prata(a)
