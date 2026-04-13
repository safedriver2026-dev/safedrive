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
        path_prata = f"datalake/prata/ssp_consolidada_{ano}.parquet"

        try:
            try:
                self.s3.head_object(Bucket=self.bucket, Key=path_prata)
                logger.info(f"PRATA: O ano {ano} ja esta consolidado.")
                return True
            except:
                pass

            logger.info(f"PRATA: Lendo arquivo Excel de {ano}...")
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            lf = pl.read_excel(io.BytesIO(resp['Body'].read()))

            lf = lf.filter(
                (pl.col("LATITUDE").is_not_null()) & 
                (pl.col("LONGITUDE").is_not_null()) &
                (pl.col("LATITUDE") < -20) & (pl.col("LATITUDE") > -26)
            )

            df_pandas = lf.to_pandas()
            df_pandas['h3_index'] = df_pandas.apply(
                lambda row: h3.geo_to_h3(row['LATITUDE'], row['LONGITUDE'], self.h3_resolution), 
                axis=1
            )
            lf = pl.from_pandas(df_pandas)

            lf = self._aplicar_engenharia_atributos(lf, ano)

            buffer = io.BytesIO()
            lf.write_parquet(buffer)
            buffer.seek(0)

            self.s3.put_object(
                Bucket=self.bucket,
                Key=path_prata,
                Body=buffer.getvalue()
            )

            logger.info(f"PRATA: Camada consolidada para {ano} salva com sucesso.")
            return True

        except Exception as e:
            logger.error(f"Falha no processamento Prata de {ano}: {e}")
            raise e

    def _aplicar_engenharia_atributos(self, lf, ano):
        return lf.with_columns([
            pl.lit(ano).alias("ANO_REFERENCIA"),
            (pl.col("TOTAL_GERAL") - pl.col("TOTAL_RESIDENCIAL")).alias("TOTAL_NAO_RESIDENCIAL"),
            (pl.col("TOTAL_RESIDENCIAL") / pl.col("TOTAL_GERAL")).alias("INDICE_RESIDENCIAL"),
            (pl.col("TOTAL_GERAL") / pl.col("AREA_HEXAGONO")).alias("DENSIDADE_ENDERECOS")
        ])

if __name__ == "__main__":
    prata = ProcessamentoPrata()
    ano_inicio = 2022
    ano_fim = datetime.now().year
    
    for a in range(ano_inicio, ano_fim + 1):
        prata.executar_prata(a)
