import polars as pl
import h3
import boto3
from botocore.config import Config
import io
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )

    def executar_todos_os_anos(self):
        logger.info("PRATA: Iniciando processamento de tendências (Leitura Direta)...")
        
        ano_atual = datetime.now().year
        anos_para_processar = list(range(2022, ano_atual + 1))
        
        # Ignora o ListObjectsV2 e vai direto nos arquivos que sabemos que existem
        for ano in anos_para_processar:
            self.processar_ano_com_delta(ano)

    def processar_ano_com_delta(self, ano):
        path_bronze = f"safedriver/datalake/bronze/ssp_raw_{ano}.xlsx"
        path_prata_atual = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
        path_prata_anterior = f"safedriver/datalake/prata/ssp_consolidada_{ano - 1}.parquet"

        try:
            # Tenta baixar o arquivo direto. Se não existir, cai no except e ignora em silêncio.
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            logger.info(f"--- Refinando Ciclo: {ano} ---")
            
            df = pl.read_excel(io.BytesIO(resp['Body'].read()))
            df = df.rename({c: c.replace("Ç", "C").replace("Ã", "A") for c in df.columns})

            df = df.with_columns([
                pl.col("LATITUDE").cast(pl.Float64, strict=False),
                pl.col("LONGITUDE").cast(pl.Float64, strict=False)
            ]).filter(
                pl.col("LATITUDE").is_not_null() & (pl.col("LATITUDE") != 0)
            )

            if df.is_empty():
                logger.warning(f"PRATA: Dados de {ano} vazios após limpeza geográfica.")
                return

            h3_list = [h3.geo_to_h3(lat, lon, 8) for lat, lon in zip(df["LATITUDE"], df["LONGITUDE"])]
            df = df.with_columns(pl.Series("H3_INDEX", h3_list))

            df_atual = df.group_by(["H3_INDEX"]).agg([
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Roubo|Furto|Veículo|Carga")).count().alias("TOTAL_CRIMES_MOTORISTA"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Celular|Transeunte|Pessoa")).count().alias("TOTAL_CRIMES_PEDESTRE"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Motocicleta|Moto")).count().alias("TOTAL_CRIMES_MOTOCICLISTA"),
                pl.col("DESCR_SUBTIPOLOCAL").filter(pl.col("DESCR_SUBTIPOLOCAL").str.contains("Residência|Condomínio")).count().alias("INDICE_RESIDENCIAL"),
                pl.col("DESCR_SUBTIPOLOCAL").filter(~pl.col("DESCR_SUBTIPOLOCAL").str.contains("Residência")).count().alias("TOTAL_NAO_RES_H3"),
                pl.col("LOGRADOURO").n_unique().alias("DENSIDADE_ENDERECOS")
            ]).with_columns(pl.lit(ano).alias("ANO_REFERENCIA"))

            df_final = self._injetar_deltas(df_atual, path_prata_anterior)

            buffer = io.BytesIO()
            df_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata_atual, Body=buffer.getvalue())
            logger.info(f"✅ Sucesso: {path_prata_atual} processado ({len(df_final)} áreas).")

        except Exception as e:
            if 'NoSuchKey' in str(e):
                logger.info(f"PRATA: Arquivo bruto de {ano} não encontrado. Pulando...")
            else:
                logger.error(f"Erro inesperado ao processar {ano}: {e}")

    def _injetar_deltas(self, df_atual, path_anterior):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_anterior)
            df_passado = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            
            df_join = df_atual.join(
                df_passado.select(["H3_INDEX", "TOTAL_CRIMES_MOTORISTA", "TOTAL_CRIMES_PEDESTRE", "TOTAL_CRIMES_MOTOCICLISTA"]),
                on="H3_INDEX",
                how="left",
                suffix="_PASSADO"
            ).fill_null(0)

            return df_join.with_columns([
                (pl.col("TOTAL_CRIMES_MOTORISTA") - pl.col("TOTAL_CRIMES_MOTORISTA_PASSADO")).alias("DELTA_MOTORISTA"),
                (pl.col("TOTAL_CRIMES_PEDESTRE") - pl.col("TOTAL_CRIMES_PEDESTRE_PASSADO")).alias("DELTA_PEDESTRE"),
                (pl.col("TOTAL_CRIMES_MOTOCICLISTA") - pl.col("TOTAL_CRIMES_MOTOCICLISTA_PASSADO")).alias("DELTA_MOTOCICLISTA")
            ]).drop([c for c in df_join.columns if "_PASSADO" in c])
        except:
            return df_atual.with_columns([
                pl.lit(0).alias("DELTA_MOTORISTA"),
                pl.lit(0).alias("DELTA_PEDESTRE"),
                pl.lit(0).alias("DELTA_MOTOCICLISTA")
            ])

if __name__ == "__main__":
    ProcessamentoPrata().executar_todos_os_anos()
