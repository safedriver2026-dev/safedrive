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
        self.colunas_lgpd = [
            "NUM_BO", "LOGRADOURO", "NUMERO_LOGRADOURO", 
            "NOME_DELEGACIA", "NOME_DEPARTAMENTO", "NOME_SECCIONAL",
            "NOME_DELEGACIA_CIRCUNSCRICAO", "NOME_DEPARTAMENTO_CIRCUNSCRICAO",
            "NOME_SECCIONAL_CIRCUNSCRICAO", "NOME_MUNICIPIO_CIRCUNSCRICAO"
        ]

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

            logger.info(f"PRATA: Lendo e consolidando todas as abas de {ano}...")
            resp_ssp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            conteudo_excel = resp_ssp['Body'].read()

            lf_ssp = self._extrair_todas_as_abas(conteudo_excel)

            if lf_ssp is None or lf_ssp.is_empty():
                raise ValueError(f"Nenhum dado valido encontrado no arquivo de {ano}")

            resp_geo = self.s3.get_object(Bucket=self.bucket, Key=path_geo)
            lf_geo = pl.read_parquet(io.BytesIO(resp_geo['Body'].read()))

            lf_ssp = lf_ssp.filter(
                (pl.col("LATITUDE").is_not_null()) & 
                (pl.col("LONGITUDE").is_not_null()) &
                (pl.col("LATITUDE") != 0)
            )

            df_pandas = lf_ssp.to_pandas()
            df_pandas['H3_INDEX'] = df_pandas.apply(
                lambda row: h3.latlng_to_cell(float(row['LATITUDE']), float(row['LONGITUDE']), self.h3_resolution), 
                axis=1
            )
            lf_ssp = pl.from_pandas(df_pandas)

            lf_ssp = lf_ssp.drop([c for c in self.colunas_lgpd if c in lf_ssp.columns])

            lf_crimes = self._agregar_crimes(lf_ssp)

            lf_final = lf_crimes.join(lf_geo, on="H3_INDEX", how="inner")

            lf_final = self._processar_indicadores(lf_final, ano)

            buffer = io.BytesIO()
            lf_final.write_parquet(buffer)
            buffer.seek(0)

            self.s3.put_object(
                Bucket=self.bucket,
                Key=path_prata,
                Body=buffer.getvalue()
            )

            logger.info(f"PRATA: Consolidacao total concluida para {ano}.")
            return True

        except Exception as e:
            logger.error(f"Erro na Camada Prata ({ano}): {e}")
            raise e

    def _extrair_todas_as_abas(self, conteudo):
        lista_dfs = []
        excel = pl.read_excel(io.BytesIO(conteudo), sheet_id=0) # Apenas para validar se abre
        
        # Tenta ler ate 13 abas (capa + 12 meses)
        for i in range(1, 14):
            try:
                df = pl.read_excel(io.BytesIO(conteudo), sheet_id=i)
                
                # Normaliza nomes de colunas para busca
                colunas_atuais = {c: c.upper() for c in df.columns}
                df = df.rename(colunas_atuais)

                if "LATITUDE" in df.columns:
                    logger.info(f"PRATA: Extraindo dados da aba {i}.")
                    # Garante que as colunas criticas sejam strings para evitar conflitos no concat
                    df = df.with_columns(pl.all().cast(pl.Utf8))
                    lista_dfs.append(df)
            except:
                continue
        
        if lista_dfs:
            return pl.concat(lista_dfs, how="diagonal")
        return None

    def _agregar_crimes(self, lf):
        return lf.group_by("H3_INDEX").agg([
            pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("(?i)Veículo|Automóvel|Carro")).count().alias("TOTAL_CRIMES_MOTORISTA"),
            pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("(?i)Celular|Transeunte")).count().alias("TOTAL_CRIMES_PEDESTRE"),
            pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("(?i)Motocicleta|Moto")).count().alias("TOTAL_CRIMES_MOTOCICLISTA")
        ])

    def _processar_indicadores(self, lf, ano):
        return lf.with_columns([
            pl.lit(ano).alias("ANO_REFERENCIA"),
            pl.col("TOTAL_NAO_RESIDENCIAIS_H3").alias("TOTAL_NAO_RES_H3"),
            pl.col("PROPORCAO_RESIDENCIAL_H3").alias("INDICE_RESIDENCIAL"),
            pl.col("DENSIDADE_LOGRADOUROS").alias("DENSIDADE_ENDERECOS")
        ]).fill_null(0)

if __name__ == "__main__":
    prata = ProcessamentoPrata()
    ano_inicio = 2022
    for a in range(ano_inicio, datetime.now().year + 1):
        prata.executar_prata(a)
