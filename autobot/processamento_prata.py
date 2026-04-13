import polars as pl
import h3
import boto3
import io
import os
import re
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
            aws_secret_access_key=self.secret_key
        )

    def executar_todos_os_anos(self):
        """Varre a Bronze e processa os anos por ordem para calcular os deltas."""
        logger.info("PRATA: A iniciar processamento em lote com lógica de Deltas...")
        
        try:
            prefixo_bronze = "safedriver/datalake/bronze/"
            paginator = self.s3.get_paginator('list_objects_v2')
            anos_encontrados = []
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefixo_bronze):
                for obj in page.get('Contents', []):
                    match = re.search(r'ssp_raw_(\d{4})\.xlsx', obj['Key'])
                    if match:
                        anos_encontrados.append(int(match.group(1)))
            
            if not anos_encontrados:
                logger.warning("PRATA: Nenhum ficheiro encontrado na camada Bronze.")
                return

            # Ordenação é obrigatória para que o Delta do ano N dependa do N-1 já processado
            anos_ordenados = sorted(list(set(anos_encontrados)))
            logger.info(f"PRATA: Ciclo de processamento: {anos_ordenados}")

            for ano in anos_ordenados:
                self.processar_ano_com_delta(ano)

        except Exception as e:
            logger.error(f"FALHA NO CICLO DA PRATA: {e}")

    def processar_ano_com_delta(self, ano):
        path_bronze = f"safedriver/datalake/bronze/ssp_raw_{ano}.xlsx"
        path_prata_atual = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
        path_prata_anterior = f"safedriver/datalake/prata/ssp_consolidada_{ano - 1}.parquet"

        try:
            logger.info(f"--- Processando Ano: {ano} ---")
            
            # 1. Leitura e Limpeza Geográfica
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            df = pl.read_excel(io.BytesIO(resp['Body'].read()))
            df = df.rename({c: c.replace("Ç", "C").replace("Ã", "A") for c in df.columns})

            df = df.with_columns([
                pl.col("LATITUDE").cast(pl.Float64, strict=False),
                pl.col("LONGITUDE").cast(pl.Float64, strict=False)
            ]).filter(
                pl.col("LATITUDE").is_not_null() & (pl.col("LATITUDE") != 0)
            )

            # 2. Indexação H3 (Resolução 8)
            h3_list = [h3.geo_to_h3(lat, lon, 8) for lat, lon in zip(df["LATITUDE"], df["LONGITUDE"])]
            df = df.with_columns(pl.Series("H3_INDEX", h3_list))

            # 3. Consolidação de Features do Ano Atual
            df_atual = df.group_by(["H3_INDEX"]).agg([
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Roubo|Furto|Veículo|Carga")).count().alias("TOTAL_CRIMES_MOTORISTA"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Celular|Transeunte|Pessoa")).count().alias("TOTAL_CRIMES_PEDESTRE"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Motocicleta|Moto")).count().alias("TOTAL_CRIMES_MOTOCICLISTA"),
                pl.col("DESCR_SUBTIPOLOCAL").filter(pl.col("DESCR_SUBTIPOLOCAL").str.contains("Residência|Condomínio")).count().alias("INDICE_RESIDENCIAL"),
                pl.col("DESCR_SUBTIPOLOCAL").filter(~pl.col("DESCR_SUBTIPOLOCAL").str.contains("Residência")).count().alias("TOTAL_NAO_RES_H3"),
                pl.col("LOGRADOURO").n_unique().alias("DENSIDADE_ENDERECOS")
            ]).with_columns(pl.lit(ano).alias("ANO_REFERENCIA"))

            # 4. Cálculo de Deltas (Comparação com Ano Anterior)
            df_final = self._aplicar_logica_delta(df_atual, path_prata_anterior)

            # 5. Exportação
            buffer = io.BytesIO()
            df_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata_atual, Body=buffer.getvalue())
            
            logger.info(f"✅ PRATA: {path_prata_atual} guardado com sucesso.")

        except Exception as e:
            logger.error(f"Erro no processamento de {ano}: {e}")

    def _aplicar_logica_delta(self, df_atual, path_anterior):
        """Tenta carregar o ano anterior para calcular a variação de crimes."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_anterior)
            df_passado = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            
            logger.info("IA: A calcular deltas em relação ao ano anterior...")
            
            # Une as bases pelo hexágono para ver quem subiu ou desceu
            df_delta = df_atual.join(
                df_passado.select([
                    "H3_INDEX", 
                    "TOTAL_CRIMES_MOTORISTA", 
                    "TOTAL_CRIMES_PEDESTRE",
                    "TOTAL_CRIMES_MOTOCICLISTA"
                ]),
                on="H3_INDEX",
                how="left",
                suffix="_ANTERIOR"
            ).fill_null(0)

            # Cálculo das variações (Deltas)
            return df_delta.with_columns([
                (pl.col("TOTAL_CRIMES_MOTORISTA") - pl.col("TOTAL_CRIMES_MOTORISTA_ANTERIOR")).alias("DELTA_MOTORISTA"),
                (pl.col("TOTAL_CRIMES_PEDESTRE") - pl.col("TOTAL_CRIMES_PEDESTRE_ANTERIOR")).alias("DELTA_PEDESTRE"),
                (pl.col("TOTAL_CRIMES_MOTOCICLISTA") - pl.col("TOTAL_CRIMES_MOTOCICLISTA_ANTERIOR")).alias("DELTA_MOTOCICLISTA")
            ]).drop([c for c in df_delta.columns if "_ANTERIOR" in c])

        except:
            # Cold Start: Se não houver ano anterior, os deltas são zero
            logger.info("IA: Ano anterior não encontrado. Deltas definidos como zero.")
            return df_atual.with_columns([
                pl.lit(0).alias("DELTA_MOTORISTA"),
                pl.lit(0).alias("DELTA_PEDESTRE"),
                pl.lit(0).alias("DELTA_MOTOCICLISTA")
            ])

if __name__ == "__main__":
    ProcessamentoPrata().executar_todos_os_anos()
