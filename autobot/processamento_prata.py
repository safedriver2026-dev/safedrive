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
        # Conexão R2
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
        """Varre a Bronze, identifica os anos e processa com lógica de Delta."""
        logger.info("PRATA: Iniciando processamento de tendências históricas...")
        
        try:
            prefixo_bronze = "safedriver/datalake/bronze/"
            paginator = self.s3.get_paginator('list_objects_v2')
            anos_encontrados = []
            
            # 1. Descoberta Automática de Anos
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefixo_bronze):
                for obj in page.get('Contents', []):
                    match = re.search(r'ssp_raw_(\d{4})\.xlsx', obj['Key'])
                    if match:
                        anos_encontrados.append(int(match.group(1)))
            
            if not anos_encontrados:
                logger.warning("PRATA: Nenhum dado bruto (xlsx) encontrado na Bronze.")
                return

            # Ordenação é CRÍTICA: o ano 2023 precisa do 2022 pronto para calcular o Delta
            anos_ordenados = sorted(list(set(anos_encontrados)))
            logger.info(f"PRATA: Ciclo de processamento ordenado: {anos_ordenados}")

            for ano in anos_ordenados:
                self.processar_ano_com_delta(ano)

        except Exception as e:
            logger.error(f"FALHA NO MOTOR DA PRATA: {e}")

    def processar_ano_com_delta(self, ano):
        path_bronze = f"safedriver/datalake/bronze/ssp_raw_{ano}.xlsx"
        path_prata_atual = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
        path_prata_anterior = f"safedriver/datalake/prata/ssp_consolidada_{ano - 1}.parquet"

        try:
            logger.info(f"--- Refinando Ciclo: {ano} ---")
            
            # 1. Leitura e Normalização (Trata as variações de nomes de colunas da SSP)
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            df = pl.read_excel(io.BytesIO(resp['Body'].read()))
            df = df.rename({c: c.replace("Ç", "C").replace("Ã", "A") for c in df.columns})

            # 2. Limpeza Geográfica para H3 (Resolução 8)
            df = df.with_columns([
                pl.col("LATITUDE").cast(pl.Float64, strict=False),
                pl.col("LONGITUDE").cast(pl.Float64, strict=False)
            ]).filter(
                pl.col("LATITUDE").is_not_null() & (pl.col("LATITUDE") != 0)
            )

            h3_list = [h3.geo_to_h3(lat, lon, 8) for lat, lon in zip(df["LATITUDE"], df["LONGITUDE"])]
            df = df.with_columns(pl.Series("H3_INDEX", h3_list))

            # 3. Agrupamento por Hexágono (Features de Contexto)
            df_atual = df.group_by(["H3_INDEX"]).agg([
                # Alvos de Treino (Target)
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Roubo|Furto|Veículo|Carga")).count().alias("TOTAL_CRIMES_MOTORISTA"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Celular|Transeunte|Pessoa")).count().alias("TOTAL_CRIMES_PEDESTRE"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Motocicleta|Moto")).count().alias("TOTAL_CRIMES_MOTOCICLISTA"),
                
                # Features Estruturais
                pl.col("DESCR_SUBTIPOLOCAL").filter(pl.col("DESCR_SUBTIPOLOCAL").str.contains("Residência|Condomínio")).count().alias("INDICE_RESIDENCIAL"),
                pl.col("DESCR_SUBTIPOLOCAL").filter(~pl.col("DESCR_SUBTIPOLOCAL").str.contains("Residência")).count().alias("TOTAL_NAO_RES_H3"),
                pl.col("LOGRADOURO").n_unique().alias("DENSIDADE_ENDERECOS")
            ]).with_columns(pl.lit(ano).alias("ANO_REFERENCIA"))

            # 4. Cálculo da Variação (Delta)
            # A fórmula é: $$\Delta = \text{Crime}_{Ano} - \text{Crime}_{Ano-1}$$
            df_final = self._injetar_deltas(df_atual, path_prata_anterior)

            # 5. Exportação para Prata (Parquet)
            buffer = io.BytesIO()
            df_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata_atual, Body=buffer.getvalue())
            
            logger.info(f"✅ Sucesso: {path_prata_atual} processado com {len(df_final)} áreas.")

        except Exception as e:
            logger.error(f"Erro ao processar o ano {ano}: {e}")

    def _injetar_deltas(self, df_atual, path_anterior):
        """Compara o hexágono atual com o seu estado no ano anterior."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_anterior)
            df_passado = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            
            # Une as bases pelo índice H3
            df_join = df_atual.join(
                df_passado.select(["H3_INDEX", "TOTAL_CRIMES_MOTORISTA", "TOTAL_CRIMES_PEDESTRE", "TOTAL_CRIMES_MOTOCICLISTA"]),
                on="H3_INDEX",
                how="left",
                suffix="_PASSADO"
            ).fill_null(0)

            # Calcula os Deltas (Diferença real)
            return df_join.with_columns([
                (pl.col("TOTAL_CRIMES_MOTORISTA") - pl.col("TOTAL_CRIMES_MOTORISTA_PASSADO")).alias("DELTA_MOTORISTA"),
                (pl.col("TOTAL_CRIMES_PEDESTRE") - pl.col("TOTAL_CRIMES_PEDESTRE_PASSADO")).alias("DELTA_PEDESTRE"),
                (pl.col("TOTAL_CRIMES_MOTOCICLISTA") - pl.col("TOTAL_CRIMES_MOTOCICLISTA_PASSADO")).alias("DELTA_MOTOCICLISTA")
            ]).drop([c for c in df_join.columns if "_PASSADO" in c])

        except:
            # Caso não exista ano anterior (ex: 2022), o delta é zero
            return df_atual.with_columns([
                pl.lit(0).alias("DELTA_MOTORISTA"),
                pl.lit(0).alias("DELTA_PEDESTRE"),
                pl.lit(0).alias("DELTA_MOTOCICLISTA")
            ])

if __name__ == "__main__":
    ProcessamentoPrata().executar_todos_os_anos()
