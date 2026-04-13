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
        """Varre a pasta Bronze e processa cada arquivo de ano encontrado."""
        logger.info("PRATA: Iniciando varredura completa da Camada Bronze...")
        
        try:
            # Lista todos os arquivos na pasta bronze
            prefixo_bronze = "safedriver/datalake/bronze/"
            paginator = self.s3.get_paginator('list_objects_v2')
            
            anos_encontrados = []
            
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefixo_bronze):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    # Busca o padrão 'ssp_raw_YYYY.xlsx' no nome do arquivo
                    match = re.search(r'ssp_raw_(\d{4})\.xlsx', key)
                    if match:
                        ano = match.group(1)
                        anos_encontrados.append(ano)
            
            if not anos_encontrados:
                logger.warning("PRATA: Nenhum arquivo ssp_raw_YYYY.xlsx encontrado na Bronze.")
                return

            logger.info(f"PRATA: Anos detectados para processamento: {anos_encontrados}")
            
            for ano in sorted(anos_encontrados):
                self.processar_ano_especifico(ano)

        except Exception as e:
            logger.error(f"FALHA NA VARREDURA DA PRATA: {e}")

    def processar_ano_especifico(self, ano):
        path_bronze = f"safedriver/datalake/bronze/ssp_raw_{ano}.xlsx"
        path_prata = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"

        try:
            logger.info(f"--- Processando Ano: {ano} ---")
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            
            # 1. Leitura bruta
            df = pl.read_excel(io.BytesIO(resp['Body'].read()))
            
            # 2. Normalização de Cabeçalho (ASCII)
            df = df.rename({c: c.replace("Ç", "C").replace("Ã", "A") for c in df.columns})

            # 3. Limpeza Geográfica Estrita
            df = df.with_columns([
                pl.col("LATITUDE").cast(pl.Float64, strict=False),
                pl.col("LONGITUDE").cast(pl.Float64, strict=False)
            ]).filter(
                pl.col("LATITUDE").is_not_null() & 
                pl.col("LONGITUDE").is_not_null() &
                (pl.col("LATITUDE") != 0) & 
                (pl.col("LONGITUDE") != 0)
            )

            if df.is_empty():
                logger.warning(f"Aviso: {ano} não rendeu dados após limpeza geográfica.")
                return

            # 4. Geração do H3 (Resolução 8)
            h3_indices = [
                h3.geo_to_h3(lat, lon, 8) 
                for lat, lon in zip(df["LATITUDE"], df["LONGITUDE"])
            ]
            df = df.with_columns(pl.Series("H3_INDEX", h3_indices))

            # 5. Consolidação para a Ouro (Features e Targets)
            df_final = df.group_by(["H3_INDEX", "ANO_BO"]).agg([
                # Alvos (Targets) para o Treino
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Roubo|Furto|Veículo|Carga")).count().alias("TOTAL_CRIMES_MOTORISTA"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Celular|Transeunte|Pessoa")).count().alias("TOTAL_CRIMES_PEDESTRE"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Motocicleta|Moto")).count().alias("TOTAL_CRIMES_MOTOCICLISTA"),
                
                # Características (Features) para o Modelo
                pl.col("DESCR_SUBTIPOLOCAL").filter(pl.col("DESCR_SUBTIPOLOCAL").str.contains("Residência|Condomínio")).count().alias("INDICE_RESIDENCIAL"),
                pl.col("DESCR_SUBTIPOLOCAL").filter(~pl.col("DESCR_SUBTIPOLOCAL").str.contains("Residência")).count().alias("TOTAL_NAO_RES_H3"),
                pl.col("LOGRADOURO").n_unique().alias("DENSIDADE_ENDERECOS")
            ]).with_columns(
                pl.col("ANO_BO").alias("ANO_REFERENCIA")
            )

            # 6. Upload Parquet
            buffer = io.BytesIO()
            df_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())
            
            logger.info(f"✅ Sucesso: {path_prata} sincronizado ({len(df_final)} hexágonos).")

        except Exception as e:
            logger.error(f"Erro no processamento do ano {ano}: {e}")

if __name__ == "__main__":
    ProcessamentoPrata().executar_todos_os_anos()
