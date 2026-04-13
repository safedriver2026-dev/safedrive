import polars as pl
import h3
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import io
import os
import json
import logging
from datetime import datetime

# Configuração de logging padrão corporativo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
        # Definições de ambiente para o Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        # Conexão blindada S3v4 com path addressing
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        
        # Caminho absoluto para o tracker de estado do CDC
        self.tracker_path = "safedriver/datalake/prata/tracker_estado_bronze.json"

    def _carregar_tracker(self):
        """Carrega o histórico de tamanhos dos ficheiros já processados."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.tracker_path)
            return json.loads(resp['Body'].read())
        except ClientError:
            return {} # Primeira execução ou tracker ausente

    def _salvar_tracker(self, estado):
        """Atualiza e persiste o estado do CDC no Data Lake."""
        self.s3.put_object(
            Bucket=self.bucket, 
            Key=self.tracker_path, 
            Body=json.dumps(estado)
        )

    def executar_todos_os_anos(self):
        """Orquestra a leitura direta e o processamento de Deltas."""
        logger.info("PRATA: Iniciando consolidação com Change Data Capture (CDC)...")
        ano_atual = datetime.now().year
        estado_atual = self._carregar_tracker()
        houve_alteracao = False

        for ano in range(2022, ano_atual + 1):
            if self.processar_ano_com_delta(ano, estado_atual):
                houve_alteracao = True

        if houve_alteracao:
            self._salvar_tracker(estado_atual)
            logger.info("PRATA: Tracker de estado atualizado com sucesso no Datalake.")
        else:
            logger.info("PRATA: Ficheiros em sincronia. Nenhum reprocessamento necessário. Bypass global.")

    def processar_ano_com_delta(self, ano, estado):
        # Mapeamento absoluto de chaves no R2
        path_bronze = f"safedriver/datalake/bronze/ssp_raw_{ano}.xlsx"
        path_prata_atual = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
        path_prata_anterior = f"safedriver/datalake/prata/ssp_consolidada_{ano - 1}.parquet"

        try:
            # 1. Leitura rápida de metadados para validação CDC
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_bronze)
            tamanho_atual = meta['ContentLength']
        except ClientError as e:
            if 'NoSuchKey' in str(e) or '404' in str(e):
                logger.info(f"PRATA: Dados de {ano} não encontrados na Bronze. Ignorando ciclo.")
            else:
                logger.error(f"PRATA: Falha de conectividade ao verificar metadados de {ano}: {e}")
            return False

        tamanho_registado = estado.get(str(ano), 0)
        
        # 2. Avaliação de Change Data Capture (Bypass)
        if tamanho_atual == tamanho_registado:
            logger.info(f"PRATA: [{ano}] Bypass ativado. O ficheiro não sofreu alterações ({tamanho_atual} bytes).")
            return False

        logger.info(f"PRATA: [{ano}] Nova atualização detetada ({tamanho_registado} -> {tamanho_atual} bytes). Iniciando pipeline de transformação...")

        try:
            # 3. Extração e Tratamento da Camada Bronze
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            df = pl.read_excel(io.BytesIO(resp['Body'].read()))
            df = df.rename({c: c.replace("Ç", "C").replace("Ã", "A") for c in df.columns})

            df = df.with_columns([
                pl.col("LATITUDE").cast(pl.Float64, strict=False),
                pl.col("LONGITUDE").cast(pl.Float64, strict=False)
            ]).filter(
                pl.col("LATITUDE").is_not_null() & (pl.col("LATITUDE") != 0)
            )

            if df.is_empty():
                logger.warning(f"PRATA: Ficheiro de {ano} vazio ou sem coordenadas geolocalizadas após limpeza.")
                return False

            # 4. Tradução para Malha H3 (Resolução 8)
            h3_list = [h3.geo_to_h3(lat, lon, 8) for lat, lon in zip(df["LATITUDE"], df["LONGITUDE"])]
            df = df.with_columns(pl.Series("H3_INDEX", h3_list))

            # 5. Agregação e Consolidação de Features Espaciais
            df_atual = df.group_by(["H3_INDEX"]).agg([
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Roubo|Furto|Veículo|Carga")).count().alias("TOTAL_CRIMES_MOTORISTA"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Celular|Transeunte|Pessoa")).count().alias("TOTAL_CRIMES_PEDESTRE"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("Motocicleta|Moto")).count().alias("TOTAL_CRIMES_MOTOCICLISTA"),
                pl.col("DESCR_SUBTIPOLOCAL").filter(pl.col("DESCR_SUBTIPOLOCAL").str.contains("Residência|Condomínio")).count().alias("INDICE_RESIDENCIAL"),
                pl.col("DESCR_SUBTIPOLOCAL").filter(~pl.col("DESCR_SUBTIPOLOCAL").str.contains("Residência")).count().alias("TOTAL_NAO_RES_H3"),
                pl.col("LOGRADOURO").n_unique().alias("DENSIDADE_ENDERECOS")
            ]).with_columns(pl.lit(ano).alias("ANO_REFERENCIA"))

            # 6. Cálculo Analítico de Tendências (Deltas)
            df_final = self._injetar_deltas(df_atual, path_prata_anterior)

            # 7. Persistência em Formato Colunar (Parquet)
            buffer = io.BytesIO()
            df_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata_atual, Body=buffer.getvalue())
            
            # Atualização do Tracking na memória
            estado[str(ano)] = tamanho_atual
            logger.info(f"✅ PRATA: Ciclo {ano} finalizado e exportado em formato Parquet.")
            return True

        except Exception as e:
            logger.error(f"PRATA: Falha crítica no processamento da malha de {ano}: {e}")
            return False

    def _injetar_deltas(self, df_atual, path_anterior):
        """Cruza os dados processados com o ano anterior para identificar crescimento ou redução criminal."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_anterior)
            df_passado = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            
            # Left Join baseado no índice H3 para manter o referencial geográfico
            df_join = df_atual.join(
                df_passado.select(["H3_INDEX", "TOTAL_CRIMES_MOTORISTA", "TOTAL_CRIMES_PEDESTRE", "TOTAL_CRIMES_MOTOCICLISTA"]),
                on="H3_INDEX",
                how="left",
                suffix="_PASSADO"
            ).fill_null(0)

            # Equação de Delta (Atual - Passado)
            return df_join.with_columns([
                (pl.col("TOTAL_CRIMES_MOTORISTA") - pl.col("TOTAL_CRIMES_MOTORISTA_PASSADO")).alias("DELTA_MOTORISTA"),
                (pl.col("TOTAL_CRIMES_PEDESTRE") - pl.col("TOTAL_CRIMES_PEDESTRE_PASSADO")).alias("DELTA_PEDESTRE"),
                (pl.col("TOTAL_CRIMES_MOTOCICLISTA") - pl.col("TOTAL_CRIMES_MOTOCICLISTA_PASSADO")).alias("DELTA_MOTOCICLISTA")
            ]).drop([c for c in df_join.columns if "_PASSADO" in c])
            
        except ClientError:
            # Cold Start: Se o ano base não existir, os Deltas começam em zero
            return df_atual.with_columns([
                pl.lit(0).alias("DELTA_MOTORISTA"),
                pl.lit(0).alias("DELTA_PEDESTRE"),
                pl.lit(0).alias("DELTA_MOTOCICLISTA")
            ])

if __name__ == "__main__":
    ProcessamentoPrata().executar_todos_os_anos()
