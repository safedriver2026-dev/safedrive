import polars as pl
import pandas as pd
import h3
import boto3
from botocore.config import Config
import io
import os
import json
import logging
import gc
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}, max_pool_connections=50))
        
        self.base_path = self._localizar_datalake_real()
        self.tracker_path = f"{self.base_path}/prata/tracker_estado_bronze.json"
        self.malha_path = f"{self.base_path}/base_geografica/safedriver_geo_base_sp_h3_9.parquet"
        
        self.campos_recuperados_grade = 0
        self._inicializar_dependencias()

    def _localizar_datalake_real(self):
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, MaxKeys=100):
                for obj in page.get('Contents', []):
                    if "datalake/bronze/trusted/" in obj['Key']:
                        return obj['Key'].split("datalake/")[0] + "datalake"
            return "datalake"
        except: return "datalake"

    def _limpar_texto_extremo(self, coluna):
        return (
            pl.col(coluna)
            .cast(pl.String)
            .str.to_uppercase()
            .str.replace_all(r"[ÁÀÂÃÄ]", "A")
            .str.replace_all(r"[ÉÈÊË]", "E")
            .str.replace_all(r"[ÍÌÎÏ]", "I")
            .str.replace_all(r"[ÓÒÔÕÖ]", "O")
            .str.replace_all(r"[ÚÙÛÜ]", "U")
            .str.replace_all(r"[Ç]", "C")
            .str.replace_all(r"[Ñ]", "N")
            .str.strip_chars()
            .fill_null("INDEFINIDO")
        )

    def _inicializar_dependencias(self):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            # Carregamos a malha com tipos leves
            self.df_malha = pl.read_parquet(io.BytesIO(resp['Body'].read())).with_columns([
                pl.col("DENSIDADE_AJUSTADA").cast(pl.Float32),
                pl.col("TAXA_VACANCIA").cast(pl.Float32)
            ])
            self.df_malha_lazy = self.df_malha.lazy().with_columns([
                self._limpar_texto_extremo("NM_MUN").alias("NM_MUN"),
                self._limpar_texto_extremo("NM_BAIRRO").alias("NM_BAIRRO")
            ])
            logger.info("PRATA: Malha mestre carregada e otimizada.")
        except Exception as e:
            logger.error(f"PRATA: Falha crítica: {e}")
            self.df_malha_lazy = None

    def processar_ano_com_delta(self, ano, estado, force=False):
        path_trusted = f"{self.base_path}/bronze/trusted/ssp_trusted_{ano}.parquet"
        path_prata = f"{self.base_path}/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            tamanho_atual = meta['ContentLength']
            if not force and estado.get(str(ano)) == tamanho_atual: return None

            resp = self.s3.get_object(Bucket=self.bucket, Key=path_trusted)
            lf = pl.read_parquet(io.BytesIO(resp['Body'].read())).lazy()

            # 1. Filtro e Renomeação Inteligente (Alias Machine)
            cols = lf.collect_schema().names()
            rename_map = {}
            if "NOME_MUNICIPIO" in cols: rename_map["NOME_MUNICIPIO"] = "NM_MUN_ORIGINAL"
            elif "CIDADE" in cols: rename_map["CIDADE"] = "NM_MUN_ORIGINAL"
            
            if "DATA_OCORRENCIA_BO" in cols: rename_map["DATA_OCORRENCIA_BO"] = "DATA_BRUTA"
            if "HORA_OCORRENCIA_BO" in cols: rename_map["HORA_OCORRENCIA_BO"] = "HORA"
            if "BAIRRO" in cols: rename_map["BAIRRO"] = "NM_BAIRRO_ORIGINAL"
            
            lf = lf.rename(rename_map).filter(pl.col("H3_INDEX").is_not_null())

            # 2. Engenharia de Recursos de Alta Performance (Features em Silver)
            lf = lf.with_columns([
                pl.col("DATA_BRUTA").str.to_date(format="%d/%m/%Y", strict=False).alias("DATA"),
                pl.col("HORA").str.split(":").list.first().cast(pl.Int8).alias("HORA_INT")
            ]).with_columns([
                pl.col("DATA").dt.month().cast(pl.Int8).alias("MES_OCORRENCIA"),
                pl.col("DATA").dt.weekday().cast(pl.Int8).alias("DIA_SEMANA"),
                pl.col("DATA").dt.day().cast(pl.Int8).alias("DIA_MES"),
                # Inteligência de Negócio: Dia de Pagamento (5 ao 10)
                pl.when(pl.col("DATA").dt.day().is_between(5, 10)).then(1).otherwise(0).cast(pl.Int8).alias("IS_PAGAMENTO"),
                # Inteligência: Fim de Semana
                pl.when(pl.col("DATA").dt.weekday() >= 6).then(1).otherwise(0).cast(pl.Int8).alias("IS_FDS")
            ])

            # 3. Agregação e Compactação
            lf_agg = lf.group_by([
                "H3_INDEX", "NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL", 
                "MES_OCORRENCIA", "DIA_SEMANA", "IS_PAGAMENTO", "IS_FDS"
            ]).agg([
                pl.len().cast(pl.Int32).alias("TOTAL_CRIMES")
            ])

            # 4. Join Final e Downcasting de Tipos
            df_final = lf_agg.join(self.df_malha_lazy, on="H3_INDEX", how="left").with_columns([
                pl.coalesce([pl.col("NM_MUN"), pl.col("NM_MUN_ORIGINAL")]).cast(pl.Categorical).alias("NM_MUN"),
                pl.coalesce([pl.col("NM_BAIRRO"), pl.col("NM_BAIRRO_ORIGINAL")]).cast(pl.Categorical).alias("NM_BAIRRO"),
                pl.col("DENSIDADE_AJUSTADA").cast(pl.Float32).alias("DENSIDADE"),
                pl.lit(ano).cast(pl.Int16).alias("ANO_REF")
            ]).drop(["NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL"])

            # 5. Persistência
            buffer = io.BytesIO()
            df_final.collect().write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            estado[str(ano)] = tamanho_atual 
            return True
        except Exception as e:
            logger.error(f"PRATA: Erro {ano}: {e}")
            return False

    def executar_todos_os_anos(self, force=False):
        estado = self._carregar_tracker()
        for ano in range(2022, datetime.now().year + 1):
            if self.processar_ano_com_delta(ano, estado, force):
                self._salvar_tracker(estado)
        return {"status": "✅ Prata Otimizada"}

    def _carregar_tracker(self):
        try: return json.loads(self.s3.get_object(Bucket=self.bucket, Key=self.tracker_path)['Body'].read())
        except: return {}

    def _salvar_tracker(self, estado):
        self.s3.put_object(Bucket=self.bucket, Key=self.tracker_path, Body=json.dumps(estado))
