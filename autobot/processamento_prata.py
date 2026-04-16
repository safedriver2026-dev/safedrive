import polars as pl
import pandas as pd
import h3
import boto3
from botocore.config import Config
import io
import os
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

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
            self.df_malha = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            self.df_malha_lazy = self.df_malha.lazy().with_columns([
                self._limpar_texto_extremo("NM_MUN").alias("NM_MUN"),
                self._limpar_texto_extremo("NM_BAIRRO").alias("NM_BAIRRO"),
                self._limpar_texto_extremo("LOGRADOURO").alias("LOGRADOURO_GRID")
            ])
            
            self.lookup_endereco = (
                self.df_malha_lazy
                .filter(pl.col("LOGRADOURO_GRID") != "INDEFINIDO")
                .group_by(["LOGRADOURO_GRID", "NM_BAIRRO", "NM_MUN"])
                .agg(pl.col("H3_INDEX").first())
            ).collect()
            
            logger.info("PRATA: Dependências e Tipagem de Malha inicializadas.")
        except Exception as e:
            logger.error(f"PRATA: Falha crítica na inicialização: {e}")
            self.df_malha_lazy = None

    def processar_ano_com_delta(self, ano, estado, force=False):
        path_trusted_bronze = f"{self.base_path}/bronze/trusted/ssp_trusted_{ano}.parquet"
        path_prata = f"{self.base_path}/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            # 1. Carregamento dos dados "Texto" da Bronze
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_trusted_bronze)
            lf = pl.read_parquet(io.BytesIO(resp['Body'].read())).lazy()

            # --- 2. CONVERSÃO DE TIPOS (O CORAÇÃO DA PRATA) ---
            # Aqui transformamos o texto em dados matemáticos e temporais
            
            mapeamento = {
                "CIDADE": "NM_MUN_ORIGINAL", "MUNICIPIO": "NM_MUN_ORIGINAL",
                "BAIRRO": "NM_BAIRRO_ORIGINAL", "LOGRADOURO": "LOGRADOURO_ORIGINAL",
                "DATA_OCORRENCIA_BO": "DATA_BRUTA", "DATA_OCORRENCIA": "DATA_BRUTA"
            }
            lf = lf.rename({old: new for old, new in mapeamento.items() if old in lf.collect_schema().names()})

            # Casting de Datas, Horas e Coordenadas
            lf = lf.with_columns([
                # Datas: de Texto para Date
                pl.col("DATA_BRUTA").str.to_date(format="%d/%m/%Y", strict=False).alias("DATA_PARSED"),
                
                # Coordenadas: de Texto para Float (Tratando a vírgula do Excel)
                pl.col("LATITUDE").cast(pl.String).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("LONGITUDE").cast(pl.String).str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0),
                
                # Horas: Pegando apenas o Inteiro da hora para Sazonalidade
                pl.col("HORA_OCORRENCIA_BO").cast(pl.String).str.split(":").list.first().cast(pl.Int32, strict=False).alias("HORA_INT")
            ])

            # Criação de colunas de Sazonalidade (Úteis para IA)
            lf = lf.with_columns([
                pl.col("DATA_PARSED").dt.month().fill_null(1).alias("MES"),
                pl.col("DATA_PARSED").dt.weekday().fill_null(1).alias("DIA_SEMANA"),
                pl.col("DATA_PARSED").dt.year().fill_null(ano).alias("ANO_REF")
            ])

            # --- 3. LIMPEZA DE TEXTO E RECUPERAÇÃO H3 ---
            campos_texto = ["NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL", "LOGRADOURO_ORIGINAL", "RUBRICA"]
            lf = lf.with_columns([self._limpar_texto_extremo(c) for c in campos_texto if c in lf.collect_schema().names()])

            # Resgate por endereço (Mão Dupla)
            lf = lf.join(
                self.lookup_endereco.lazy(),
                left_on=["LOGRADOURO_ORIGINAL", "NM_BAIRRO_ORIGINAL", "NM_MUN_ORIGINAL"],
                right_on=["LOGRADOURO_GRID", "NM_BAIRRO", "NM_MUN"],
                how="left"
            ).with_columns(
                pl.coalesce([pl.col("H3_INDEX"), pl.col("H3_INDEX_right")]).alias("H3_INDEX")
            ).drop("H3_INDEX_right")

            # Filtramos quem não tem H3 após o resgate
            lf = lf.filter(pl.col("H3_INDEX").is_not_null())

            # --- 4. AGREGAÇÃO FINAL (TRIPADA) ---
            # Agora que temos tipos (Int, Date), podemos somar e agrupar
            lf_final = lf.group_by(["H3_INDEX", "ANO_REF", "MES", "DIA_SEMANA", "HORA_INT", "RUBRICA"]).agg([
                pl.len().alias("QTD_CRIMES")
            ])

            # Persistência
            buffer = io.BytesIO()
            lf_final.collect().write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            logger.info(f"PRATA: [{ano}] Tipagem concluída e dados consolidados.")
            return True

        except Exception as e:
            logger.error(f"PRATA: Falha no casting/processamento de {ano}: {e}")
            return False
