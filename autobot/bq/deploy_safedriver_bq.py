import os
import io
import json
import boto3
import polars as pl
import pandas as pd
import time
import requests
import gc
from google.cloud import bigquery
from google.oauth2 import service_account
from botocore.config import Config
import warnings

warnings.filterwarnings("ignore")

class DeploySafeDriverBigQuery:
    """
    ENGINE DE DEPLOY SAFEDRIVER - GOOGLE BIGQUERY
    ------------------------------------------------
    Arquitetura: OBT (One Big Table) para Storytelling Tático.
    Integração: R2 Storage -> Pandas -> BigQuery.
    """
    def __init__(self):
        self.projeto = "SafeDriver"
        self.project_id = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9")
        self.dataset_id = os.getenv("BQ_DATASET_ID")
        
        if not self.dataset_id:
            raise ValueError("Variável BQ_DATASET_ID não configurada.")
            
        bq_json_str = os.getenv("BQ_SERVICE_ACCOUNT_JSON")
        if not bq_json_str:
            raise ValueError("Credenciais de serviço BigQuery ausentes.")
            
        credentials = service_account.Credentials.from_service_account_info(json.loads(bq_json_str))
        self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
        
        # Configuração R2 (S3 API)
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        if endpoint.endswith(f"/{self.bucket}"):
            endpoint = endpoint[: -len(f"/{self.bucket}")]
            
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint, 
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4', retries={'max_attempts': 3})
        )
        self.webhook_url = os.getenv("DISCORD_SUCESSO")

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=10)
            except: pass

    def _ler_parquet_r2(self, key):
        print(f"[INFO] Baixando artefato do R2: {key}", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pl.read_parquet(io.BytesIO(obj['Body'].read())).to_pandas()

    def _upload_table(self, df_pandas, table_name):
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        # Truncate para garantir que o grão equalizado não duplique dados
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        
        print(f"[INFO] Upload para o BigQuery: {table_name}...", flush=True)
        job = self.bq_client.load_table_from_dataframe(df_pandas, table_id, job_config=job_config)
        job.result()
        print(f"[SUCCESS] {table_name} populada com {len(df_pandas)} linhas.", flush=True)

    def _construir_matriz_risco_intermediaria(self):
        """
        Calcula os Quadrantes de Risco baseados nos BOs de 2025.
        Essencial para o gráfico de dispersão (Bolhas).
        """
        print("[INFO] Gerando tb_matriz_risco (Quadrantes 2025)...", flush=True)
        sql_matriz = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.tb_matriz_risco` AS
        WITH Base AS (
          SELECT H3_INDEX, COUNT(1) as VOLUME_REAL, AVG(RISCO_IA) as RISCO_MEDIO_REAL
          FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos`
          WHERE IS_MALHA = FALSE AND ANO_JOIN = 2025
          GROUP BY H3_INDEX
        ),
        CrimeRank AS (
          SELECT H3_INDEX, RUBRICA, COUNT(1) as qtd, ROW_NUMBER() OVER(PARTITION BY H3_INDEX ORDER BY COUNT(1) DESC) as rnk
          FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos`
          WHERE IS_MALHA = FALSE AND ANO_JOIN = 2025
          GROUP BY H3_INDEX, RUBRICA
        )
        SELECT 
          b.H3_INDEX, b.VOLUME_REAL, b.RISCO_MEDIO_REAL, c.RUBRICA as TOP_CRIME,
          CASE 
            WHEN b.VOLUME_REAL >= 30 AND b.RISCO_MEDIO_REAL >= 7.0 THEN '🔴 1 - ZONA CRÍTICA'
            WHEN b.VOLUME_REAL < 30  AND b.RISCO_MEDIO_REAL >= 7.0 THEN '🟠 2 - RISCO VITAL'
            WHEN b.VOLUME_REAL >= 30 AND b.RISCO_MEDIO_REAL < 7.0  THEN '🟡 3 - ATENÇÃO ALTA'
            ELSE '🟢 4 - MONITORAMENTO'
          END AS QUADRANTE
        FROM Base b LEFT JOIN CrimeRank c ON b.H3_INDEX = c.H3_INDEX AND c.rnk = 1
        """
        self.bq_client.query(sql_matriz).result()

    def _construir_obt_looker(self):
        """
        Cria a One Big Table final com GEOGRAPHY e SHAP DNA integrados.
        """
        print("[INFO] Fundindo dados na OBT Master Final...", flush=True)
        sql_obt = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.tb_looker_master_final` AS
        WITH Base_Fix AS (
          SELECT *,
            CASE WHEN ABS(CAST(LATITUDE AS FLOAT64)) > 90 THEN CAST(LATITUDE AS FLOAT64) / 1000000 ELSE CAST(LATITUDE AS FLOAT64) END as lat_fix,
            CASE WHEN ABS(CAST(LONGITUDE AS FLOAT64)) > 180 THEN CAST(LONGITUDE AS FLOAT64) / 1000000 ELSE CAST(LONGITUDE AS FLOAT64) END as lon_fix
          FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos`
        )
        SELECT 
            -- 1. EIXO TEMPORAL
            DATE_TRUNC(CAST(e.DATAOCORRENCIA AS DATE), MONTH) AS DATA_REFERENCIA_MES,
            e.ANO_JOIN AS ANO,
            CASE WHEN e.IS_MALHA THEN 'PREVISÃO (MALHA)' ELSE 'HISTÓRICO (B.O.)' END AS TIPO_REGISTRO,

            -- 2. EIXO GEOGRÁFICO
            e.H3_INDEX, e.CIDADE, e.BAIRRO, e.LOGRADOURO,
            ST_GEOGPOINT(e.lon_fix, e.lat_fix) AS GEOMETRIA_PONTO,

            -- 3. MÉTRICAS IA (Honrando Tweedie)
            e.RISCO_IA,
            e.VOLUME_TWEEDIE,
            e.KPI_RISCO_EVOLUCAO,
            e.KPI_VOLUME_TOTAL,
            e.STATUS_OPERACIONAL,
            e.CLUSTER_RANK,
            
            -- 4. CONTEXTO OPERACIONAL
            COALESCE(m.QUADRANTE, '⚪ ÁREA SEM REGISTRO 2025') AS QUADRANTE_RISCO,
            m.TOP_CRIME AS CRIME_PREDOMINANTE_H3,
            e.SAZON_PERIODO AS PERIODO_DIA,
            e.FEAT_CONTEXTO_CRITICO AS CENARIO_ALVO,
            
            -- 5. DNA DO CRIME (SHAP)
            s.* EXCEPT(CIDADE, BAIRRO)

        FROM Base_Fix e
        LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_matriz_risco` m ON e.H3_INDEX = m.H3_INDEX
        LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_dim_shap` s ON CAST(e.CIDADE AS STRING) = CAST(s.CIDADE AS STRING) 
                                                                    AND CAST(e.BAIRRO AS STRING) = CAST(s.BAIRRO AS STRING)
        """
        self.bq_client.query(sql_obt).result()

    def executar_deploy(self):
        inicio_deploy = time.time()
        print(f"🌐 [START] Deploy SafeDriver Project: {self.project_id}", flush=True)

        # 1. Sincronização de Tabelas (R2 -> BQ)
        df_eventos = self._ler_parquet_r2("datalake/ouro/looker_dossie_eventos.parquet")
        self._upload_table(df_eventos, "tb_dossie_eventos")
        
        df_shap = self._ler_parquet_r2("datalake/ouro/looker_dim_dna_shap.parquet")
        self._upload_table(df_shap, "tb_dim_shap")

        # 2. Refinamento SQL Dimensional
        self._construir_matriz_risco_intermediaria()
        self._construir_obt_looker()

        duracao = round(time.time() - inicio_deploy, 2)
        print(f"✅ [FINISHED] Deploy concluído em {duracao}s")
        self._notificar_discord(f"🌐 **DEPLOY SAFEDRIVER CONCLUÍDO**\n\nBigQuery OBT: Atualizada\nGrão: 2025-2026 Equalizado\nStatus: Pronto para Visualização.")

if __name__ == "__main__":
    DeploySafeDriverBigQuery().executar_deploy()
