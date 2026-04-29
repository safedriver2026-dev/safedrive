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
    Motor de Implantação SafeDriver no Google BigQuery.
    Realiza a consolidação da One Big Table (OBT) unificando dados geoespaciais,
    projeções preditivas e dimensões explicativas (SHAP).
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

    def _notificar_webhook(self, msg):
        """Notifica o encerramento das operações no canal de monitoramento."""
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=10)
            except Exception: pass

    def _ler_parquet_r2(self, key):
        """Extrai artefatos gerados pelo motor preditivo."""
        print(f"[SISTEMA] Extraindo artefato do repositório: {key}", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pl.read_parquet(io.BytesIO(obj['Body'].read())).to_pandas()

    def _upload_table(self, df_pandas, table_name):
        """Carrega os dataframes temporários no data warehouse."""
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        
        print(f"[SISTEMA] Processando carga para tabela: {table_name}", flush=True)
        job = self.bq_client.load_table_from_dataframe(df_pandas, table_id, job_config=job_config)
        job.result()
        print(f"[SISTEMA] Tabela {table_name} atualizada com {len(df_pandas)} registros.", flush=True)

    def _construir_matriz_risco_intermediaria(self):
        """
        Gera os quadrantes de risco operacional com base nos registros históricos de 2025.
        Utilizado para validação estatística no gráfico de dispersão.
        """
        print("[PROCESSAMENTO] Compilando matriz de risco operacional...", flush=True)
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
            WHEN b.VOLUME_REAL >= 30 AND b.RISCO_MEDIO_REAL >= 7.0 THEN '1 - ZONA CRITICA'
            WHEN b.VOLUME_REAL < 30  AND b.RISCO_MEDIO_REAL >= 7.0 THEN '2 - RISCO VITAL'
            WHEN b.VOLUME_REAL >= 30 AND b.RISCO_MEDIO_REAL < 7.0  THEN '3 - ATENCAO ALTA'
            ELSE '4 - MONITORAMENTO'
          END AS QUADRANTE
        FROM Base b LEFT JOIN CrimeRank c ON b.H3_INDEX = c.H3_INDEX AND c.rnk = 1
        """
        self.bq_client.query(sql_matriz).result()

    def _construir_obt_looker(self):
        """
        Sintetiza a tabela final (OBT), aplicando transformações geoespaciais
        e integrando os tensores SHAP no nível municipal.
        """
        print("[PROCESSAMENTO] Consolidando arquitetura OBT (One Big Table)...", flush=True)
        sql_obt = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.tb_looker_master_final` AS
        WITH Base_Fix AS (
          SELECT *,
            CASE WHEN ABS(CAST(LATITUDE AS FLOAT64)) > 90 THEN CAST(LATITUDE AS FLOAT64) / 1000000 ELSE CAST(LATITUDE AS FLOAT64) END as lat_fix,
            CASE WHEN ABS(CAST(LONGITUDE AS FLOAT64)) > 180 THEN CAST(LONGITUDE AS FLOAT64) / 1000000 ELSE CAST(LONGITUDE AS FLOAT64) END as lon_fix
          FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos`
        )
        SELECT 
            -- Temporalidade
            DATE_TRUNC(CAST(e.DATAOCORRENCIA AS DATE), MONTH) AS DATA_REFERENCIA_MES,
            e.ANO_JOIN AS ANO,
            CASE WHEN e.IS_MALHA THEN 'PREVISAO (MALHA)' ELSE 'HISTORICO (BO)' END AS TIPO_REGISTRO,

            -- Geografia espacial
            e.H3_INDEX, e.CIDADE, e.BAIRRO, e.LOGRADOURO,
            ST_GEOGPOINT(e.lon_fix, e.lat_fix) AS GEOMETRIA_PONTO,

            -- Indicadores de modelagem (Tweedie)
            e.RISCO_IA,
            e.VOLUME_TWEEDIE,
            e.KPI_RISCO_EVOLUCAO,
            e.KPI_VOLUME_TOTAL,
            e.STATUS_OPERACIONAL,
            e.CLUSTER_RANK,
            
            -- Contexto tático
            COALESCE(m.QUADRANTE, 'AREA SEM REGISTRO 2025') AS QUADRANTE_RISCO,
            m.TOP_CRIME AS CRIME_PREDOMINANTE_H3,
            e.SAZON_PERIODO AS PERIODO_DIA,
            e.FEAT_CONTEXTO_CRITICO AS CENARIO_ALVO,
            
            -- Explicabilidade (SHAP agregado por Cidade)
            s.* EXCEPT(CIDADE)

        FROM Base_Fix e
        LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_matriz_risco` m ON e.H3_INDEX = m.H3_INDEX
        LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_dim_dna_cidade` s ON CAST(e.CIDADE AS STRING) = CAST(s.CIDADE AS STRING) 
        """
        self.bq_client.query(sql_obt).result()

    def executar_deploy(self):
        inicio_deploy = time.time()
        print(f"[SISTEMA] Iniciando pipeline de integração de dados - Projeto: {self.project_id}", flush=True)

        # 1. Carga dos dados processados (Eventos e DNA Municipal)
        df_eventos = self._ler_parquet_r2("datalake/ouro/looker_dossie_eventos.parquet")
        self._upload_table(df_eventos, "tb_dossie_eventos")
        
        df_shap_cidade = self._ler_parquet_r2("datalake/ouro/looker_dim_dna_cidade.parquet")
        self._upload_table(df_shap_cidade, "tb_dim_dna_cidade")

        # 2. Execução das rotinas de transformação no DW
        self._construir_matriz_risco_intermediaria()
        self._construir_obt_looker()

        duracao = round(time.time() - inicio_deploy, 2)
        print(f"[SISTEMA] Processo de integração finalizado. Tempo de execução: {duracao}s")
        self._notificar_webhook("[INFO] Pipeline BigQuery executado com sucesso. Tabela master estruturada e atualizada.")

if __name__ == "__main__":
    DeploySafeDriverBigQuery().executar_deploy()
