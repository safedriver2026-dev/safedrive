import os
import io
import json
import boto3
import polars as pl
import time
import requests
import warnings
from google.cloud import bigquery
from google.oauth2 import service_account
from botocore.config import Config

warnings.filterwarnings("ignore")

class DeploySafeDriverBigQuery:
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
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=10)
            except Exception: pass

    def _ler_parquet_r2(self, key):
        print(f"[SISTEMA] Extraindo artefato do repositório: {key}", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        if "LATITUDE" in df.columns:
            df = df.with_columns(pl.col("LATITUDE").cast(pl.Float64, strict=False))
        if "LONGITUDE" in df.columns:
            df = df.with_columns(pl.col("LONGITUDE").cast(pl.Float64, strict=False))
        return df.to_pandas()

    def _upload_table(self, df_pandas, table_name):
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        
        print(f"[SISTEMA] Processando carga para tabela: {table_name}", flush=True)
        job = self.bq_client.load_table_from_dataframe(df_pandas, table_id, job_config=job_config)
        job.result()
        print(f"[SISTEMA] Tabela {table_name} atualizada com {len(df_pandas)} registros.", flush=True)

    def _construir_matriz_risco_intermediaria(self):
        print("[PROCESSAMENTO] Compilando matriz de risco operacional...", flush=True)
        sql_matriz = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.tb_matriz_risco` AS
        WITH Base AS (
          SELECT H3_INDEX, COUNT(1) as VOLUME_REAL, ROUND(AVG(RISCO_IA), 2) as RISCO_MEDIO_REAL
          FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos`
          WHERE IS_MALHA = FALSE AND ANO_JOIN = 2025
          GROUP BY H3_INDEX
        ),
        CrimeRank AS (
          SELECT H3_INDEX, RUBRICA, COUNT(1) as qtd, 
                 ROW_NUMBER() OVER(PARTITION BY H3_INDEX ORDER BY COUNT(1) DESC) as rnk
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
        print("[PROCESSAMENTO] Consolidando arquitetura OBT H3-Agregada com Bounding Box SP e Exclusão Cidades Inválidas...", flush=True)
        sql_obt = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.tb_looker_master_final` AS
        WITH Base_Limpa AS (
          SELECT 
            *,
            CASE WHEN ABS(LATITUDE) > 90 THEN LATITUDE / 1000000 ELSE LATITUDE END as lat_fix,
            CASE WHEN ABS(LONGITUDE) > 180 THEN LONGITUDE / 1000000 ELSE LONGITUDE END as lon_fix,
            EXTRACT(MONTH FROM CAST(DATAOCORRENCIA AS DATE)) AS MES
          FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos`
        ),
        Base_Geo_Filtrada AS (
          SELECT 
            *,
            ST_GEOGPOINT(lon_fix, lat_fix) AS GEOMETRIA_PONTO
          FROM Base_Limpa
          WHERE lat_fix IS NOT NULL AND lon_fix IS NOT NULL
            AND lat_fix != 0.0 AND lon_fix != 0.0
            AND lat_fix BETWEEN -25.50 AND -19.50
            AND lon_fix BETWEEN -53.50 AND -44.00
            -- Nova regra: Só aceita registros onde a cidade não seja DESCONHECIDO
            -- e exista na nossa tabela de DNA de cidades (que é o espelho do shapefile SP)
            AND CIDADE != 'DESCONHECIDO'
            AND CAST(CIDADE AS STRING) IN (SELECT CAST(CIDADE AS STRING) FROM `{self.project_id}.{self.dataset_id}.tb_dim_dna_cidade`)
        ),
        Base_Agregada AS (
          SELECT 
            H3_INDEX,
            ANO_JOIN AS ANO,
            MES,
            ANY_VALUE(CIDADE) AS CIDADE,
            ANY_VALUE(BAIRRO) AS BAIRRO,
            ANY_VALUE(GEOMETRIA_PONTO) AS GEOMETRIA_PONTO,
            
            COUNT(1) AS QTD_EVENTOS_HISTORICOS,
            ROUND(AVG(RISCO_IA), 2) AS RISCO_IA_MEDIO,
            ROUND(SUM(VOLUME_TWEEDIE), 2) AS SOMA_VOLUME_TWEEDIE,
            ROUND(AVG(KPI_RISCO_EVOLUCAO), 2) AS MEDIA_RISCO_EVOLUCAO,
            
            MAX(STATUS_OPERACIONAL) AS STATUS_OPERACIONAL_PREDOMINANTE,
            MAX(CLUSTER_RANK) AS CLUSTER_RANK_PREDOMINANTE,
            LOGICAL_OR(IS_MALHA) AS TEM_PREVISAO_MALHA

          FROM Base_Geo_Filtrada
          GROUP BY H3_INDEX, ANO_JOIN, MES
        )
        
        SELECT 
            DATE(b.ANO, b.MES, 1) AS DATA_REFERENCIA_MES,
            b.ANO,
            CASE WHEN b.TEM_PREVISAO_MALHA THEN 'PREVISAO HIBRIDA' ELSE 'HISTORICO (BO)' END AS TIPO_REGISTRO,

            b.H3_INDEX, b.CIDADE, b.BAIRRO,
            b.GEOMETRIA_PONTO,

            b.QTD_EVENTOS_HISTORICOS,
            b.RISCO_IA_MEDIO,
            b.SOMA_VOLUME_TWEEDIE,
            b.MEDIA_RISCO_EVOLUCAO,
            b.STATUS_OPERACIONAL_PREDOMINANTE,
            b.CLUSTER_RANK_PREDOMINANTE,
            
            COALESCE(m.QUADRANTE, 'AREA SEM REGISTRO HISTORICO') AS QUADRANTE_RISCO,
            m.TOP_CRIME AS CRIME_PREDOMINANTE_H3,
            
            s.* EXCEPT(CIDADE)

        FROM Base_Agregada b
        LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_matriz_risco` m ON b.H3_INDEX = m.H3_INDEX
        LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_dim_dna_cidade` s ON CAST(b.CIDADE AS STRING) = CAST(s.CIDADE AS STRING) 
        """
        self.bq_client.query(sql_obt).result()

    def executar_deploy(self):
        inicio_deploy = time.time()
        print(f"[SISTEMA] Iniciando pipeline de integração de dados - Projeto: {self.project_id}", flush=True)

        df_eventos = self._ler_parquet_r2("datalake/ouro/looker_dossie_eventos.parquet")
        self._upload_table(df_eventos, "tb_dossie_eventos")
        
        df_shap_cidade = self._ler_parquet_r2("datalake/ouro/looker_dim_dna_cidade.parquet")
        self._upload_table(df_shap_cidade, "tb_dim_dna_cidade")

        self._construir_matriz_risco_intermediaria()
        self._construir_obt_looker()

        duracao = round(time.time() - inicio_deploy, 2)
        print(f"[SISTEMA] Processo de integração finalizado. Tempo de execução: {duracao}s")
        self._notificar_webhook(f"[INFO] Pipeline BigQuery OBT executado com sucesso em {duracao}s. Tabela tb_looker_master_final atualizada sem anomalias e cidades de fora.")

if __name__ == "__main__":
    DeploySafeDriverBigQuery().executar_deploy()
