import os
import io
import json
import boto3
import polars as pl
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from botocore.config import Config
from datetime import datetime

class DeploySafeDriverBigQuery:
    def __init__(self):
        self.project_id = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9")
        self.dataset_id = os.getenv("BQ_DATASET_ID")
        
        if not self.dataset_id:
            raise ValueError("A variavel BQ_DATASET_ID precisa estar configurada.")
            
        bq_json_str = os.getenv("BQ_SERVICE_ACCOUNT_JSON")
        if not bq_json_str:
            raise ValueError("Credenciais de servico do BigQuery ausentes.")
            
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

    def _ler_parquet_r2(self, key):
        print(f"Acessando artefato: {key}", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        df_pandas = pl.read_parquet(io.BytesIO(obj['Body'].read())).to_pandas()
        return df_pandas.fillna(pd.NA)

    def _upload_table(self, df_pandas, table_name):
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        print(f"Subindo tabela {table_id}...", flush=True)
        job = self.bq_client.load_table_from_dataframe(df_pandas, table_id, job_config=job_config)
        job.result()
        print(f"✅ Tabela {table_name} atualizada.", flush=True)

    def _construir_dim_calendario(self):
        print("\nGerando Dimensao Calendario nativa (2020-2030)...", flush=True)
        sql = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.tb_dim_calendario` AS
        WITH datas AS (
            SELECT dt AS DATA_BASE
            FROM UNNEST(GENERATE_DATE_ARRAY('2020-01-01', '2030-12-31', INTERVAL 1 DAY)) AS dt
        )
        SELECT
            DATA_BASE,
            EXTRACT(YEAR FROM DATA_BASE) AS ANO,
            CASE EXTRACT(MONTH FROM DATA_BASE)
                WHEN 1 THEN 'Janeiro' WHEN 2 THEN 'Fevereiro' WHEN 3 THEN 'Março'
                WHEN 4 THEN 'Abril' WHEN 5 THEN 'Maio' WHEN 6 THEN 'Junho'
                WHEN 7 THEN 'Julho' WHEN 8 THEN 'Agosto' WHEN 9 THEN 'Setembro'
                WHEN 10 THEN 'Outubro' WHEN 11 THEN 'Novembro' WHEN 12 THEN 'Dezembro'
            END AS NOME_MES_PT,
            CASE WHEN EXTRACT(DAYOFWEEK FROM DATA_BASE) IN (1, 7) THEN 'FIM DE SEMANA' ELSE 'DIA UTIL' END AS CAL_TIPO_DIA
        FROM datas;
        """
        self.bq_client.query(sql).result()

    def executar_deploy(self):
        print("Iniciando Deploy Analitico...", flush=True)

        # 1. Carga Fato e Dimensão SHAP
        df_eventos = self._ler_parquet_r2("datalake/ouro/looker_dossie_eventos.parquet")
        if 'DATAOCORRENCIA' in df_eventos.columns:
            df_eventos['DATAOCORRENCIA'] = pd.to_datetime(df_eventos['DATAOCORRENCIA'], errors='coerce')
        self._upload_table(df_eventos, "tb_dossie_eventos")

        df_shap = self._ler_parquet_r2("datalake/ouro/looker_dim_shap.parquet")
        self._upload_table(df_shap, "tb_dim_shap")

        # 2. Calendário
        self._construir_dim_calendario()

        # 3. Master View Arquitetural Definitiva
        print("\nConstruindo Master View Analítica...", flush=True)
        sql_view = f"""
        CREATE OR REPLACE VIEW `{self.project_id}.{self.dataset_id}.vw_safedriver_dossie_master` AS

        WITH Base_Enriquecida AS (
            SELECT 
                e.* EXCEPT(DATAOCORRENCIA), 
                DATE(e.DATAOCORRENCIA) AS DATA_FATO,
                ST_GEOGPOINT(CAST(e.LONGITUDE AS FLOAT64), CAST(e.LATITUDE AS FLOAT64)) AS GEOMETRIA_PONTO,
                s.* EXCEPT(CIDADE, BAIRRO),
                cal.* EXCEPT(DATA_BASE)
            FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos` e
            LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_dim_shap` s 
                ON e.CIDADE = s.CIDADE AND e.BAIRRO = s.BAIRRO
            LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_dim_calendario` cal
                ON DATE(e.DATAOCORRENCIA) = cal.DATA_BASE
        )

        SELECT
            *,
            -- A. STATUS DE PREDIÇÃO (Filtro Rápido: Realidade vs Futuro)
            CASE 
                WHEN LABEL_PESO_RISCO IS NULL OR LABEL_PESO_RISCO = 0 THEN 'PREVISÃO' 
                ELSE 'HISTÓRICO REAL' 
            END AS STATUS_DADO,

            -- B. DELTA DE PRECISÃO (Avaliação do Erro do Modelo)
            CASE 
                WHEN LABEL_PESO_RISCO > 0 THEN (RISCO_PREDITO_IA - LABEL_PESO_RISCO)
                ELSE NULL 
            END AS DELTA_IA_REAL,

            -- C. CLASSIFICAÇÃO SEMÂNTICA (Cores Rápidas no Mapa)
            CASE
                WHEN RISCO_PREDITO_IA >= 7.0 THEN '1 - CRÍTICO'
                WHEN RISCO_PREDITO_IA >= 4.0 THEN '2 - ALTO'
                WHEN RISCO_PREDITO_IA >= 2.0 THEN '3 - MODERADO'
                ELSE '4 - BAIXO'
            END AS NIVEL_ALERTA,

            -- D. QUALIDADE DA INFERÊNCIA (Auditoria do Algoritmo)
            CASE
                WHEN LABEL_PESO_RISCO = 0 OR LABEL_PESO_RISCO IS NULL THEN 'N/A (Previsão)'
                WHEN ABS(RISCO_PREDITO_IA - LABEL_PESO_RISCO) <= 1.5 THEN 'ALTA PRECISÃO'
                WHEN (RISCO_PREDITO_IA - LABEL_PESO_RISCO) > 1.5 THEN 'FALSO POSITIVO (Alarmista)'
                ELSE 'FALSO NEGATIVO (Subestimado)'
            END AS QUALIDADE_PREDICAO

        FROM Base_Enriquecida;
        """
        self.bq_client.query(sql_view).result()
        print(f"✨ Deploy finalizado. A melhor Master View possivel esta pronta no BQ.")

if __name__ == "__main__":
    DeploySafeDriverBigQuery().executar_deploy()
