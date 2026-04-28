import os
import io
import json
import boto3
import polars as pl
import pandas as pd
import numpy as np
import time
import requests
from google.cloud import bigquery
from google.oauth2 import service_account
from botocore.config import Config
from datetime import datetime
import warnings

warnings.filterwarnings("ignore") # Proteção contra ruídos de log

class DeploySafeDriverBigQuery:
    """
    Engine de Deploy SafeDriver para Google BigQuery.
    Sincronizado com: GeradorDossieSafeDriver (Ouro Inteligente).
    Gera: tb_dossie_eventos, tb_dim_shap, tb_dim_calendario e tb_matriz_risco.
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

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=10)
            except: pass

    def _ler_parquet_r2(self, key):
        print(f"[INFO] Baixando artefato: {key}", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pl.read_parquet(io.BytesIO(obj['Body'].read())).to_pandas()

    def _upload_table(self, df_pandas, table_name):
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        self.bq_client.delete_table(table_id, not_found_ok=True)
        print(f"[INFO] Resetando tabela {table_id}...", flush=True)
        
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        job = self.bq_client.load_table_from_dataframe(df_pandas, table_id, job_config=job_config)
        job.result()
        print(f"[SUCCESS] Tabela {table_name} populada com sucesso.", flush=True)

    def _construir_matriz_risco(self):
        print("[INFO] Gerando tb_matriz_risco (Reincidência vs Periculosidade)...", flush=True)
        # O SQL usa a DATAOCORRENCIA para garantir que o Volume de Reincidência 
        # seja calculado apenas sobre fatos reais (antes de 2026).
        sql_matriz = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.tb_matriz_risco` AS
        WITH Base AS (
          SELECT 
            H3_INDEX,
            MAX(CIDADE) as CIDADE,
            MAX(BAIRRO) as BAIRRO,
            COUNT(1) as VOLUME_REINCIDENCIA,
            AVG(RISCO_PREDITO_IA) as PERICULOSIDADE_MEDIA
          FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos`
          WHERE EXTRACT(YEAR FROM DATAOCORRENCIA) < 2026 
          GROUP BY H3_INDEX
        ),
        CrimeRank AS (
          SELECT 
            H3_INDEX,
            RUBRICA,
            COUNT(1) as qtd,
            ROW_NUMBER() OVER(PARTITION BY H3_INDEX ORDER BY COUNT(1) DESC) as rnk
          FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos`
          WHERE EXTRACT(YEAR FROM DATAOCORRENCIA) < 2026
          GROUP BY H3_INDEX, RUBRICA
        )
        SELECT 
          b.*,
          c.RUBRICA as TOP_CRIME,
          CASE 
            WHEN b.VOLUME_REINCIDENCIA >= 50 AND b.PERICULOSIDADE_MEDIA >= 3.0 THEN '🔴 1 - ZONA CRÍTICA'
            WHEN b.VOLUME_REINCIDENCIA < 50  AND b.PERICULOSIDADE_MEDIA >= 3.0 THEN '🟠 2 - RISCO VITAL'
            WHEN b.VOLUME_REINCIDENCIA >= 50 AND b.PERICULOSIDADE_MEDIA < 3.0  THEN '🟡 3 - ATENÇÃO (FURTOS)'
            ELSE '🟢 4 - ZONA SEGURA'
          END AS QUADRANTE
        FROM Base b
        LEFT JOIN CrimeRank c ON b.H3_INDEX = c.H3_INDEX AND c.rnk = 1
        """
        self.bq_client.query(sql_matriz).result()

    def executar_deploy(self):
        inicio = time.time()
        print("[START] Iniciando Deploy de Inteligência Geográfica...", flush=True)

        # 1. CARGA DOS EVENTOS (Dossiê + Projeções)
        df_eventos = self._ler_parquet_r2("datalake/ouro/looker_dossie_eventos.parquet")
        
        # Alinhamento de Tipagem e Blindagem
        df_eventos['DATAOCORRENCIA'] = pd.to_datetime(df_eventos['DATAOCORRENCIA'], errors='coerce')
        df_eventos['CIDADE'] = df_eventos.get('CIDADE', 'DESCONHECIDO').fillna('DESCONHECIDO').astype(str)
        df_eventos['BAIRRO'] = df_eventos.get('BAIRRO', 'DESCONHECIDO').fillna('DESCONHECIDO').astype(str)
        df_eventos['RUBRICA'] = df_eventos.get('RUBRICA', 'DESCONHECIDO').fillna('DESCONHECIDO').astype(str)
        
        self._upload_table(df_eventos, "tb_dossie_eventos")

        # 2. CARGA DA DIMENSÃO SHAP (DNA do Crime por Bairro)
        df_shap = self._ler_parquet_r2("datalake/ouro/looker_dim_shap.parquet")
        self._upload_table(df_shap, "tb_dim_shap")

        # 3. CRIAÇÃO DE VIEWS SEMÂNTICAS
        print("[INFO] Criando vw_safedriver_dossie_master...", flush=True)
        sql_view = f"""
        CREATE OR REPLACE VIEW `{self.project_id}.{self.dataset_id}.vw_safedriver_dossie_master` AS
        SELECT 
            *,
            CASE WHEN EXTRACT(YEAR FROM DATAOCORRENCIA) >= 2026 THEN 'PREVISÃO' ELSE 'HISTÓRICO REAL' END AS ORIGEM_DADO,
            CASE
                WHEN RISCO_PREDITO_IA >= 8.5 THEN '🔴 1 - CRÍTICO'
                WHEN RISCO_PREDITO_IA >= 6.0 THEN '🟠 2 - ALTO'
                WHEN RISCO_PREDITO_IA >= 3.0 THEN '🟡 3 - MODERADO'
                ELSE '🟢 4 - BAIXO'
            END AS STATUS_ALERTA
        FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos`
        """
        self.bq_client.query(sql_view).result()

        # 4. CRIAÇÃO DO DATA MART (Matriz de Risco)
        self._construir_matriz_risco()

        duracao = round(time.time() - inicio, 2)
        msg = f"🌐 Deploy BigQuery SafeDriver Concluído em {duracao}s. Dados prontos no Looker Studio."
        print(msg)
        self._notificar_discord(msg)

if __name__ == "__main__":
    DeploySafeDriverBigQuery().executar_deploy()
