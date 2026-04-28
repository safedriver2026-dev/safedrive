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

warnings.filterwarnings("ignore") # Esconde warnings do Pandas/GBQ

class DeploySafeDriverBigQuery:
    """
    Engine de Deploy SafeDriver para Google BigQuery.
    Blindagem Extrema: Reconstrução On-The-Fly + Matriz de Risco em SQL.
    """
    def __init__(self):
        self.projeto = "SafeDriver"
        self.project_id = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9")
        self.dataset_id = os.getenv("BQ_DATASET_ID")
        
        if not self.dataset_id:
            raise ValueError("Variavel BQ_DATASET_ID nao configurada.")
            
        bq_json_str = os.getenv("BQ_SERVICE_ACCOUNT_JSON")
        if not bq_json_str:
            raise ValueError("Credenciais de servico BigQuery ausentes.")
            
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
        print(f"[INFO] Acessando artefato R2: {key}", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pl.read_parquet(io.BytesIO(obj['Body'].read())).to_pandas()

    def _upload_table(self, df_pandas, table_name):
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        self.bq_client.delete_table(table_id, not_found_ok=True)
        print(f"[INFO] Schema de {table_id} resetado no BigQuery.", flush=True)
        
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        print(f"[INFO] Uploading dataframe para {table_id}...", flush=True)
        job = self.bq_client.load_table_from_dataframe(df_pandas, table_id, job_config=job_config)
        job.result()
        print(f"[SUCCESS] Tabela {table_name} criada e populada.", flush=True)

    def _construir_dim_calendario(self):
        print("[INFO] Gerando Dimensao Calendario...", flush=True)
        sql = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.tb_dim_calendario` AS
        WITH datas AS (
            SELECT dt AS DATA_BASE
            FROM UNNEST(GENERATE_DATE_ARRAY('2020-01-01', '2030-12-31', INTERVAL 1 DAY)) AS dt
        )
        SELECT
            DATA_BASE,
            EXTRACT(YEAR FROM DATA_BASE) AS CAL_ANO,
            EXTRACT(MONTH FROM DATA_BASE) AS CAL_MES,
            FORMAT_DATE('%B', DATA_BASE) AS CAL_NOME_MES,
            CASE WHEN EXTRACT(DAYOFWEEK FROM DATA_BASE) IN (1, 7) THEN 'FIM DE SEMANA' ELSE 'DIA UTIL' END AS CAL_TIPO_DIA
        FROM datas;
        """
        self.bq_client.query(sql).result()

    def _construir_matriz_risco(self):
        print("[INFO] Gerando Data Mart da Matriz de Risco (Reincidencia vs Periculosidade)...", flush=True)
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
          WHERE ANO_JOIN != 2026 -- Mede a reincidência apenas com dados reais do passado
          GROUP BY H3_INDEX
        ),
        CrimeRank AS (
          SELECT 
            H3_INDEX,
            RUBRICA,
            COUNT(1) as qtd,
            ROW_NUMBER() OVER(PARTITION BY H3_INDEX ORDER BY COUNT(1) DESC) as rnk
          FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos`
          WHERE ANO_JOIN != 2026
          GROUP BY H3_INDEX, RUBRICA
        )
        SELECT 
          b.H3_INDEX,
          b.CIDADE,
          b.BAIRRO,
          b.VOLUME_REINCIDENCIA,
          b.PERICULOSIDADE_MEDIA,
          c.RUBRICA as TOP_CRIME,
          -- Definindo os 4 Quadrantes (Ajuste o limiar 50/3.0 conforme a realidade do seu Looker)
          CASE 
            WHEN b.VOLUME_REINCIDENCIA >= 50 AND b.PERICULOSIDADE_MEDIA >= 3.0 THEN '🔴 1 - ZONA CRÍTICA'
            WHEN b.VOLUME_REINCIDENCIA < 50  AND b.PERICULOSIDADE_MEDIA >= 3.0 THEN '🟠 2 - RISCO VITAL (Severo)'
            WHEN b.VOLUME_REINCIDENCIA >= 50 AND b.PERICULOSIDADE_MEDIA < 3.0  THEN '🟡 3 - ATENÇÃO (Furtos)'
            ELSE '🟢 4 - ZONA SEGURA'
          END AS QUADRANTE
        FROM Base b
        LEFT JOIN CrimeRank c ON b.H3_INDEX = c.H3_INDEX AND c.rnk = 1
        """
        self.bq_client.query(sql_matriz).result()
        print("[SUCCESS] tb_matriz_risco criada com sucesso!", flush=True)

    def executar_deploy(self):
        inicio_deploy = time.time()
        print("[START] Iniciando Deploy de Inteligencia Geografica...", flush=True)

        # =====================================================================
        # 1. DOSSIÊ EVENTOS (Blindagem e Reconstrução On-The-Fly)
        # =====================================================================
        df_eventos = self._ler_parquet_r2("datalake/ouro/looker_dossie_eventos.parquet")
        linhas_eventos = len(df_eventos)
        
        if 'DATAOCORRENCIA' in df_eventos.columns:
            df_eventos['DATAOCORRENCIA'] = pd.to_datetime(df_eventos['DATAOCORRENCIA'], errors='coerce')
        
        if 'CIDADE' in df_eventos.columns:
            df_eventos['CIDADE'] = df_eventos['CIDADE'].fillna('DESCONHECIDO').astype(str)
        if 'BAIRRO' in df_eventos.columns:
            df_eventos['BAIRRO'] = df_eventos['BAIRRO'].fillna('DESCONHECIDO').astype(str)
            
        # Reconstrução da Massa Criminal
        if 'FS_VOL_CRIMES_ANO_ANT' not in df_eventos.columns:
            print("[WARN] Coluna de volume ausente. Injetando valores fallback (0.0).", flush=True)
            df_eventos['FS_VOL_CRIMES_ANO_ANT'] = 0.0
        else:
            df_eventos['FS_VOL_CRIMES_ANO_ANT'] = df_eventos['FS_VOL_CRIMES_ANO_ANT'].fillna(0.0).astype(float)
            
        # Reconstrução do Contexto Crítico
        if 'FEAT_CONTEXTO_CRITICO' not in df_eventos.columns:
            print("[WARN] Coluna FEAT_CONTEXTO_CRITICO ausente. Reconstruindo na memoria...", flush=True)
            if 'SAZON_PERIODO' in df_eventos.columns and 'FEAT_PERFIL_VITIMA' in df_eventos.columns:
                df_eventos['FEAT_CONTEXTO_CRITICO'] = df_eventos['SAZON_PERIODO'].astype(str) + "_" + df_eventos['FEAT_PERFIL_VITIMA'].astype(str)
            else:
                df_eventos['FEAT_CONTEXTO_CRITICO'] = 'DESCONHECIDO'
            
        self._upload_table(df_eventos, "tb_dossie_eventos")

        # =====================================================================
        # 2. DIMENSÃO SHAP
        # =====================================================================
        df_shap = self._ler_parquet_r2("datalake/ouro/looker_dim_shap.parquet")
        linhas_shap = len(df_shap)
        
        if 'CIDADE' in df_shap.columns:
            df_shap['CIDADE'] = df_shap['CIDADE'].fillna('DESCONHECIDO').astype(str)
        if 'BAIRRO' in df_shap.columns:
            df_shap['BAIRRO'] = df_shap['BAIRRO'].fillna('DESCONHECIDO').astype(str)
            
        self._upload_table(df_shap, "tb_dim_shap")

        # =====================================================================
        # 3. CONSTRUÇÃO DE VIEWS E DATA MARTS (SQL)
        # =====================================================================
        self._construir_dim_calendario()

        print("[INFO] Construindo Master View Semantica (vw_safedriver_dossie_master)...", flush=True)
        sql_view = f"""
        CREATE OR REPLACE VIEW `{self.project_id}.{self.dataset_id}.vw_safedriver_dossie_master` AS

        WITH Base_Final AS (
            SELECT 
                e.* EXCEPT(DATAOCORRENCIA), 
                DATE(e.DATAOCORRENCIA) AS DATA_FATO,
                ST_GEOGPOINT(CAST(e.LONGITUDE AS FLOAT64), CAST(e.LATITUDE AS FLOAT64)) AS GEOMETRIA_PONTO,
                s.* EXCEPT(CIDADE, BAIRRO),
                cal.* EXCEPT(DATA_BASE)
            FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos` e
            LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_dim_shap` s 
                ON CAST(e.CIDADE AS STRING) = CAST(s.CIDADE AS STRING) 
                AND CAST(e.BAIRRO AS STRING) = CAST(s.BAIRRO AS STRING)
            LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_dim_calendario` cal
                ON DATE(e.DATAOCORRENCIA) = cal.DATA_BASE
        )

        SELECT
            *,
            CASE 
                WHEN EXTRACT(YEAR FROM DATA_FATO) >= 2026 THEN 'PREVISÃO' 
                ELSE 'HISTÓRICO REAL' 
            END AS ORIGEM_DADO,

            RISCO_PREDITO_IA AS RISCO_EXPOSICAO_FINAL,
            LABEL_PESO_RISCO AS SEVERIDADE_CRIME_BASE,
            FS_VOL_CRIMES_ANO_ANT AS FREQUENCIA_CRIMINAL_HIST,

            CASE
                WHEN RISCO_PREDITO_IA >= 8.5 THEN '🔴 1 - CRÍTICO (ZONA VERMELHA)'
                WHEN RISCO_PREDITO_IA >= 6.0 THEN '🟠 2 - ALTO (ATENÇÃO MÁXIMA)'
                WHEN RISCO_PREDITO_IA >= 3.0 THEN '🟡 3 - MODERADO (ESTADO DE ALERTA)'
                ELSE '🟢 4 - BAIXO (RISCO RESIDUAL)'
            END AS STATUS_ALERTA,

            UPPER(REPLACE(FEAT_CONTEXTO_CRITICO, '_', ' ')) AS CENARIO_AVALIADO

        FROM Base_Final;
        """
        self.bq_client.query(sql_view).result()
        
        # Cria a Matriz de Risco no BigQuery
        self._construir_matriz_risco()
        
        duracao = round(time.time() - inicio_deploy, 2)
        print("[SUCCESS] Deploy Finalizado sem erros de schema ou tipagem!")
        
        # Report Final no Discord
        report = (
            f"==============================================================\n"
            f" 🌐 DEPLOY BIGQUERY CONCLUÍDO - {self.projeto.upper()} \n"
            f"==============================================================\n"
            f"📍 Camada Semântica Atualizada:\n"
            f"   • Tabela Fato: {linhas_eventos:,} linhas\n"
            f"   • Dimensão SHAP: {linhas_shap:,} linhas\n"
            f"   • Data Mart: Matriz de Risco H3 Gerada via SQL\n"
            f"==============================================================\n"
            f"Tempo de Deploy: {duracao}s | Disponível no Looker Studio\n"
            f"==============================================================\n"
        )
        print(report)
        self._notificar_discord(f"```text\n{report}\n```")

if __name__ == "__main__":
    DeploySafeDriverBigQuery().executar_deploy()
