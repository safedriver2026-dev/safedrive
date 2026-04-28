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
    Inclui suporte nativo a Geometria (ST_GEOGPOINT) e Curadoria de Colunas.
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
        print(f"[INFO] Baixando artefato do R2: {key}", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pl.read_parquet(io.BytesIO(obj['Body'].read())).to_pandas()

    def _upload_table(self, df_pandas, table_name):
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        self.bq_client.delete_table(table_id, not_found_ok=True)
        print(f"[INFO] Resetando tabela {table_id} no BigQuery...", flush=True)
        
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        job = self.bq_client.load_table_from_dataframe(df_pandas, table_id, job_config=job_config)
        job.result()
        print(f"[SUCCESS] Tabela {table_name} populada com {len(df_pandas)} linhas.", flush=True)

    def _construir_dim_calendario(self):
        print("[INFO] Gerando Dimensão Calendário via SQL...", flush=True)
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
        print("[INFO] Gerando tb_matriz_risco (Agregada por Hexágono H3)...", flush=True)
        sql_matriz = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.tb_matriz_risco` AS
        WITH Base AS (
          SELECT 
            H3_INDEX,
            MAX(CIDADE) as CIDADE,
            MAX(BAIRRO) as BAIRRO,
            COUNT(1) as VOLUME_REINCIDENCIA,
            AVG(RISCO_PREDITO_IA) as PERICULOSIDADE_MEDIA,
            ANY_VALUE(ST_GEOGPOINT(CAST(LONGITUDE AS FLOAT64), CAST(LATITUDE AS FLOAT64))) as GEOMETRIA_H3
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
          b.H3_INDEX,
          b.CIDADE,
          b.BAIRRO,
          b.VOLUME_REINCIDENCIA,
          b.PERICULOSIDADE_MEDIA,
          c.RUBRICA as TOP_CRIME,
          b.GEOMETRIA_H3,
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
        inicio_deploy = time.time()
        print("[START] Iniciando Deploy SafeDriver...", flush=True)

        # 1. PROCESSAMENTO DOS EVENTOS
        df_eventos = self._ler_parquet_r2("datalake/ouro/looker_dossie_eventos.parquet")
        
        # Blindagem de campos e alinhamento de tipos
        df_eventos['DATAOCORRENCIA'] = pd.to_datetime(df_eventos['DATAOCORRENCIA'], errors='coerce')
        df_eventos['CIDADE'] = df_eventos.get('CIDADE', 'DESCONHECIDO').fillna('DESCONHECIDO').astype(str)
        df_eventos['BAIRRO'] = df_eventos.get('BAIRRO', 'DESCONHECIDO').fillna('DESCONHECIDO').astype(str)
        df_eventos['RUBRICA'] = df_eventos.get('RUBRICA', 'DESCONHECIDO').fillna('DESCONHECIDO').astype(str)
        
        # Reconstrução da Massa Criminal
        if 'FS_VOL_CRIMES_ANO_ANT' in df_eventos.columns:
            df_eventos['FS_VOL_CRIMES_ANO_ANT'] = df_eventos['FS_VOL_CRIMES_ANO_ANT'].fillna(0.0).astype(float)
        else:
            df_eventos['FS_VOL_CRIMES_ANO_ANT'] = 0.0

        self._upload_table(df_eventos, "tb_dossie_eventos")

        # 2. DIMENSÃO SHAP
        df_shap = self._ler_parquet_r2("datalake/ouro/looker_dim_shap.parquet")
        self._upload_table(df_shap, "tb_dim_shap")

        # 3. CONSTRUÇÃO DA CAMADA SEMÂNTICA (VIEW CURADA)
        # O "SELECT *" foi trocado por uma seleção explícita das colunas que importam.
        print("[INFO] Criando Master View Curada com GEOMETRIA...", flush=True)
        sql_view = f"""
        CREATE OR REPLACE VIEW `{self.project_id}.{self.dataset_id}.vw_safedriver_dossie_master` AS
        SELECT 
            -- Chaves e Geografia
            ID_BO,
            DATAOCORRENCIA,
            H3_INDEX,
            CIDADE,
            BAIRRO,
            ST_GEOGPOINT(CAST(LONGITUDE AS FLOAT64), CAST(LATITUDE AS FLOAT64)) AS GEOMETRIA_PONTO,
            
            -- Contexto do Crime
            RUBRICA AS TIPO_CRIME,
            FEAT_PERFIL_VITIMA AS PERFIL_VITIMA,
            SAZON_PERIODO AS PERIODO_DIA,
            FEAT_TIPO_DIA AS TIPO_DIA,
            FEAT_CONTEXTO_CRITICO AS CENARIO_COMPLETO,
            
            -- Inteligência e Risco (O Coração da IA)
            FS_VOL_CRIMES_ANO_ANT AS VOLUME_HISTORICO_LOCAL,
            RISCO_PREDITO_IA AS RISCO_EXPOSICAO,
            
            -- Classificações para Filtros no Looker
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

        # 4. DATA MARTS ADICIONAIS
        self._construir_dim_calendario()
        self._construir_matriz_risco()

        duracao = round(time.time() - inicio_deploy, 2)
        
        # Report Final
        report = (
            f"==============================================================\n"
            f" 🌐 DEPLOY BIGQUERY CONCLUÍDO - {self.projeto.upper()} \n"
            f"==============================================================\n"
            f"📍 Artefatos Disponíveis:\n"
            f"   • Tabela Fato: tb_dossie_eventos ({len(df_eventos):,} linhas)\n"
            f"   • View Master: vw_safedriver_dossie_master (View Curada)\n"
            f"   • Data Mart: tb_matriz_risco (4 Quadrantes)\n"
            f"   • Calendário: tb_dim_calendario (2020-2030)\n"
            f"==============================================================\n"
            f"Tempo Total: {duracao}s | Pronto para Storytelling\n"
            f"==============================================================\n"
        )
        print(report)
        self._notificar_discord(f"```text\n{report}\n```")

if __name__ == "__main__":
    DeploySafeDriverBigQuery().executar_deploy()
