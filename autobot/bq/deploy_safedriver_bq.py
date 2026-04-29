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
    Engine de Deploy SafeDriver para Google BigQuery.
    Arquitetura: OBT Dimensional alinhada com a Estratégia BQ Free (2025-2026).
    Atualizado: Suporte aos KPIs fidedignos (KPI_RISCO_MEDIO e KPI_VOLUME).
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
        
        print(f"[INFO] Fazendo upload para o BigQuery: {table_name}...", flush=True)
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE", autodetect=True)
        job = self.bq_client.load_table_from_dataframe(df_pandas, table_id, job_config=job_config)
        job.result()
        print(f"[SUCCESS] {table_name} populada com {len(df_pandas)} linhas.", flush=True)

    def _construir_matriz_risco_intermediaria(self):
        print("[INFO] Gerando tb_matriz_risco (Apoio - Foco 2025)...", flush=True)
        sql_matriz = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.tb_matriz_risco` AS
        WITH Base_Fix AS (
          SELECT *,
            CASE WHEN ABS(CAST(LATITUDE AS FLOAT64)) > 90 THEN CAST(LATITUDE AS FLOAT64) / 100000000 ELSE CAST(LATITUDE AS FLOAT64) END as lat_fix,
            CASE WHEN ABS(CAST(LONGITUDE AS FLOAT64)) > 180 THEN CAST(LONGITUDE AS FLOAT64) / 100000000 ELSE CAST(LONGITUDE AS FLOAT64) END as lon_fix
          FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos`
        ),
        Base AS (
          SELECT H3_INDEX, COUNT(1) as VOLUME_REINCIDENCIA, AVG(RISCO_PREDITO_IA) as PERICULOSIDADE_MEDIA
          FROM Base_Fix WHERE EXTRACT(YEAR FROM DATAOCORRENCIA) = 2025 GROUP BY H3_INDEX
        ),
        CrimeRank AS (
          SELECT H3_INDEX, RUBRICA, COUNT(1) as qtd, ROW_NUMBER() OVER(PARTITION BY H3_INDEX ORDER BY COUNT(1) DESC) as rnk
          FROM Base_Fix WHERE EXTRACT(YEAR FROM DATAOCORRENCIA) = 2025 GROUP BY H3_INDEX, RUBRICA
        )
        SELECT 
          b.H3_INDEX, b.VOLUME_REINCIDENCIA, b.PERICULOSIDADE_MEDIA, c.RUBRICA as TOP_CRIME,
          CASE 
            WHEN b.VOLUME_REINCIDENCIA >= 50 AND b.PERICULOSIDADE_MEDIA >= 3.0 THEN '🔴 1 - ZONA CRÍTICA'
            WHEN b.VOLUME_REINCIDENCIA < 50  AND b.PERICULOSIDADE_MEDIA >= 3.0 THEN '🟠 2 - RISCO VITAL'
            WHEN b.VOLUME_REINCIDENCIA >= 50 AND b.PERICULOSIDADE_MEDIA < 3.0  THEN '🟡 3 - ATENÇÃO (FURTOS)'
            ELSE '🟢 4 - ZONA SEGURA'
          END AS QUADRANTE
        FROM Base b LEFT JOIN CrimeRank c ON b.H3_INDEX = c.H3_INDEX AND c.rnk = 1
        """
        self.bq_client.query(sql_matriz).result()

    def _construir_obt_looker(self):
        print("[INFO] Fundindo dados na OBT final...", flush=True)
        sql_obt = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.tb_looker_master_final` AS
        WITH Base_Fix AS (
          SELECT *,
            CASE WHEN ABS(CAST(LATITUDE AS FLOAT64)) > 90 THEN CAST(LATITUDE AS FLOAT64) / 100000000 ELSE CAST(LATITUDE AS FLOAT64) END as lat_fix,
            CASE WHEN ABS(CAST(LONGITUDE AS FLOAT64)) > 180 THEN CAST(LONGITUDE AS FLOAT64) / 100000000 ELSE CAST(LONGITUDE AS FLOAT64) END as lon_fix
          FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos`
        )
        SELECT 
            -- 1. TEMPO (O Eixo Perfeito de Mensal/Sazonalidade)
            e.DATAOCORRENCIA,
            DATE_TRUNC(CAST(e.DATAOCORRENCIA AS DATE), MONTH) AS DATA_REFERENCIA_MES,
            EXTRACT(YEAR FROM e.DATAOCORRENCIA) AS ANO,
            EXTRACT(MONTH FROM e.DATAOCORRENCIA) AS MES_NUMERO,
            CASE EXTRACT(MONTH FROM e.DATAOCORRENCIA)
                WHEN 1 THEN 'Janeiro' WHEN 2 THEN 'Fevereiro' WHEN 3 THEN 'Março'
                WHEN 4 THEN 'Abril' WHEN 5 THEN 'Maio' WHEN 6 THEN 'Junho'
                WHEN 7 THEN 'Julho' WHEN 8 THEN 'Agosto' WHEN 9 THEN 'Setembro'
                WHEN 10 THEN 'Outubro' WHEN 11 THEN 'Novembro' WHEN 12 THEN 'Dezembro'
            END AS MES_NOME,
            CASE WHEN EXTRACT(YEAR FROM e.DATAOCORRENCIA) >= 2026 THEN 'PREVISÃO' ELSE 'HISTÓRICO REAL' END AS ORIGEM_DADO,

            -- 2. GEOGRAFIA (Hierarquia Completa)
            e.H3_INDEX, e.CIDADE, e.BAIRRO, e.LOGRADOURO,
            CASE WHEN e.lat_fix BETWEEN -90 AND 90 AND e.lon_fix BETWEEN -180 AND 180 
                 THEN ST_GEOGPOINT(e.lon_fix, e.lat_fix) ELSE NULL END AS GEOMETRIA_PONTO,

            -- 3. CONTEXTO E RISCO IA E KPIS HOMOGENEIZADOS
            e.RUBRICA AS TIPO_CRIME, e.SAZON_PERIODO AS PERIODO_DIA, 
            e.FEAT_CONTEXTO_CRITICO AS CENARIO_COMPLETO,
            e.RISCO_PREDITO_IA AS RISCO_EXPOSICAO,
            e.KPI_RISCO_MEDIO,
            e.KPI_VOLUME,
            
            CASE
                WHEN e.RISCO_PREDITO_IA >= 8.5 THEN '🔴 1 - CRÍTICO'
                WHEN e.RISCO_PREDITO_IA >= 6.0 THEN '🟠 2 - ALTO'
                WHEN e.RISCO_PREDITO_IA >= 3.0 THEN '🟡 3 - MODERADO'
                ELSE '🟢 4 - BAIXO'
            END AS STATUS_ALERTA,

            -- 4. MATRIZ E SHAP
            COALESCE(m.QUADRANTE, '⚪ SEM CLASSIFICAÇÃO') AS QUADRANTE_RISCO,
            m.TOP_CRIME AS CRIME_PREDOMINANTE_H3,
            s.* EXCEPT(CIDADE, BAIRRO)

        FROM Base_Fix e
        LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_matriz_risco` m ON e.H3_INDEX = m.H3_INDEX
        LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_dim_shap` s ON CAST(e.CIDADE AS STRING) = CAST(s.CIDADE AS STRING) 
                                                                   AND CAST(e.BAIRRO AS STRING) = CAST(s.BAIRRO AS STRING);
        """
        self.bq_client.query(sql_obt).result()

    def executar_deploy(self):
        inicio_deploy = time.time()
        print("[START] Iniciando Deploy SafeDriver Arquitetura OBT (BQ Free)...", flush=True)

        # 1. Carregar Eventos e TRATAR COLUNAS
        df_eventos = self._ler_parquet_r2("datalake/ouro/looker_dossie_eventos.parquet")
        df_eventos['DATAOCORRENCIA'] = pd.to_datetime(df_eventos['DATAOCORRENCIA'], errors='coerce')
        
        # Lógica Robusta para Logradouro
        if 'LOGRADOURO' not in df_eventos.columns:
            if 'RUA' in df_eventos.columns:
                df_eventos['LOGRADOURO'] = df_eventos['RUA']
            else:
                df_eventos['LOGRADOURO'] = 'NÃO INFORMADO'
        
        df_eventos['LOGRADOURO'] = df_eventos['LOGRADOURO'].fillna('NÃO INFORMADO').astype(str)
        df_eventos['CIDADE'] = df_eventos.get('CIDADE', 'DESCONHECIDO').fillna('DESCONHECIDO').astype(str)
        df_eventos['BAIRRO'] = df_eventos.get('BAIRRO', 'DESCONHECIDO').fillna('DESCONHECIDO').astype(str)
        df_eventos['RUBRICA'] = df_eventos.get('RUBRICA', 'DESCONHECIDO').fillna('DESCONHECIDO').astype(str)
        
        self._upload_table(df_eventos, "tb_dossie_eventos")

        # Libera a RAM pesada antes de prosseguir
        del df_eventos
        gc.collect()

        # 2. Carregar SHAP
        df_shap = self._ler_parquet_r2("datalake/ouro/looker_dim_shap.parquet")
        self._upload_table(df_shap, "tb_dim_shap")
        
        del df_shap
        gc.collect()

        # 3. Rodar SQLs de Fusão no BigQuery (O trabalho pesado acontece na nuvem agora)
        self._construir_matriz_risco_intermediaria()
        self._construir_obt_looker()

        duracao = round(time.time() - inicio_deploy, 2)
        print(f"[SUCCESS] Deploy Concluído em {duracao}s!")
        self._notificar_discord(f"🌐 Deploy Finalizado em {duracao}s! Tabela OBT Biênio (2025-2026) disponível no BigQuery com KPIs Homogeneizados.")

if __name__ == "__main__":
    DeploySafeDriverBigQuery().executar_deploy()
