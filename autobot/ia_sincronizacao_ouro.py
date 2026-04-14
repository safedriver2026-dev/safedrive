import polars as pl
import pandas as pd
import boto3
from botocore.config import Config
import joblib
import io
import os
import json
import logging
import numpy as np
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account
from autobot.calendario_estrategico import CalendarioEstrategico

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class CamadaOuroSafeDriver:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        self.project_id = os.getenv("BQ_PROJECT_ID", "").strip()
        self.dataset_id = os.getenv("BQ_DATASET_ID", "").strip()

        # Configuração S3 para Cloudflare R2
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        
        self.cal = CalendarioEstrategico()
        self.pesos = {"catboost": 0.85, "lightgbm": 0.15}
        self._conectar_bigquery()

    def _conectar_bigquery(self):
        gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
        try:
            cred_info = json.loads(gcp_json)
            credentials = service_account.Credentials.from_service_account_info(cred_info)
            self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
        except Exception as e:
            logger.error(f"OURO: Erro na autenticação GCP: {e}")

    def executar_predicao_atual(self):
        try:
            modelos = self._carregar_modelos_producao()
            df_input = self._obter_contexto_datalake()
            
            if df_input is None: 
                logger.error("OURO: Falha ao carregar dados do Data Lake.")
                return False

            # Gera predições e prepara o esquema estrela
            df_final = self._gerar_scores_ponderados(df_input, modelos)
            self._sincronizar_infraestrutura_bq(df_final)
            
            return {
                "embeds": [{"fields": [
                    {"name": "Registros Processados", "value": str(len(df_final)), "inline": True},
                    {"name": "Modelo", "value": "Tweedie Ensemble", "inline": True}
                ]}]
            }
        except Exception as e:
            logger.error(f"OURO: Erro crítico na Camada Ouro: {e}")
            return False

    def _carregar_modelos_producao(self):
        modelos = {"geral": {}}
        for alg in ["cat", "lgb"]:
            path = f"datalake/modelos_ml/latest_{alg}_geral.pkl"
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                modelos["geral"][alg] = joblib.load(io.BytesIO(obj['Body'].read()))
            except Exception as e:
                logger.warning(f"OURO: Modelo {alg} não encontrado: {e}")
                modelos["geral"][alg] = None
        return modelos

    def _obter_contexto_datalake(self):
        ano_atual = datetime.now().year
        for ano in range(ano_atual, 2021, -1):
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=f"datalake/prata/ssp_consolidada_{ano}.parquet")
                df = pl.read_parquet(io.BytesIO(resp['Body'].read())).to_pandas()
                
                # Tratamento de nulos antes da categorização
                df['DENSIDADE'] = df['DENSIDADE'].fillna(0.0)
                df['PERFIL_AREA'] = np.where(df['DENSIDADE'] > 5000, "RESIDENCIAL", 
                                    np.where(df['DENSIDADE'] == 0, "COMERCIAL_INDUSTRIAL", "MISTO"))
                
                for col in ['NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']:
                    df[col] = df[col].astype(str).fillna("DESCONHECIDO").astype('category')
                
                return df
            except Exception: continue
        return None

    def _gerar_scores_ponderados(self, df, modelos):
        features = ['DENSIDADE', 'NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']
        mults = self.cal.obter_multiplicadores()
        mods = modelos["geral"]
        
        # Predição com modelos Tweedie
        p_cat = mods["cat"].predict(df[features]) if mods["cat"] else 0
        p_lgb = mods["lgb"].predict(df[features]) if mods["lgb"] else 0
        
        score_base = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
        df['SCORE_RISCO_GERAL'] = score_base * mults.get("geral", 1.0)
        
        # Ajuste por perfil de área
        mask_comercial = df['PERFIL_AREA'] == "COMERCIAL_INDUSTRIAL"
        df.loc[mask_comercial, 'SCORE_RISCO_GERAL'] *= mults.get("comercial", 1.0)
        df.loc[~mask_comercial, 'SCORE_RISCO_GERAL'] *= mults.get("residencial", 1.0)
        
        # Normalização 0-100
        max_val = df['SCORE_RISCO_GERAL'].max() if df['SCORE_RISCO_GERAL'].max() > 0 else 1
        df['SCORE_RISCO_GERAL'] = (df['SCORE_RISCO_GERAL'] / max_val * 100).clip(0, 100).round(2)
        df['DT_ULTIMA_SINCRONIZACAO'] = datetime.now()
        
        return df[['H3_INDEX', 'SCORE_RISCO_GERAL', 'NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA', 'DENSIDADE', 'DT_ULTIMA_SINCRONIZACAO']]

    def _sincronizar_infraestrutura_bq(self, df):
        dataset_path = f"{self.project_id}.{self.dataset_id}"
        tabela_fato = f"{dataset_path}.fato_risco_h3_atual"
        tabela_staging = f"{tabela_fato}_staging"
        
        # 1. Upload para Staging
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df, tabela_staging, job_config=job_config).result()

        # 2. Criação/Merge da Tabela Fato
        try:
            self.bq_client.get_table(tabela_fato)
        except NotFound:
            self.bq_client.query(f"CREATE TABLE `{tabela_fato}` AS SELECT H3_INDEX, SCORE_RISCO_GERAL, DT_ULTIMA_SINCRONIZACAO FROM `{tabela_staging}`").result()

        # Merge de atualização
        self.bq_client.query(f"""
            MERGE `{tabela_fato}` T USING `{tabela_staging}` S ON T.H3_INDEX = S.H3_INDEX
            WHEN MATCHED THEN UPDATE SET T.SCORE_RISCO_GERAL = S.SCORE_RISCO_GERAL, T.DT_ULTIMA_SINCRONIZACAO = S.DT_ULTIMA_SINCRONIZACAO
            WHEN NOT MATCHED THEN INSERT (H3_INDEX, SCORE_RISCO_GERAL, DT_ULTIMA_SINCRONIZACAO) VALUES(S.H3_INDEX, S.SCORE_RISCO_GERAL, S.DT_ULTIMA_SINCRONIZACAO)
        """).result()

        # 3. Construção Automática das Dimensões e View
        logger.info("OURO: Gerando Dimensões e View Analítica...")
        self._build_star_schema(dataset_path, tabela_staging, tabela_fato)

    def _build_star_schema(self, dataset, staging, fato):
        # Dimensão Geográfica
        self.bq_client.query(f"""
            CREATE OR REPLACE TABLE `{dataset}.dim_h3` AS
            SELECT DISTINCT H3_INDEX, NM_BAIRRO, NM_MUN as CIDADE, PERFIL_AREA, DENSIDADE FROM `{staging}`
        """).result()

        # Dimensão Tempo
        self.bq_client.query(f"""
            CREATE OR REPLACE TABLE `{dataset}.dim_tempo` AS
            SELECT d as DATA, EXTRACT(DAYOFWEEK FROM d) as DIA_SEMANA, FORMAT_DATE('%A', d) as NOME_DIA,
            IF(EXTRACT(DAY FROM d) BETWEEN 1 AND 10, TRUE, FALSE) as SEMANA_PAGAMENTO
            FROM UNNEST(GENERATE_DATE_ARRAY('2022-01-01', '2026-12-31')) d
        """).result()

        # View Analítica Final (O que o Streamlit consome)
        self.bq_client.query(f"""
            CREATE OR REPLACE VIEW `{dataset}.v_safedriver_analitico` AS
            SELECT f.H3_INDEX, f.SCORE_RISCO_GERAL, g.NM_BAIRRO, g.CIDADE, g.PERFIL_AREA, g.DENSIDADE,
            t.NOME_DIA, t.SEMANA_PAGAMENTO, f.DT_ULTIMA_SINCRONIZACAO
            FROM `{fato}` f
            JOIN `{dataset}.dim_h3` g ON f.H3_INDEX = g.H3_INDEX
            LEFT JOIN `{dataset}.dim_tempo` t ON DATE(f.DT_ULTIMA_SINCRONIZACAO) = t.DATA
        """).result()
