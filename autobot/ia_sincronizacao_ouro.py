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
                logger.error("OURO: Não foi possível carregar os dados de contexto do R2.")
                return False

            df_final = self._gerar_scores_ponderados(df_input, modelos)
            self._sincronizar_fato_risco(df_final)
            
            return {
                "embeds": [{"fields": [{}, {}, {"value": str(len(df_final))}]}]
            }
        except Exception as e:
            logger.error(f"OURO: Falha na Camada Ouro: {e}")
            return False

    def _carregar_modelos_producao(self):
        modelos = {"geral": {}}
        for alg in ["cat", "lgb"]:
            path = f"datalake/modelos_ml/latest_{alg}_geral.pkl"
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                modelos["geral"][alg] = joblib.load(io.BytesIO(obj['Body'].read()))
            except Exception as e:
                logger.warning(f"OURO: Não achou o modelo {alg}: {e}")
                modelos["geral"][alg] = None
        return modelos

    def _obter_contexto_datalake(self):
        ano_atual = datetime.now().year
        for ano in range(ano_atual, 2021, -1):
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=f"datalake/prata/ssp_consolidada_{ano}.parquet")
                df = pl.read_parquet(io.BytesIO(resp['Body'].read())).to_pandas()
                
                if 'DENSIDADE' in df.columns:
                    df['DENSIDADE'] = df['DENSIDADE'].fillna(0.0)

                df['PERFIL_AREA'] = np.where(df['DENSIDADE'] > 5000, "RESIDENCIAL", 
                                    np.where(df['DENSIDADE'] == 0, "COMERCIAL_INDUSTRIAL", "MISTO"))
                
                for col in ['NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']:
                    df[col] = df[col].astype(str).fillna("DESCONHECIDO").astype('category')
                
                return df
            except Exception as e:
                logger.warning(f"OURO: Ano {ano} não encontrado ou erro de leitura: {e}")
                continue
        return None

    def _gerar_scores_ponderados(self, df, modelos):
        features = ['DENSIDADE', 'NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']
        multiplicadores = self.cal.obter_multiplicadores()
        
        mods = modelos["geral"]
        nome_col = "SCORE_RISCO_GERAL"
        
        p_cat = mods["cat"].predict(df[features]) if mods["cat"] else 0
        p_lgb = mods["lgb"].predict(df[features]) if mods["lgb"] else 0
        
        score = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
        df[nome_col] = score * multiplicadores.get("geral", 1.0)
        
        mask_comercial = df['PERFIL_AREA'] == "COMERCIAL_INDUSTRIAL"
        df.loc[mask_comercial, nome_col] *= multiplicadores.get("comercial", 1.0)
        df.loc[~mask_comercial, nome_col] *= multiplicadores.get("residencial", 1.0)
        
        max_val = df[nome_col].max() if df[nome_col].max() > 0 else 1
        df[nome_col] = (df[nome_col] / max_val * 100).clip(0, 100).round(2)
            
        df['DT_ULTIMA_SINCRONIZACAO'] = datetime.now()
        
        # Agora retornamos o DataFrame "Largo" (Wide) para conseguir montar as dimensões no BigQuery
        return df[['H3_INDEX', 'SCORE_RISCO_GERAL', 'NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA', 'DENSIDADE', 'DT_ULTIMA_SINCRONIZACAO']]

    def _sincronizar_fato_risco(self, df):
        tabela_destino = f"{self.project_id}.{self.dataset_id}.fato_risco_h3_atual"
        tabela_staging = f"{tabela_destino}_staging"
        
        # 1. Carrega tudo (Fato + Dimensões) na tabela de Staging primeiro
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df, tabela_staging, job_config=job_config).result()

        # 2. Garante que a tabela Fato exista antes do MERGE
        try:
            self.bq_client.get_table(tabela_destino)
        except NotFound:
            logger.info("OURO: Tabela Fato não existe. Criando estrutura inicial...")
            sql_init = f"CREATE TABLE `{tabela_destino}` AS SELECT H3_INDEX, SCORE_RISCO_GERAL, DT_ULTIMA_SINCRONIZACAO FROM `{tabela_staging}`"
            self.bq_client.query(sql_init).result()

        # 3. Orquestra a criação do Modelo Semântico (Estrela + View)
        logger.info("OURO: Construindo Modelo Semântico Estrela no BigQuery...")
        self._construir_modelo_semantico(tabela_staging, tabela_destino)

    def _construir_modelo_semantico(self, tabela_staging, tabela_destino):
        # A. Atualiza a Tabela Fato (Merge Clássico)
        query_fato = f"""
        MERGE `{tabela_destino}` T USING `{tabela_staging}` S ON T.H3_INDEX = S.H3_INDEX
        WHEN MATCHED THEN 
            UPDATE SET T.SCORE_RISCO_GERAL = S.SCORE_RISCO_GERAL, T.DT_ULTIMA_SINCRONIZACAO = S.DT_ULTIMA_SINCRONIZACAO
        WHEN NOT MATCHED THEN 
            INSERT (H3_INDEX, SCORE_RISCO_GERAL, DT_ULTIMA_SINCRONIZACAO) VALUES(S.H3_INDEX, S.SCORE_RISCO_GERAL, S.DT_ULTIMA_SINCRONIZACAO)
        """
        self.bq_client.query(query_fato).result()

        # B. Recria a Dimensão Geográfica H3
        query_dim_h3 = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.dim_h3` AS
        SELECT DISTINCT H3_INDEX, NM_BAIRRO, NM_MUN as CIDADE, PERFIL_AREA, DENSIDADE
        FROM `{tabela_staging}`
        """
        self.bq_client.query(query_dim_h3).result()

        # C. Recria a Dimensão Tempo
        query_dim_tempo = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.dim_tempo` AS
        SELECT
            d as DATA,
            EXTRACT(DAYOFWEEK FROM d) as DIA_SEMANA,
            EXTRACT(MONTH FROM d) as MES,
            FORMAT_DATE('%A', d) as NOME_DIA,
            IF(EXTRACT(DAY FROM d) BETWEEN 1 AND 10, TRUE, FALSE) as SEMANA_PAGAMENTO
        FROM UNNEST(GENERATE_DATE_ARRAY('2022-01-01', '2026-12-31')) d
        """
        self.bq_client.query(query_dim_tempo).result()

        # D. Cria a View Analítica (A entrega de Ouro Final)
        query_view = f"""
        CREATE OR REPLACE VIEW `{self.project_id}.{self.dataset_id}.v_safedriver_analitico` AS
        SELECT 
            f.H3_INDEX, 
            f.SCORE_RISCO_GERAL,
            g.NM_BAIRRO, 
            g.CIDADE, 
            g.PERFIL_AREA, 
            g.DENSIDADE,
            t.NOME_DIA, 
            t.SEMANA_PAGAMENTO, 
            f.DT_ULTIMA_SINCRONIZACAO
        FROM `{tabela_destino}` f
        JOIN `{self.project_id}.{self.dataset_id}.dim_h3` g ON f.H3_INDEX = g.H3_INDEX
        LEFT JOIN `{self.project_id}.{self.dataset_id}.dim_tempo` t ON DATE(f.DT_ULTIMA_SINCRONIZACAO) = t.DATA
        """
        self.bq_client.query(query_view).result()
        logger.info("OURO: View Analítica (v_safedriver_analitico) gerada com sucesso e pronta para o Streamlit!")
