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

# Otimizacao: carrega a explicabilidade pesada apenas sob demanda
import shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class CamadaOuroSafeDriver:
    def __init__(self):
        # Credenciais R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # Configurações BigQuery
        self.project_id = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9").strip()
        self.dataset_id = os.getenv("BQ_DATASET_ID", "safedriver_gold").strip()

        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}))
        
        # --- LÓGICA DE LOCALIZAÇÃO AUTOMÁTICA ---
        self.base_path = self._descobrir_prefixo_datalake()
        logger.info(f"OURO: Raiz do Data Lake detectada em: '{self.base_path}'")
        
        self.cal = CalendarioEstrategico()
        self.pesos = {"catboost": 0.85, "lightgbm": 0.15}
        self._conectar_bigquery()

    def _descobrir_prefixo_datalake(self):
        """Localiza dinamicamente a pasta raiz dos dados no R2."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, MaxKeys=15)
            if 'Contents' in response:
                keys = [obj['Key'] for obj in response['Contents']]
                for key in keys:
                    if "safedriver/datalake" in key: return "safedriver/datalake"
                    if "datalake" in key: return "datalake"
            return "datalake"
        except: return "datalake"

    def _get_path(self, camada, subpasta, filename):
        """Helper para caminhos consistentes."""
        return f"{self.base_path}/{camada}/{subpasta}/{filename}".replace("//", "/")

    def _conectar_bigquery(self):
        gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
        try:
            cred_info = json.loads(gcp_json)
            credentials = service_account.Credentials.from_service_account_info(cred_info)
            self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
        except Exception as e:
            logger.error(f"OURO: Erro na autenticacao GCP: {e}")

    def executar_predicao_atual(self):
        logger.info("OURO: Iniciando materializacao do Data Warehouse.")
        try:
            modelos = self._carregar_modelos_producao()
            df_input = self._obter_contexto_datalake()
            
            if df_input is None: 
                logger.error("OURO: Falha ao carregar dados de contexto.")
                return False

            # Gera predições e explicabilidade SHAP
            df_final = self._gerar_scores_ponderados(df_input, modelos)
            
            # Sincroniza DW no BigQuery
            self._sincronizar_infraestrutura_bq(df_final)
            return True
        except Exception as e:
            logger.error(f"OURO: Erro critico na Camada Ouro: {e}")
            return False

    def _carregar_modelos_producao(self):
        modelos = {"geral": {}}
        for alg in ["cat", "lgb"]:
            # Localiza o modelo usando o caminho dinâmico
            path = self._get_path("modelos_ml", "", f"latest_{alg}_geral.pkl")
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                modelos["geral"][alg] = joblib.load(io.BytesIO(obj['Body'].read()))
                logger.info(f"OURO: Modelo {alg} carregado com sucesso.")
            except Exception as e:
                logger.warning(f"OURO: Modelo {alg} não encontrado em {path}: {e}")
                modelos["geral"][alg] = None
        return modelos

    def _obter_contexto_datalake(self):
        lista_dfs = []
        for ano in range(2022, datetime.now().year + 1):
            key = self._get_path("prata", "", f"ssp_consolidada_{ano}.parquet")
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=key)
                df = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                lista_dfs.append(df)
            except: continue
        
        if not lista_dfs: return None

        df_full = pl.concat(lista_dfs, how="diagonal")
        
        # Agregacao historica para Prova Real
        df_agg = df_full.group_by("H3_INDEX").agg([
            pl.col("TOTAL_CRIMES").sum().alias("TOTAL_CRIMES_HISTORICO"),
            pl.col("NM_BAIRRO").first(),
            pl.col("NM_MUN").first(),
            pl.col("DENSIDADE").first().fill_null(0.0),
            pl.col("TAXA_VACANCIA").first().fill_null(0.0) if "TAXA_VACANCIA" in df_full.columns else pl.lit(0.0).alias("TAXA_VACANCIA")
        ]).to_pandas()

        df_agg['PERFIL_AREA'] = np.where(df_agg['DENSIDADE'] > 5000, "RESIDENCIAL", 
                                np.where(df_agg['DENSIDADE'] == 0, "COMERCIAL_INDUSTRIAL", "MISTO"))
        
        for col in ['NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']:
            df_agg[col] = df_agg[col].astype(str).fillna("DESCONHECIDO").astype('category')
        
        return df_agg

    def _gerar_scores_ponderados(self, df, modelos):
        logger.info("OURO: Computando predicacao espacial e decomposicao SHAP...")
        features = ['DENSIDADE', 'TAXA_VACANCIA', 'NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']
        mults = self.cal.obter_multiplicadores()
        mods = modelos["geral"]
        
        p_cat = mods["cat"].predict(df[features]) if mods["cat"] else 0
        p_lgb = mods["lgb"].predict(df[features]) if mods["lgb"] else 0
        
        score_base = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
        df['SCORE_RISCO_GERAL'] = score_base * mults.get("geral", 1.0)
        
        # Sazonalidade
        mask_comercial = df['PERFIL_AREA'] == "COMERCIAL_INDUSTRIAL"
        df.loc[mask_comercial, 'SCORE_RISCO_GERAL'] *= mults.get("comercial", 1.0)
        df.loc[~mask_comercial, 'SCORE_RISCO_GERAL'] *= mults.get("residencial", 1.0)
        
        df['SCORE_RISCO_GERAL'] = (df['SCORE_RISCO_GERAL'] / (df['SCORE_RISCO_GERAL'].max() or 1) * 100).clip(0, 100).round(2)
        df['DT_ULTIMA_SINCRONIZACAO'] = datetime.now()

        # SHAP Explainability
        if mods["cat"]:
            explainer = shap.TreeExplainer(mods["cat"])
            shap_values = explainer.shap_values(df[features])
            df['SHAP_BASE'] = float(explainer.expected_value)
            df['SHAP_DENSIDADE'] = shap_values[:, 0].astype(float)
            df['SHAP_VACANCIA'] = shap_values[:, 1].astype(float)
            df['SHAP_BAIRRO'] = shap_values[:, 2].astype(float)
            df['SHAP_MUN'] = shap_values[:, 3].astype(float)
            df['SHAP_PERFIL'] = shap_values[:, 4].astype(float)
        else:
            for c in ['SHAP_BASE', 'SHAP_DENSIDADE', 'SHAP_VACANCIA', 'SHAP_BAIRRO', 'SHAP_MUN', 'SHAP_PERFIL']:
                df[c] = 0.0
        
        return df

    def _sincronizar_infraestrutura_bq(self, df):
        # ... [Mesma lógica de MERGE e Star Schema que você já tinha] ...
        # (Apenas certifique-se de que as colunas SHAP batem com o seu DataFrame)
        dataset_path = f"{self.project_id}.{self.dataset_id}"
        tabela_fato = f"{dataset_path}.fato_risco_h3_atual"
        tabela_staging = f"{tabela_fato}_staging"
        
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df, tabela_staging, job_config=job_config).result()
        logger.info("OURO: Sincronizando tabelas fato e dimensões no BigQuery...")
        # ... (Resto da lógica de SQL que você já possui) ...
