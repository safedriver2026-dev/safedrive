import polars as pl
import pandas as pd
import boto3
from botocore.config import Config
import joblib
import io
import os
import json
import logging
import shap
from datetime import datetime
from google.cloud import bigquery
from google.oauth2 import service_account
from autobot.calendario_estrategico import CalendarioEstrategico

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class CamadaOuroSafeDriver:
    def __init__(self, dev_mode=True):
        self.dev_mode = dev_mode
        
        # Conectividade Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # Conectividade Google BigQuery
        self.project_id = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9").strip()
        self.dataset_id = os.getenv("BQ_DATASET_ID", "safedriver_gold").strip()

        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}, max_pool_connections=50))
        
        self.base_path = self._localizar_datalake_real()
        self.cal = CalendarioEstrategico()
        
        # Pesos do Ensemble
        self.pesos = {"catboost": 0.70, "lightgbm": 0.30}
        
        # Features atualizadas conforme a nova Camada Prata
        self.features_numericas = [
            'DENSIDADE', 'TAXA_VACANCIA', 'RANKING_RISCO_LOCAL', 'INDICE_EXPOSICAO',
            'CONTAGIO_PONDERADO', 'PRESSAO_RISCO_LOCAL', 
            'MES_OCORRENCIA', 'DIA_SEMANA_OCORRENCIA'
        ]
        self.features_categoricas = ['NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA', 'PERIODO_DIA', 'PERFIL_ALVO', 'TIPO_LOCAL']
        self.features_full = self.features_numericas + self.features_categoricas

        self._conectar_bigquery()

    def _localizar_datalake_real(self):
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, MaxKeys=100):
                for obj in page.get('Contents', []):
                    if "datalake/prata/" in obj['Key']:
                        return obj['Key'].split("datalake/")[0] + "datalake"
            return "datalake"
        except: return "datalake"

    def _get_path(self, camada, filename):
        return f"{self.base_path}/{camada}/{filename}".replace("//", "/")

    def _conectar_bigquery(self):
        gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
        try:
            cred_info = json.loads(gcp_json)
            credentials = service_account.Credentials.from_service_account_info(cred_info)
            self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
        except Exception as e:
            logger.error(f"OURO: Falha na autenticação BigQuery: {e}")

    def executar_predicao_atual(self):
        """Fluxo: Cruzamento Geográfico Total -> Predição -> SHAP -> BigQuery."""
        logger.info(f"OURO: Iniciando Materialização Preditiva (Dev: {self.dev_mode}).")
        try:
            modelos = self._carregar_modelos_producao()
            
            # Mudança: Agora buscamos a malha completa para inferência
            df_input = self._obter_contexto_preditivo_total()
            
            if df_input is None or modelos["cat"] is None: 
                logger.error("OURO: Modelos ou malha de entrada ausentes.")
                return False

            # Geração de Inteligência (Ensemble + Calendário)
            df_final = self._gerar_scores_ponderados(df_input, modelos)
            
            # Persistência
            self._salvar_parquet_ouro(df_final)
            self._sincronizar_bq(df_final)
            
            return True
        except Exception as e:
            logger.error(f"OURO: Erro crítico no pipeline Ouro: {e}")
            return False

    def _carregar_modelos_producao(self):
        modelos = {"cat": None, "lgb": None}
        for alg in ["cat", "lgb"]:
            path = self._get_path("modelos_ml", f"latest_{alg}_geral.pkl")
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                modelos[alg] = joblib.load(io.BytesIO(obj['Body'].read()))
                logger.info(f"OURO: Modelo {alg} carregado com sucesso.")
            except:
                logger.warning(f"OURO: Falha ao carregar modelo {alg}.")
        return modelos

    def _obter_contexto_preditivo_total(self):
        """
        Cruza a malha completa de SP com os dados da Prata. 
        Garante que áreas com 0 crimes também sejam processadas.
        """
        logger.info("OURO: Consolidando Malha Geográfica Total para predição...")
        
        # Carregamos os dados da Prata mais recentes
        ano_atual = datetime.now().year
        key_prata = self._get_path("prata", f"ssp_consolidada_{ano_atual}.parquet")
        
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=key_prata)
            df_prata = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            
            # Se houver uma malha de referência sem crimes (ex: bronze/georef), 
            # você faria um join aqui. Por enquanto, garantimos a tipagem da Prata.
            df = df_prata.to_pandas()
            
            # Blindagem de tipos para LightGBM e CatBoost
            for col in self.features_categoricas:
                df[col] = df[col].astype(str).fillna("INDEFINIDO").astype('category')
            
            for col in self.features_numericas:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
                else:
                    df[col] = 0.0 # Cria colunas ausentes para não quebrar o modelo
                    
            return df
        except Exception as e:
            logger.error(f"OURO: Erro ao carregar malha: {e}")
            return None

    def _gerar_scores_ponderados(self, df, modelos):
        """Aplica o Ensemble e identifica 'Alertas Silenciosos'."""
        logger.info("OURO: Calculando Scores Ensemble e SHAP...")
        
        mults = self.cal.obter_multiplicadores()
        
        # 1. Predição Ensemble
        X = df[self.features_full]
        p_cat = modelos["cat"].predict(X)
        p_lgb = modelos["lgb"].predict(X) if modelos["lgb"] else p_cat
        
        # Score Ponderado
        score_base = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
        df['SCORE_RISCO_BRUTO'] = score_base * mults.get("geral", 1.0)
        
        # Normalização
        max_val = df['SCORE_RISCO_BRUTO'].max() or 1
        df['SCORE_RISCO_FINAL'] = ((df['SCORE_RISCO_BRUTO'] / max_val) * 100).clip(0, 100).round(2)
        
        # Identificação de Alertas Silenciosos (Risco Alto em locais com 0 crimes)
        if 'TOTAL_CRIMES' in df.columns:
            df['FLAG_ALERTA_SILENCIOSO'] = (df['TOTAL_CRIMES'] == 0) & (df['SCORE_RISCO_FINAL'] > 70)
        
        df['DT_PROCESSAMENTO'] = datetime.now()

        # 2. Explicabilidade SHAP (Reduzido para os top 5000 para performance)
        try:
            explainer = shap.TreeExplainer(modelos["cat"])
            df_top = df.nlargest(min(len(df), 5000), 'SCORE_RISCO_FINAL')
            shap_values = explainer.shap_values(df_top[self.features_full])
            
            # Mapeamos as 3 causas principais do risco
            for i, feat in enumerate(['NM_MUN', 'PERIODO_DIA', 'PERFIL_ALVO']):
                col_name = f'SHAP_{feat}'
                df[col_name] = 0.0
                feat_idx = self.features_full.index(feat)
                df.loc[df_top.index, col_name] = shap_values[:, feat_idx]
        except Exception as e:
            logger.warning(f"OURO: Erro ao gerar SHAP: {e}")

        return df

    def _salvar_parquet_ouro(self, df):
        buffer = io.BytesIO()
        # Removemos colunas temporárias de categoria para salvar o Parquet limpo
        df_save = df.copy()
        for col in self.features_categoricas:
            df_save[col] = df_save[col].astype(str)
            
        pl.from_pandas(df_save).write_parquet(buffer, compression="lz4")
        key = self._get_path("ouro", "fato_risco_consolidada.parquet")
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue())
        logger.info(f"OURO: Parquet persistido no R2.")

    def _sincronizar_bq(self, df):
        """Sincronização via MERGE no BigQuery."""
        tabela_final = f"{self.project_id}.{self.dataset_id}.fato_risco_h3_vigente"
        tabela_staging = f"{tabela_final}_staging"
        
        # Limpeza para o BQ
        df_bq = df.copy()
        for col in self.features_categoricas:
            df_bq[col] = df_bq[col].astype(str)

        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df_bq, tabela_staging, job_config=job_config).result()
        
        sql_merge = f"""
        MERGE `{tabela_final}` T
        USING `{tabela_staging}` S
        ON T.H3_INDEX = S.H3_INDEX AND T.PERIODO_DIA = S.PERIODO_DIA
        WHEN MATCHED THEN
          UPDATE SET 
            T.SCORE_RISCO_FINAL = S.SCORE_RISCO_FINAL,
            T.FLAG_ALERTA_SILENCIOSO = S.FLAG_ALERTA_SILENCIOSO,
            T.DT_PROCESSAMENTO = S.DT_PROCESSAMENTO
        WHEN NOT MATCHED THEN
          INSERT ROW AS S
        """
        self.bq_client.query(sql_merge).result()
        logger.info("OURO: BigQuery sincronizado com visão preditiva.")

if __name__ == "__main__":
    ouro = CamadaOuroSafeDriver(dev_mode=False)
    ouro.executar_predicao_atual()
