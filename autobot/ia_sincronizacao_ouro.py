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
from google.oauth2 import service_account
from autobot.calendario_estrategico import CalendarioEstrategico
import shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class CamadaOuroSafeDriver:
    def __init__(self, dev_mode=True):
     
        self.dev_mode = dev_mode
        
       
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
    
        self.project_id = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9").strip()
        self.dataset_id = os.getenv("BQ_DATASET_ID", "safedriver_gold").strip()

     
        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}, max_pool_connections=50))
        
        self.base_path = self._localizar_datalake_real()
        self.cal = CalendarioEstrategico()
        self.pesos = {"catboost": 0.85, "lightgbm": 0.15}
        
      
        self.features_numericas = ['DENSIDADE', 'TAXA_VACANCIA', 'RANKING_RISCO_LOCAL', 'INDICE_EXPOSICAO']
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
            logger.error(f"OURO: Erro na autenticacao GCP: {e}")

    def executar_predicao_atual(self):
        logger.info(f"OURO: Iniciando materializacao do Data Warehouse (Dev Mode: {self.dev_mode}).")
        try:
            modelos = self._carregar_modelos_producao()
            df_input = self._obter_contexto_datalake()
            
            if df_input is None: 
                logger.error("OURO: Falha ao carregar dados de contexto da Prata.")
                return False

            df_final = self._gerar_scores_ponderados(df_input, modelos)
            self._sincronizar_infraestrutura_bq(df_final)
            return True
        except Exception as e:
            logger.error(f"OURO: Erro critico na Camada Ouro: {e}")
            return False

    def _carregar_modelos_producao(self):
        modelos = {"geral": {}}
        for alg in ["cat", "lgb"]:
            path = self._get_path("modelos_ml", f"latest_{alg}_geral.pkl")
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                modelos["geral"][alg] = joblib.load(io.BytesIO(obj['Body'].read()))
                logger.info(f"OURO: Modelo {alg} carregado com sucesso.")
            except Exception as e:
                logger.warning(f"OURO: Modelo {alg} não encontrado no R2: {e}")
                modelos["geral"][alg] = None
        return modelos

    def _obter_contexto_datalake(self, ano_teste=2026):
        """Carrega e prepara os dados utilizando Lazy API do Polars para máxima performance."""
        lista_lfs = []
        anos_para_processar = [ano_teste] if self.dev_mode else range(2022, datetime.now().year + 1)
        
        for ano in anos_para_processar:
            key = self._get_path("prata", f"ssp_consolidada_{ano}.parquet")
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=key)
                # 🔥 Lê como LazyFrame (Plano de Execução)
                lf = pl.read_parquet(io.BytesIO(resp['Body'].read())).lazy()
                lista_lfs.append(lf)
            except: continue
        
        if not lista_lfs: return None
        
        # Concatena os grafos de execução
        lf_full = pl.concat(lista_lfs, how="diagonal")
        
        # 🔥 Feature Engineering feita nativamente em Rust (Milissegundos)
        lf_full = lf_full.with_columns([
            pl.when(pl.col('DENSIDADE') > 5000).then(pl.lit("RESIDENCIAL"))
              .when(pl.col('DENSIDADE') == 0).then(pl.lit("COMERCIAL_INDUSTRIAL"))
              .otherwise(pl.lit("MISTO")).alias('PERFIL_AREA')
        ])
        
        # Executa todo o grafo em modo Streaming e converte para Pandas
        logger.info("OURO: Avaliando motor Lazy e convertendo para DataFrame ML...")
        df_pandas = lf_full.collect(streaming=True).to_pandas()
        
        # Ajuste de Categorias obrigatório para os Modelos
        for col in self.features_categoricas:
            df_pandas[col] = df_pandas[col].astype(str).fillna("DESCONHECIDO").astype('category')
            
        return df_pandas

    def _gerar_scores_ponderados(self, df, modelos):
        logger.info("OURO: Gerando predições multi-modais e SHAP...")
        mults = self.cal.obter_multiplicadores()
        mods = modelos["geral"]
        
        # Predição com o Ensemble (CatBoost + LightGBM)
        p_cat = mods["cat"].predict(df[self.features_full]) if mods["cat"] else 0
        p_lgb = mods["lgb"].predict(df[self.features_full]) if mods["lgb"] else 0
        
        score_base = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
        df['SCORE_RISCO_GERAL'] = score_base * mults.get("geral", 1.0)
        
        # Ajustes Estratégicos Comerciais
        mask_comercial = df['PERFIL_AREA'] == "COMERCIAL_INDUSTRIAL"
        df.loc[mask_comercial, 'SCORE_RISCO_GERAL'] *= mults.get("comercial", 1.0)
        
        # Normalização 0-100
        df['SCORE_RISCO_GERAL'] = (df['SCORE_RISCO_GERAL'] / (df['SCORE_RISCO_GERAL'].max() or 1) * 100).clip(0, 100).round(2)
        df['DT_ULTIMA_SINCRONIZACAO'] = datetime.now()

        # Explicabilidade SHAP (Otimizada com Trava de CPU)
        if mods["cat"]:
            logger.info("OURO: Calculando SHAP Values (Nível Município)...")
            explainer = shap.TreeExplainer(mods["cat"])
            
            # 🔥 Trava de Performance Sênior: Evita travar a máquina calculando SHAP para 5 milhões de linhas
            if len(df) > 5000:
                logger.warning(f"OURO: Base com {len(df)} linhas. Calculando SHAP apenas para o Top 5000 de maior risco.")
                df_shap = df.nlargest(5000, 'SCORE_RISCO_GERAL')
                shap_matrix = explainer.shap_values(df_shap[self.features_full])
                
                for col in ['SHAP_MUNICIPIO', 'SHAP_PERIODO', 'SHAP_PERFIL_ALVO', 'SHAP_EXPOSICAO']:
                    df[col] = 0.0
                    
                df.loc[df_shap.index, 'SHAP_MUNICIPIO'] = shap_matrix[:, self.features_full.index('NM_MUN')]
                df.loc[df_shap.index, 'SHAP_PERIODO'] = shap_matrix[:, self.features_full.index('PERIODO_DIA')]
                df.loc[df_shap.index, 'SHAP_PERFIL_ALVO'] = shap_matrix[:, self.features_full.index('PERFIL_ALVO')]
                df.loc[df_shap.index, 'SHAP_EXPOSICAO'] = shap_matrix[:, self.features_full.index('INDICE_EXPOSICAO')]
            else:
                shap_values = explainer.shap_values(df[self.features_full])
                df['SHAP_MUNICIPIO'] = shap_values[:, self.features_full.index('NM_MUN')]
                df['SHAP_PERIODO'] = shap_values[:, self.features_full.index('PERIODO_DIA')]
                df['SHAP_PERFIL_ALVO'] = shap_values[:, self.features_full.index('PERFIL_ALVO')]
                df['SHAP_EXPOSICAO'] = shap_values[:, self.features_full.index('INDICE_EXPOSICAO')]
            
            df['SHAP_BASE'] = float(explainer.expected_value)
        else:
            for c in ['SHAP_BASE', 'SHAP_MUNICIPIO', 'SHAP_PERIODO', 'SHAP_PERFIL_ALVO', 'SHAP_EXPOSICAO']:
                df[c] = 0.0
        
        return df

    def _sincronizar_infraestrutura_bq(self, df):
        dataset_path = f"{self.project_id}.{self.dataset_id}"
        tabela_fato = f"{dataset_path}.fato_risco_h3_atual"
        tabela_staging = f"{tabela_fato}_staging"
        
     
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df, tabela_staging, job_config=job_config).result()
        
        logger.info("OURO: Executando MERGE atômico e inserindo novos scores no BigQuery...")
        
       
        self.bq_client.query(f"""
            CREATE TABLE IF NOT EXISTS `{tabela_fato}` AS SELECT * FROM `{tabela_staging}` WHERE 1=0;
            
            MERGE `{tabela_fato}` T USING `{tabela_staging}` S 
            ON T.H3_INDEX = S.H3_INDEX AND T.PERIODO_DIA = S.PERIODO_DIA AND T.PERFIL_ALVO = S.PERFIL_ALVO
            WHEN MATCHED THEN UPDATE SET 
                T.SCORE_RISCO_GERAL = S.SCORE_RISCO_GERAL,
                T.RANKING_RISCO_LOCAL = S.RANKING_RISCO_LOCAL,
                T.SHAP_MUNICIPIO = S.SHAP_MUNICIPIO,
                T.SHAP_PERIODO = S.SHAP_PERIODO,
                T.SHAP_PERFIL_ALVO = S.SHAP_PERFIL_ALVO,
                T.SHAP_EXPOSICAO = S.SHAP_EXPOSICAO,
                T.DT_ULTIMA_SINCRONIZACAO = S.DT_ULTIMA_SINCRONIZACAO
            WHEN NOT MATCHED THEN INSERT ROW LIKE S;
        """).result()

        self._build_semantic_views(dataset_path, tabela_fato)

    def _build_semantic_views(self, dataset, fato):
        self.bq_client.query(f"""
            CREATE OR REPLACE VIEW `{dataset}.v_safedriver_dashboard` AS
            SELECT * FROM `{fato}`
        """).result()
        logger.info("OURO: Camada Semântica atualizada! Base pronta para consumo no Looker Studio.")

if __name__ == "__main__":
    
    ouro = CamadaOuroSafeDriver(dev_mode=True)
    ouro.executar_predicao_atual()
