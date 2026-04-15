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
        
        # Pesos do Ensemble (Ajustados conforme MAE do Treinador)
        self.pesos = {"catboost": 0.80, "lightgbm": 0.20}
        
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
            logger.error(f"OURO: Falha na autenticação BigQuery: {e}")

    def executar_predicao_atual(self):
        """Fluxo principal: Predição -> SHAP -> BigQuery."""
        logger.info(f"OURO: Iniciando Materialização (Dev Mode: {self.dev_mode}).")
        try:
            modelos = self._carregar_modelos_producao()
            df_input = self._obter_contexto_datalake()
            
            if df_input is None or modelos["cat"] is None: 
                logger.error("OURO: Modelos ou dados de entrada ausentes.")
                return False

            # Geração de Inteligência
            df_final = self._gerar_scores_ponderados(df_input, modelos)
            
            # Persistência em Camadas
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
                logger.info(f"OURO: Modelo {alg} carregado da produção.")
            except:
                logger.warning(f"OURO: Modelo {alg} não encontrado no R2.")
        return modelos

    def _obter_contexto_datalake(self):
        """Busca o estado mais atual da Prata para predição."""
        # No modo Dev, focamos no ano corrente para velocidade
        anos = [2026] if self.dev_mode else range(2022, datetime.now().year + 1)
        lista_lfs = []
        
        for ano in anos:
            key = self._get_path("prata", f"ssp_consolidada_{ano}.parquet")
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=key)
                lf = pl.read_parquet(io.BytesIO(resp['Body'].read())).lazy()
                lista_lfs.append(lf)
            except: continue
        
        if not lista_lfs: return None
        
        # Consolidação e Engenharia de Features final
        lf_full = pl.concat(lista_lfs, how="diagonal")
        lf_full = lf_full.with_columns([
            pl.when(pl.col('DENSIDADE') > 5000).then(pl.lit("ALTO_FLUXO"))
              .otherwise(pl.lit("MODERADO")).alias('PERFIL_AREA')
        ])
        
        df = lf_full.collect(engine="streaming").to_pandas()
        for col in self.features_categoricas:
            df[col] = df[col].astype(str).astype('category')
        
        return df

    def _gerar_scores_ponderados(self, df, modelos):
        """Aplica a média ponderada e os multiplicadores do calendário estratégico."""
        logger.info("OURO: Calculando Scores e explicabilidade SHAP...")
        mults = self.cal.obter_multiplicadores()
        
        # 1. Predição Ensemble
        p_cat = modelos["cat"].predict(df[self.features_full])
        p_lgb = modelos["lgb"].predict(df[self.features_full]) if modelos["lgb"] else p_cat
        
        # Fórmula: Score = (CatBoost * W1 + LightGBM * W2) * Mult_Calendario
        score_base = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
        df['SCORE_RISCO_BRUTO'] = score_base * mults.get("geral", 1.0)
        
        # Normalização 0-100 para o Dashboard
        max_val = df['SCORE_RISCO_BRUTO'].max() or 1
        df['SCORE_RISCO_FINAL'] = ((df['SCORE_RISCO_BRUTO'] / max_val) * 100).clip(0, 100).round(2)
        df['DT_PROCESSAMENTO'] = datetime.now()

        # 2. Explicabilidade SHAP (Apenas para o Top Risco para economizar processamento)
        explainer = shap.TreeExplainer(modelos["cat"])
        df_top = df.nlargest(min(len(df), 10000), 'SCORE_RISCO_FINAL')
        
        shap_values = explainer.shap_values(df_top[self.features_full])
        
        # Mapeamento dos principais ofensores
        df['SHAP_MUNICIPIO'] = 0.0
        df['SHAP_PERIODO'] = 0.0
        df['SHAP_ALVO'] = 0.0
        
        df.loc[df_top.index, 'SHAP_MUNICIPIO'] = shap_values[:, self.features_full.index('NM_MUN')]
        df.loc[df_top.index, 'SHAP_PERIODO'] = shap_values[:, self.features_full.index('PERIODO_DIA')]
        df.loc[df_top.index, 'SHAP_ALVO'] = shap_values[:, self.features_full.index('PERFIL_ALVO')]
        
        return df

    def _salvar_parquet_ouro(self, df):
        buffer = io.BytesIO()
        pl.from_pandas(df).write_parquet(buffer, compression="lz4")
        key = self._get_path("ouro", "fato_risco_consolidada.parquet")
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue())
        logger.info(f"OURO: Artefato persistido no R2: {key}")

    def _sincronizar_bq(self, df):
        """Executa carga atômica via Staging + MERGE no BigQuery."""
        tabela_final = f"{self.project_id}.{self.dataset_id}.fato_risco_h3_vigente"
        tabela_staging = f"{tabela_final}_staging"
        
        logger.info(f"OURO: Sincronizando {len(df)} registros com BigQuery...")
        
        # 1. Carga para Staging (Overwrite total)
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df, tabela_staging, job_config=job_config).result()
        
        # 2. MERGE Atômico para manter histórico e evitar duplicatas
        sql_merge = f"""
        MERGE `{tabela_final}` T
        USING `{tabela_staging}` S
        ON T.H3_INDEX = S.H3_INDEX AND T.PERIODO_DIA = S.PERIODO_DIA AND T.PERFIL_ALVO = S.PERFIL_ALVO
        WHEN MATCHED THEN
          UPDATE SET 
            T.SCORE_RISCO_FINAL = S.SCORE_RISCO_FINAL,
            T.SHAP_MUNICIPIO = S.SHAP_MUNICIPIO,
            T.SHAP_PERIODO = S.SHAP_PERIODO,
            T.DT_PROCESSAMENTO = S.DT_PROCESSAMENTO
        WHEN NOT MATCHED THEN
          INSERT ROW AS S
        """
        self.bq_client.query(sql_merge).result()
        
        # 3. Atualização da Camada Semântica (Views)
        self.bq_client.query(f"CREATE OR REPLACE VIEW `{self.project_id}.{self.dataset_id}.v_dashboard_seguranca` AS SELECT * FROM `{tabela_final}`").result()
        logger.info("OURO: BigQuery MERGE e Views concluídos.")

if __name__ == "__main__":
    ouro = CamadaOuroSafeDriver(dev_mode=False)
    ouro.executar_predicao_atual()
