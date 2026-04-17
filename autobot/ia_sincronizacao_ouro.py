import polars as pl
import pandas as pd
import h3, boto3, joblib, io, os, json, logging, shap, gc
from botocore.config import Config
from datetime import datetime
from google.cloud import bigquery
from google.api_core import exceptions
from google.oauth2 import service_account

# Importa o Calendário Estratégico refatorado
from autobot.calendario_estrategico import CalendarioEstrategico

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class CamadaOuroSafeDriver:
    def __init__(self, dev_mode=False):
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
        self.malha_path = f"{self.base_path}/base_geografica/safedriver_geo_base_sp_h3_9.parquet"
        
        self.cal = CalendarioEstrategico()
        self.pesos = {"catboost": 0.85, "lightgbm": 0.15}
        
        # --- ALINHAMENTO COM O TREINADOR V3 ---
        self.features_numericas = [
            'DENSIDADE', 
            'TAXA_VACANCIA', 'CONTAGIO_PONDERADO', 'PRESSAO_RISCO_LOCAL', 
            'GRAVIDADE_HISTORICA', 'VOLUME_HISTORICO', 
            'MES_OCORRENCIA', 'DIA_SEMANA', 'IS_PAGAMENTO', 'IS_FDS'
        ]
        self.features_categoricas = ['NM_BAIRRO', 'NM_MUN', 'PERIODO_DIA', 'PERFIL_ALVO', 'TIPO_LOCAL']
        self.features_full = self.features_numericas + self.features_categoricas

        self._conectar_e_preparar_bq()

    def _localizar_datalake_real(self):
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, MaxKeys=100):
                for obj in page.get('Contents', []):
                    if "datalake/prata/" in obj['Key']: return obj['Key'].split("datalake/")[0] + "datalake"
            return "datalake"
        except: return "datalake"

    def _conectar_e_preparar_bq(self):
        gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
        try:
            cred_info = json.loads(gcp_json)
            credentials = service_account.Credentials.from_service_account_info(cred_info)
            self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
            
            dataset_ref = bigquery.DatasetReference(self.project_id, self.dataset_id)
            try:
                self.bq_client.get_dataset(dataset_ref)
            except exceptions.NotFound:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US"
                dataset.description = "Camada Ouro SafeDriver - IA Preditiva Granular"
                self.bq_client.create_dataset(dataset)
        except Exception as e:
            logger.error(f"OURO: Erro de Infra BQ: {e}")

    def executar_predicao_atual(self):
        logger.info(f"OURO: Iniciando inferência dinâmica com Memória Histórica Defasada.")
        try:
            modelos = self._carregar_modelos_producao()
            df_input = self._obter_contexto_preditivo_total()
            
            if df_input is None or df_input.empty: 
                return False

            # --- PREPARAÇÃO DE ENTRADA PARA OS MODELOS ---
            X = df_input[self.features_full].copy()
            for col in self.features_categoricas: X[col] = X[col].astype('category')
            for col in self.features_numericas: X[col] = X[col].astype('float32')

            logger.info("OURO: Gerando Scores IA para Malha Total (Scaffold Mode)...")
            p_cat = modelos["cat"].predict(X) if modelos["cat"] else None
            p_lgb = modelos["lgb"].predict(X) if modelos["lgb"] else p_cat

            # Cálculo de Risco Baseado em Ensemble Tweedie
            score_raw = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
            df_input['SCORE_RISCO_BRUTO'] = score_raw

            # --- NORMALIZAÇÃO DE CONTRASTE INTRA-PERFIL ---
            logger.info("OURO: Aplicando Normalização de Contraste por Perfil...")
            df_input['SCORE_RISCO_FINAL'] = df_input.groupby('PERFIL_ALVO')['SCORE_RISCO_BRUTO'].transform(
                lambda x: ((x - x.min()) / (x.max() - x.min() + 0.001) * 100)
            ).clip(0, 100).round(2)

            df_input['DT_REF'] = datetime.now() 
            
            # SHAP para Explicabilidade (Auditoria de Decisão da IA)
            try:
                explainer = shap.TreeExplainer(modelos["cat"])
                df_top = df_input.nlargest(1000, 'SCORE_RISCO_FINAL')
                shap_values = explainer.shap_values(df_top[self.features_full])
                for i, feat in enumerate(['NM_MUN', 'PERIODO_DIA', 'PERFIL_ALVO']):
                    col_name = f'SHAP_{feat}'
                    df_input[col_name] = 0.0
                    feat_idx = self.features_full.index(feat)
                    df_input.loc[df_top.index, col_name] = shap_values[:, feat_idx]
            except Exception as e: 
                logger.warning(f"OURO: Aviso ao gerar valores SHAP: {e}")

            self._sincronizar_star_schema(df_input)
            return True
        except Exception as e:
            logger.error(f"OURO Erro Crítico: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _sincronizar_star_schema(self, df):
        """Alimenta a Tabela de Fatos com Clustering para Alta Performance no Dashboard."""
        table_id = f"{self.project_id}.{self.dataset_id}.fato_risco_consolidada"
        df_bq = df.copy()
        for col in self.features_categoricas: df_bq[col] = df_bq[col].astype(str)
        df_bq['DT_REF'] = pd.to_datetime(df_bq['DT_REF'])

        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
            time_partitioning=bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field="DT_REF"),
            clustering_fields=["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO"] 
        )
        self.bq_client.load_table_from_dataframe(df_bq, table_id, job_config=job_config).result()

    def _carregar_modelos_producao(self):
        modelos = {"cat": None, "lgb": None}
        for alg in ["cat", "lgb"]:
            path = f"{self.base_path}/modelos_ml/latest_{alg}_geral.pkl"
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                modelos[alg] = joblib.load(io.BytesIO(obj['Body'].read()))
            except: logger.warning(f"OURO: Modelo {alg} não encontrado no Datalake.")
        return modelos

    def _obter_contexto_preditivo_total(self):
        """Gera o Scaffold de predição integrando a Memória da Camada Prata."""
        try:
            # 1. Malha Geográfica Base
            resp_m = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            df_malha = pl.read_parquet(io.BytesIO(resp_m['Body'].read())).unique(subset=["H3_INDEX"])
            
            # 2. Carrega Memória da Prata (Histórico Real)
            ano = datetime.now().year
            path_prata = f"{self.base_path}/prata/ssp_consolidada_{ano}.parquet"
            df_prata = pl.read_parquet(io.BytesIO(self.s3.get_object(Bucket=self.bucket, Key=path_prata)['Body'].read()))
            
            # 🚨 CORREÇÃO DO VAZAMENTO: 
            # Pegamos o ÚLTIMO dado conhecido do local e o transformamos no passado (HISTORICO) de hoje.
            df_memoria = df_prata.sort("MES_OCORRENCIA").group_by(["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO"]).last().select([
                "H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO",
                pl.col("INDICE_GRAVIDADE").alias("GRAVIDADE_HISTORICA"),
                pl.col("TOTAL_CRIMES").alias("VOLUME_HISTORICO"),
                "CONTAGIO_PONDERADO"
            ])

            # 3. SCAFFOLD: Criação da Malha Total de Predição (24h)
            periodos = pl.DataFrame({"PERIODO_DIA": ["MANHA", "TARDE", "NOITE", "MADRUGADA"]})
            perfis = pl.DataFrame({"PERFIL_ALVO": ["PEDESTRE", "MOTORISTA"]})
            scaffold = periodos.join(perfis, how="cross")
            
            df_base = df_malha.select(["H3_INDEX", "DENSIDADE_AJUSTADA", "TAXA_VACANCIA", "NM_MUN", "NM_BAIRRO"]).join(
                scaffold, how="cross"
            )

            # 4. Join: Injecão da Memória Recente no Scaffold Atual
            df_full = df_base.join(
                df_memoria, on=["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO"], how="left"
            ).with_columns([
                pl.col("GRAVIDADE_HISTORICA").fill_null(0.0),
                pl.col("VOLUME_HISTORICO").fill_null(0.0),
                pl.col("CONTAGIO_PONDERADO").fill_null(0.0),
                pl.col("DENSIDADE_AJUSTADA").alias("DENSIDADE")
            ])

            df_pd = df_full.to_pandas()

            # 5. Cálculo Dinâmico de Pressão de Risco
            df_pd['PRESSAO_RISCO_LOCAL'] = df_pd['CONTAGIO_PONDERADO'] / (df_pd['DENSIDADE'] + 0.001)

            # 6. Contexto Temporal do Calendário Inteligente
            contexto_cal = self.cal.obter_contexto_ia()
            df_pd['MES_OCORRENCIA'] = self.cal.hoje.month
            df_pd['DIA_SEMANA'] = self.cal.hoje.weekday()
            df_pd['IS_PAGAMENTO'] = contexto_cal['IS_PAGAMENTO']
            df_pd['IS_FDS'] = contexto_cal['IS_FDS']
            df_pd['TIPO_LOCAL'] = "VIA PUBLICA" 
            
            return df_pd
        except Exception as e:
            logger.error(f"OURO Erro Contexto: {e}")
            return None

if __name__ == "__main__":
    ouro = CamadaOuroSafeDriver()
    ouro.executar_predicao_atual()
