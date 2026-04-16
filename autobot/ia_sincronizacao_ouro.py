import polars as pl
import pandas as pd
import h3, boto3, joblib, io, os, json, logging, shap, gc
from botocore.config import Config
from datetime import datetime
from google.cloud import bigquery
from google.api_core import exceptions
from google.oauth2 import service_account
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
        
        # Pesos Ensemble: CatBoost (6.21 MAE) é o nosso pilar de confiança
        self.pesos = {"catboost": 0.80, "lightgbm": 0.20}
        
        # SCHEMA DE TREINAMENTO (13 features)
        self.features_numericas = ['DENSIDADE', 'TAXA_VACANCIA', 'CONTAGIO_PONDERADO', 'PRESSAO_RISCO_LOCAL', 'MES_OCORRENCIA', 'DIA_SEMANA', 'IS_PAGAMENTO', 'IS_FDS']
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
        """Autentica e garante que o Dataset Star Schema exista no BigQuery."""
        gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
        try:
            cred_info = json.loads(gcp_json)
            credentials = service_account.Credentials.from_service_account_info(cred_info)
            self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
            
            # Garantir Dataset
            dataset_ref = bigquery.DatasetReference(self.project_id, self.dataset_id)
            try:
                self.bq_client.get_dataset(dataset_ref)
            except exceptions.NotFound:
                dataset = bigquery.Dataset(dataset_ref)
                dataset.location = "US" # Ou sua região preferida
                dataset.description = "Camada Ouro SafeDriver - Star Schema"
                self.bq_client.create_dataset(dataset)
                logger.info(f"OURO: Dataset {self.dataset_id} criado com sucesso.")
        except Exception as e:
            logger.error(f"OURO: Erro de Infra BQ: {e}")

    def executar_predicao_atual(self):
        logger.info(f"OURO: Iniciando ciclo de inteligência preditiva (Star Schema Mode).")
        try:
            modelos = self._carregar_modelos_producao()
            df_input = self._obter_contexto_preditivo_total()
            
            if df_input is None: return False

            # --- PREPARAÇÃO PARA IA (Blindagem de Dtypes) ---
            X = df_input[self.features_full].copy()
            for col in self.features_categoricas: X[col] = X[col].astype('category')
            for col in self.features_numericas: X[col] = X[col].astype('float32')

            logger.info("OURO: Gerando Scores IA...")
            p_cat = modelos["cat"].predict(X) if modelos["cat"] else None
            
            # Fallback seguro para o LightGBM (evita o erro de categorical_feature)
            try:
                p_lgb = modelos["lgb"].predict(X) if modelos["lgb"] else p_cat
            except:
                logger.warning("OURO: LightGBM recusou dtypes. Usando apenas CatBoost.")
                p_lgb = p_cat

            score_base = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
            mults = self.cal.obter_multiplicadores()
            
            # --- SCHEMA STAR: FATO DE RISCO ---
            df_input['SCORE_RISCO_BRUTO'] = score_base * mults.get("geral", 1.0)
            max_val = df_input['SCORE_RISCO_BRUTO'].max() or 1
            df_input['SCORE_RISCO_FINAL'] = ((df_input['SCORE_RISCO_BRUTO'] / max_val) * 100).clip(0, 100).round(2)
            df_input['DT_REF'] = datetime.now() # Coluna de partição
            
            # SHAP (Explicabilidade da Fato)
            try:
                explainer = shap.TreeExplainer(modelos["cat"])
                df_top = df_input.nlargest(min(len(df_input), 1500), 'SCORE_RISCO_FINAL')
                shap_values = explainer.shap_values(df_top[self.features_full])
                for i, feat in enumerate(['NM_MUN', 'PERIODO_DIA', 'PERFIL_ALVO']):
                    col_name = f'SHAP_{feat}'
                    df_input[col_name] = 0.0
                    feat_idx = self.features_full.index(feat)
                    df_input.loc[df_top.index, col_name] = shap_values[:, feat_idx]
            except: pass

            self._sincronizar_star_schema(df_input)
            return True
        except Exception as e:
            logger.error(f"OURO: Erro Crítico: {e}")
            return False

    def _sincronizar_star_schema(self, df):
        """Cria e alimenta a Tabela de Fatos no BigQuery com clustering geográfico."""
        table_id = f"{self.project_id}.{self.dataset_id}.fato_risco_consolidada"
        
        # Limpeza de Dtypes para o BigQuery
        df_bq = df.copy()
        for col in self.features_categoricas: df_bq[col] = df_bq[col].astype(str)
        df_bq['DT_REF'] = pd.to_datetime(df_bq['DT_REF'])

        # Configuração de Tabela de Alta Performance
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
            time_partitioning=bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field="DT_REF"),
            clustering_fields=["H3_INDEX", "NM_MUN"],
            schema=[
                bigquery.SchemaField("H3_INDEX", "STRING"),
                bigquery.SchemaField("SCORE_RISCO_FINAL", "FLOAT"),
                bigquery.SchemaField("DT_REF", "TIMESTAMP"),
            ]
        )

        job = self.bq_client.load_table_from_dataframe(df_bq, table_id, job_config=job_config)
        job.result()
        logger.info(f"OURO: Tabela de Fatos sincronizada: {table_id} (Clustered & Partitioned)")

    def _carregar_modelos_producao(self):
        modelos = {"cat": None, "lgb": None}
        for alg in ["cat", "lgb"]:
            path = f"{self.base_path}/modelos_ml/latest_{alg}_geral.pkl"
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                modelos[alg] = joblib.load(io.BytesIO(obj['Body'].read()))
            except: pass
        return modelos

    def _obter_contexto_preditivo_total(self):
        try:
            # Carrega Malha
            resp_m = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            df_malha = pl.read_parquet(io.BytesIO(resp_m['Body'].read()))
            
            # Carrega Prata (Ano Atual)
            ano = datetime.now().year
            path_prata = f"{self.base_path}/prata/ssp_consolidada_{ano}.parquet"
            df_prata = pl.read_parquet(io.BytesIO(self.s3.get_object(Bucket=self.bucket, Key=path_prata)['Body'].read()))
            
            # Agregação para Features Geográficas
            df_prata_agg = df_prata.group_by("H3_INDEX").agg([
                pl.col("TOTAL_CRIMES").sum().alias("TOTAL_CRIMES"),
                pl.col("RANKING_RISCO_LOCAL").mean().alias("RANKING_RISCO_LOCAL"),
                pl.col("INDICE_EXPOSICAO").mean().alias("INDICE_EXPOSICAO")
            ])

            df_pd = df_malha.join(df_prata_agg, on="H3_INDEX", how="left").to_pandas()
            
            # Contágio Espacial (K-Ring 1 e 2)
            crimes_dit = dict(zip(df_pd['H3_INDEX'], df_pd['TOTAL_CRIMES'].fillna(0)))
            contagio = []
            for h3_index in df_pd['H3_INDEX']:
                try:
                    v1 = set(h3.k_ring(h3_index, 1)); v1.discard(h3_index)
                    c1 = sum(crimes_dit.get(v, 0) for v in v1)
                    v2 = set(h3.k_ring(h3_index, 2)) - v1; v2.discard(h3_index)
                    contagio.append((c1 * 1.0) + (sum(crimes_dit.get(v, 0) for v in v2) * 0.5))
                except: contagio.append(0.0)
            
            df_pd['CONTAGIO_PONDERADO'] = contagio
            df_pd['PRESSAO_RISCO_LOCAL'] = df_pd['CONTAGIO_PONDERADO'] / (df_pd['DENSIDADE_AJUSTADA'] + 0.001)
            
            # Contexto Temporal
            hoje = datetime.now()
            df_pd['NM_MUN'] = df_pd['NM_MUN'].fillna("SAO PAULO")
            df_pd['NM_BAIRRO'] = df_pd['NM_BAIRRO'].fillna("INDEFINIDO")
            df_pd['PERIODO_DIA'] = "TARDE"
            df_pd['PERFIL_ALVO'] = "PEDESTRE"
            df_pd['TIPO_LOCAL'] = "VIA PUBLICA"
            df_pd['MES_OCORRENCIA'] = hoje.month
            df_pd['DIA_SEMANA'] = hoje.weekday()
            df_pd['IS_PAGAMENTO'] = 1 if 5 <= hoje.day <= 10 else 0
            df_pd['IS_FDS'] = 1 if hoje.weekday() >= 6 else 0
            df_pd['DENSIDADE'] = df_pd['DENSIDADE_AJUSTADA'].fillna(0)
            df_pd['TAXA_VACANCIA'] = df_pd['TAXA_VACANCIA'].fillna(0)
            df_pd['RANKING_RISCO_LOCAL'] = df_pd['RANKING_RISCO_LOCAL'].fillna(0.5)
            df_pd['INDICE_EXPOSICAO'] = df_pd['INDICE_EXPOSICAO'].fillna(0.1)
            
            return df_pd
        except Exception as e:
            logger.error(f"OURO: Erro Contexto: {e}"); return None

if __name__ == "__main__":
    ouro = CamadaOuroSafeDriver(dev_mode=False)
    ouro.executar_predicao_atual()
