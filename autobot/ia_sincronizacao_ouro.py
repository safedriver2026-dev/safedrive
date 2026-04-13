import polars as pl
import pandas as pd
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import joblib
import io
import os
import json
import logging
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account
from autobot.calendario_estrategico import CalendarioEstrategico
from autobot.comunicador import ComunicadorSafeDriver

# Configuração de logging padrão corporativo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class CamadaOuroSafeDriver:
    def __init__(self):
        # Credenciais Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # Credenciais BigQuery
        self.project_id = os.getenv("BQ_PROJECT_ID", "").strip()
        self.dataset_id = os.getenv("BQ_DATASET_ID", "").strip()

        # Conexão R2 blindada
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        
        self.cal = CalendarioEstrategico()
        self.comunicador = ComunicadorSafeDriver()
        self.pesos = {"catboost": 0.80, "lightgbm": 0.20}
        
        self._conectar_bigquery()

    def _conectar_bigquery(self):
        """Autenticação via Service Account JSON."""
        gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
        try:
            cred_info = json.loads(gcp_json)
            credentials = service_account.Credentials.from_service_account_info(cred_info)
            self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
        except Exception as e:
            logger.error(f"Erro Crítico na autenticação Google Cloud: {e}")

    def executar_predicao_atual(self):
        """Gera os scores de risco e sincroniza com o Data Warehouse."""
        logger.info("OURO: Iniciando ciclo de Inferência e Sincronização.")
        
        try:
            modelos = self._carregar_modelos_producao()
            df_input = self._obter_contexto_datalake()
            
            if df_input is None:
                raise FileNotFoundError("Datalake Prata não localizado para inferência.")

            # Aplica os modelos de ML e os multiplicadores de feriados/negócio
            df_final = self._gerar_scores_ponderados(df_input, modelos)
            
            # Persistência atómica no BigQuery
            self._sincronizar_fato_risco(df_final)
            return True

        except Exception as e:
            logger.error(f"OURO: Falha crítica na Camada Ouro: {e}")
            return False

    def _carregar_modelos_producao(self):
        """Lê os últimos modelos (latest) salvos no R2 pelo Treinador."""
        modelos = {}
        for persona in ["motorista", "pedestre", "motociclista"]:
            modelos[persona] = {}
            for alg in ["cat", "lgb"]:
                path = f"safedriver/modelos_ml/latest_{alg}_{persona}.pkl"
                try:
                    obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                    modelos[persona][alg] = joblib.load(io.BytesIO(obj['Body'].read()))
                except:
                    modelos[persona][alg] = None
        return modelos

    def _obter_contexto_datalake(self):
        """Busca o ano mais recente na Prata para servir de base para o score."""
        ano_atual = datetime.now().year
        for ano in range(ano_atual, 2021, -1):
            path = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                df = pl.read_parquet(io.BytesIO(resp['Body'].read())).to_pandas().fillna(0)
                
                # Injeta as métricas de performance (meta-features)
                for persona in ["motorista", "pedestre", "motociclista"]:
                    try:
                        m_path = f"safedriver/modelos_ml/meta_perf_{persona}.json"
                        m_obj = self.s3.get_object(Bucket=self.bucket, Key=m_path)
                        meta = json.loads(m_obj['Body'].read())
                        df['ULTIMO_MAE_CAT'] = meta.get('mae_cat', 0.0)
                        df['ULTIMO_MAE_LGB'] = meta.get('mae_lgb', 0.0)
                    except:
                        df['ULTIMO_MAE_CAT'], df['ULTIMO_MAE_LGB'] = 0.0, 0.0
                return df
            except: continue
        return None

    def _gerar_scores_ponderados(self, df, modelos):
        """Calcula o risco final baseado no Ensemble e no Calendário Estratégico."""
        features = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS', 
                    'DELTA_MOTORISTA', 'DELTA_PEDESTRE', 'DELTA_MOTOCICLISTA', 
                    'ULTIMO_MAE_CAT', 'ULTIMO_MAE_LGB']
        
        multiplicadores = self.cal.obter_multiplicadores()
        
        for persona, mods in modelos.items():
            nome_col = f"RISCO_PREDICAO_ATUAL_{persona.upper()}"
            
            p_cat = mods["cat"].predict(df[features]) if mods["cat"] else 0
            p_lgb = mods["lgb"].predict(df[features]) if mods["lgb"] else 0
            
            # Score ponderado (80/20)
            score = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
            df[nome_col] = score
            
            # Heurísticas de Negócio
            is_comercial = df['TOTAL_NAO_RES_H3'] > df['INDICE_RESIDENCIAL']
            df.loc[is_comercial, nome_col] *= multiplicadores["comercial"]
            df.loc[~is_comercial, nome_col] *= multiplicadores["residencial"]
            df[nome_col] *= multiplicadores["geral"]
            
            # Normalização 0-100
            df[nome_col] = (df[nome_col] / (df[nome_col].max() or 1) * 100).clip(0, 100)
            
        df['TIMESTAMP_SINCRONIZACAO'] = datetime.now()
        return df

    def _sincronizar_fato_risco(self, df):
        """Realiza o Upsert no BigQuery com verificação de existência da tabela."""
        tabela_destino = f"{self.project_id}.{self.dataset_id}.fato_risco_predicao_atual"
        tabela_staging = f"{tabela_destino}_staging"
        
        # 1. Carrega dados para a tabela de Staging (Sobrescreve sempre)
        logger.info(f"OURO: Alimentando staging: {tabela_staging}")
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df, tabela_staging, job_config=job_config).result()

        # 2. Gerenciamento de Tabela Principal (Cria se não existir)
        try:
            self.bq_client.get_table(tabela_destino)
            tabela_existe = True
        except NotFound:
            tabela_existe = False

        if not tabela_existe:
            logger.info("OURO: Tabela destino não encontrada. Criando via carga inicial.")
            copy_config = bigquery.CopyJobConfig(write_disposition="WRITE_TRUNCATE")
            self.bq_client.copy_table(tabela_staging, tabela_destino, job_config=copy_config).result()
        else:
            # 3. MERGE (Upsert) para manter apenas uma linha por Hexágono H3
            logger.info("OURO: Iniciando MERGE atómico (Upsert).")
            sql = f"""
            MERGE `{tabela_destino}` T
            USING `{tabela_staging}` S
            ON T.H3_INDEX = S.H3_INDEX
            WHEN MATCHED THEN
              UPDATE SET 
                T.RISCO_PREDICAO_ATUAL_MOTORISTA = S.RISCO_PREDICAO_ATUAL_MOTORISTA,
                T.RISCO_PREDICAO_ATUAL_PEDESTRE = S.RISCO_PREDICAO_ATUAL_PEDESTRE,
                T.RISCO_PREDICAO_ATUAL_MOTOCICLISTA = S.RISCO_PREDICAO_ATUAL_MOTOCICLISTA,
                T.TIMESTAMP_SINCRONIZACAO = S.TIMESTAMP_SINCRONIZACAO
            WHEN NOT MATCHED THEN
              INSERT ROW
            """
            self.bq_client.query(sql).result()
        
        logger.info("✅ OURO: Dados sincronizados com sucesso no BigQuery.")

if __name__ == "__main__":
    CamadaOuroSafeDriver().executar_predicao_atual()
