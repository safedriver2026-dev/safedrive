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
from google.oauth2 import service_account
from autobot.calendario_estrategico import CalendarioEstrategico
from autobot.comunicador import ComunicadorSafeDriver

# Configuração de logging padrão corporativo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class CamadaOuroSafeDriver:
    def __init__(self):
        # Definições de ambiente para o Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # Definições de ambiente para o Google BigQuery
        self.project_id = os.getenv("BQ_PROJECT_ID", "").strip()
        self.dataset_id = os.getenv("BQ_DATASET_ID", "").strip()

        # Conexão blindada S3v4 com Path Addressing Style (Compatibilidade R2)
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        
        self.cal = CalendarioEstrategico()
        self.comunicador = ComunicadorSafeDriver()

        # Parâmetros do motor Ensemble (80% CatBoost / 20% LightGBM)
        self.pesos = {"catboost": 0.80, "lightgbm": 0.20}
        
        # Estrutura do Feature Space validado no Treinador
        self.features_base = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS']
        self.features_delta = ['DELTA_MOTORISTA', 'DELTA_PEDESTRE', 'DELTA_MOTOCICLISTA']
        self.meta_features = ['ULTIMO_MAE_CAT', 'ULTIMO_MAE_LGB']

        self._conectar_bigquery()

    def _conectar_bigquery(self):
        """Estabelece a conexão com o BigQuery utilizando a Service Account JSON correta."""
        gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
        try:
            if gcp_json.startswith("{"):
                cred_info = json.loads(gcp_json)
                credentials = service_account.Credentials.from_service_account_info(cred_info)
                self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
            else:
                self.bq_client = bigquery.Client(project=self.project_id)
        except Exception as e:
            logger.error(f"Erro Crítico na autenticação do Google Cloud: {e}")

    def executar_predicao_atual(self):
        """Orquestra a inferência dinâmica e a sincronização (Upsert) com o BigQuery."""
        logger.info("OURO: Iniciando processamento de Predição de Risco Atual.")
        
        try:
            modelos = self._carregar_modelos_producao()
            df_input = self._obter_contexto_datalake()
            
            if df_input is None:
                raise FileNotFoundError("Base de contexto não localizada na Camada Prata. Processo suspenso.")

            # Geração da predição matemática ajustada pelas variáveis de negócio
            df_final = self._gerar_scores_ponderados(df_input, modelos)
            
            # Persistência atómica no Data Warehouse
            self._sincronizar_fato_risco(df_final)
            
            return True

        except Exception as e:
            logger.error(f"Falha na execução da Camada Ouro: {e}")
            return False

    def _carregar_modelos_producao(self):
        """Recupera os modelos compilados na última execução de treino."""
        modelos = {}
        for persona in ["motorista", "pedestre", "motociclista"]:
            modelos[persona] = {}
            for alg in ["cat", "lgb"]:
                path = f"safedriver/modelos_ml/latest_{alg}_{persona}.pkl"
                try:
                    obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                    modelos[persona][alg] = joblib.load(io.BytesIO(obj['Body'].read()))
                except ClientError as e:
                    if 'NoSuchKey' in str(e) or '404' in str(e):
                        logger.warning(f"OURO: Modelo de produção {path} não encontrado no storage.")
                    else:
                        logger.error(f"OURO: Erro de acesso ao modelo {path}: {e}")
                    modelos[persona][alg] = None
        return modelos

    def _obter_contexto_datalake(self):
        """Busca retroativamente o cenário geográfico mais recente consolidado na Camada Prata."""
        ano_atual = datetime.now().year
        
        # Itera do ano atual para trás até encontrar o ficheiro consolidado mais recente
        for ano in range(ano_atual, 2021, -1):
            path = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                logger.info(f"OURO: Contexto de inferência alocado a partir do ciclo {ano}.")
                df = pl.read_parquet(io.BytesIO(resp['Body'].read())).to_pandas().fillna(0)
                
                # Injeção dinâmica de Meta-Features (Métricas de erro)
                for persona in ["motorista", "pedestre", "motociclista"]:
                    try:
                        meta_path = f"safedriver/modelos_ml/meta_perf_{persona}.json"
                        meta_obj = self.s3.get_object(Bucket=self.bucket, Key=meta_path)
                        meta = json.loads(meta_obj['Body'].read())
                        df['ULTIMO_MAE_CAT'] = meta.get('mae_cat', 0.0)
                        df['ULTIMO_MAE_LGB'] = meta.get('mae_lgb', 0.0)
                    except:
                        df['ULTIMO_MAE_CAT'], df['ULTIMO_MAE_LGB'] = 0.0, 0.0
                        
                return df # Retorna o DataFrame assim que encontra o ano mais recente disponível
                
            except ClientError as e:
                # Se não encontrar, continua a busca pelo ano anterior
                continue
                
        return None

    def _gerar_scores_ponderados(self, df, modelos):
        """Aplica o Ensemble de ML e os multiplicadores de negócio (Calendário)."""
        features = self.features_base + self.features_delta + self.meta_features
        multiplicadores = self.cal.obter_multiplicadores()
        
        for persona, mods in modelos.items():
            nome_coluna = f"RISCO_PREDICAO_ATUAL_{persona.upper()}"
            
            # Predição matemática primária
            p_cat = mods["cat"].predict(df[features]) if mods["cat"] is not None else 0
            p_lgb = mods["lgb"].predict(df[features]) if mods["lgb"] is not None else 0
            
            score_base = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
            df[nome_coluna] = score_base
            
            # Aplicação de heurística de negócio baseada em tipologia H3
            is_comercial = df['TOTAL_NAO_RES_H3'] > df['INDICE_RESIDENCIAL']
            df.loc[is_comercial, nome_coluna] *= multiplicadores["comercial"]
            df.loc[~is_comercial, nome_coluna] *= multiplicadores["residencial"]
            
            # Aplicação de heurística global (Feriados/Semana de Pagamento)
            df[nome_coluna] = (df[nome_coluna] * multiplicadores["geral"])
            
            # Normalização estrita da escala (0 a 100)
            max_risk = df[nome_coluna].max() if df[nome_coluna].max() > 0 else 1
            df[nome_coluna] = (df[nome_coluna] / max_risk * 100).clip(0, 100)
            
        df['TIMESTAMP_SINCRONIZACAO'] = datetime.now()
        return df

    def _sincronizar_fato_risco(self, df):
        """Executa operação atómica MERGE (Upsert) para garantir integridade analítica."""
        tabela_destino = f"{self.project_id}.{self.dataset_id}.fato_risco_predicao_atual"
        tabela_staging = f"{tabela_destino}_staging"
        
        # 1. Carregamento Full-Load para a Tabela de Staging Temporária
        config_job = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df, tabela_staging, job_config=config_job).result()

        # 2. Operação MERGE para evitar duplicidades no índice geográfico (H3_INDEX)
        query_merge = f"""
        MERGE `{tabela_destino}` T
        USING `{tabela_staging}` S
        ON T.H3_INDEX = S.H3_INDEX
        WHEN MATCHED THEN
          UPDATE SET 
            T.RISCO_PREDICAO_ATUAL_MOTORISTA = S.RISCO_PREDICAO_ATUAL_MOTORISTA,
            T.RISCO_PREDICAO_ATUAL_PEDESTRE = S.RISCO_PREDICAO_ATUAL_PEDESTRE,
            T.RISCO_PREDICAO_ATUAL_MOTOCICLISTA = S.RISCO_PREDICAO_ATUAL_MOTOCICLISTA,
            T.TIMESTAMP_SINCRONIZACAO = S.TIMESTAMP_SINCRONIZACAO,
            T.DELTA_MOTORISTA = S.DELTA_MOTORISTA
        WHEN NOT MATCHED THEN
          INSERT ROW
        """
        self.bq_client.query(query_merge).result()
        logger.info("✅ OURO: Sincronização e Upsert no BigQuery concluídos com sucesso.")

if __name__ == "__main__":
    CamadaOuroSafeDriver().executar_predicao_atual()
