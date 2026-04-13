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
        self.comunicador = ComunicadorSafeDriver()
        self.pesos = {"catboost": 0.80, "lightgbm": 0.20}
        
        self.features_base = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS']
        self.features_delta = ['DELTA_MOTORISTA', 'DELTA_PEDESTRE', 'DELTA_MOTOCICLISTA']
        self.meta_features = ['ULTIMO_MAE_CAT', 'ULTIMO_MAE_LGB']

        self._conectar_bigquery()

    def _conectar_bigquery(self):
        gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
        try:
            if gcp_json.startswith("{"):
                cred_info = json.loads(gcp_json)
                credentials = service_account.Credentials.from_service_account_info(cred_info)
                self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
            else:
                self.bq_client = bigquery.Client(project=self.project_id)
        except Exception as e:
            logger.error(f"Falha na autenticação do Google Cloud: {e}")

    def executar_predicao_atual(self):
        logger.info("OURO: Iniciando processamento de Predição de Risco Atual.")
        try:
            modelos = self._carregar_modelos_producao()
            df_input = self._obter_contexto_datalake()
            
            if df_input is None:
                raise FileNotFoundError("Base de contexto (Prata) não localizada em safedriver/datalake/prata/.")

            df_final = self._gerar_scores_ponderados(df_input, modelos)
            self._sincronizar_fato_risco(df_final)
            return True
        except Exception as e:
            logger.error(f"Falha na execução da Camada Ouro: {e}")
            return False

    def _carregar_modelos_producao(self):
        modelos = {}
        for persona in ["motorista", "pedestre", "motociclista"]:
            modelos[persona] = {}
            for alg in ["cat", "lgb"]:
                # Caminho atualizado conforme estrutura do bucket
                path = f"safedriver/modelos_ml/latest_{alg}_{persona}.pkl"
                try:
                    obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                    modelos[persona][alg] = joblib.load(io.BytesIO(obj['Body'].read()))
                except ClientError:
                    modelos[persona][alg] = None
        return modelos

    def _obter_contexto_datalake(self):
        ano_atual = datetime.now().year
        for ano in range(ano_atual, 2021, -1):
            # Caminho atualizado: safedriver/datalake/prata/
            path = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                logger.info(f"OURO: Contexto de inferência baseado no ciclo {ano}.")
                df = pl.read_parquet(io.BytesIO(resp['Body'].read())).to_pandas().fillna(0)
                
                for persona in ["motorista", "pedestre", "motociclista"]:
                    try:
                        meta_path = f"safedriver/modelos_ml/meta_perf_{persona}.json"
                        meta_obj = self.s3.get_object(Bucket=self.bucket, Key=meta_path)
                        meta = json.loads(meta_obj['Body'].read())
                        df['ULTIMO_MAE_CAT'] = meta.get('mae_cat', 0.0)
                        df['ULTIMO_MAE_LGB'] = meta.get('mae_lgb', 0.0)
                    except:
                        df['ULTIMO_MAE_CAT'], df['ULTIMO_MAE_LGB'] = 0.0, 0.0
                return df
            except ClientError:
                continue
        return None

    def _gerar_scores_ponderados(self, df, modelos):
        features = self.features_base + self.features_delta + self.meta_features
        multiplicadores = self.cal.obter_multiplicadores()
        
        for persona, mods in modelos.items():
            nome_col = f"RISCO_PREDICAO_ATUAL_{persona.upper()}"
            p_cat = mods["cat"].predict(df[features]) if mods["cat"] is not None else 0
            p_lgb = mods["lgb"].predict(df[features]) if mods["lgb"] is not None else 0
            
            score_base = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
            
            df[nome_col] = score_base
            is_comercial = df['TOTAL_NAO_RES_H3'] > df['INDICE_RESIDENCIAL']
            
            df.loc[is_comercial, nome_col] *= multiplicadores["comercial"]
            df.loc[~is_comercial, nome_col] *= multiplicadores["residencial"]
            df[nome_col] = (df[nome_col] * multiplicadores["geral"])
            
            max_risk = df[nome_col].max() if df[nome_col].max() > 0 else 1
            df[nome_col] = (df[nome_col] / max_risk * 100).clip(0, 100)
            
        df['TIMESTAMP_SINCRONIZACAO'] = datetime.now()
        return df

    def _sincronizar_fato_risco(self, df):
        tabela_destino = f"{self.project_id}.{self.dataset_id}.fato_risco_predicao_atual"
        tabela_staging = f"{tabela_destino}_staging"
        
        config_job = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df, tabela_staging, job_config=config_job).result()

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
        logger.info("OURO: Sincronização BigQuery concluída.")
