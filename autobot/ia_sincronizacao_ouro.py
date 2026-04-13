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
        # Conexões Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # Conexões Google BigQuery
        self.project_id = os.getenv("BQ_PROJECT_ID", "").strip()
        self.dataset_id = os.getenv("BQ_DATASET_ID", "").strip()

        # BLINDAGEM PARA CLOUDFLARE R2
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        
        self.cal = CalendarioEstrategico()
        self.comunicador = ComunicadorSafeDriver()

        # Configuração de Pesos do Ensemble (Produção)
        self.pesos = {"catboost": 0.80, "lightgbm": 0.20}
        
        # Features alinhadas com o Treinador
        self.features_base = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS']
        self.features_delta = ['DELTA_MOTORISTA', 'DELTA_PEDESTRE', 'DELTA_MOTOCICLISTA']
        self.meta_features = ['ULTIMO_MAE_CAT', 'ULTIMO_MAE_LGB']

        self._conectar_bigquery()

    def _conectar_bigquery(self):
        gcp_json = os.getenv("GCP_SA_JSON", "").strip()
        try:
            if gcp_json.startswith("{"):
                cred_info = json.loads(gcp_json)
                credentials = service_account.Credentials.from_service_account_info(cred_info)
                self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
            else:
                self.bq_client = bigquery.Client(project=self.project_id)
        except Exception as e:
            logger.error(f"Erro Crítico ao autenticar BigQuery: {e}")

    def executar_predicao_atual(self):
        """Orquestra a inferência e sincronização dos dados de risco."""
        logger.info("OURO: Iniciando motor de Predição Atual...")
        
        try:
            modelos = self._carregar_modelos_vivos()
            df_input = self._obter_contexto_recente()
            
            if df_input is None:
                raise Exception("Base de contexto (Prata) não localizada no Data Lake.")

            df_predicao = self._gerar_scores_com_contexto(df_input, modelos)
            self._upsert_bigquery(df_predicao)
            
            return True

        except Exception as e:
            logger.error(f"FALHA NA CAMADA OURO: {e}")
            return False

    def _carregar_modelos_vivos(self):
        modelos = {}
        for p in ["motorista", "pedestre", "motociclista"]:
            modelos[p] = {}
            for alg in ["cat", "lgb"]:
                path = f"modelos_ml/latest_{alg}_{p}.pkl"
                try:
                    obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                    modelos[p][alg] = joblib.load(io.BytesIO(obj['Body'].read()))
                except ClientError as e:
                    if 'NoSuchKey' in str(e):
                        logger.warning(f"Modelo {path} não encontrado no storage.")
                    else:
                        logger.error(f"Erro de acesso ao modelo {path}: {e}")
                    modelos[p][alg] = None
        return modelos

    def _obter_contexto_recente(self):
        """Busca a Prata mais atual via direct read (sem ListObjects) e injeta Meta-Features."""
        ano_atual = datetime.now().year
        
        for ano in range(ano_atual, 2021, -1):
            path = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                logger.info(f"OURO: Utilizando Datalake Prata referência {ano}.")
                df = pl.read_parquet(io.BytesIO(resp['Body'].read())).to_pandas().fillna(0)
                
                # Injeta a memória de erro (MAE)
                for p in ["motorista", "pedestre", "motociclista"]:
                    try:
                        meta_resp = self.s3.get_object(Bucket=self.bucket, Key=f"modelos_ml/meta_perf_{p}.json")
                        meta = json.loads(meta_resp['Body'].read())
                        df['ULTIMO_MAE_CAT'] = meta.get('mae_cat', 0.0)
                        df['ULTIMO_MAE_LGB'] = meta.get('mae_lgb', 0.0)
                    except:
                        df['ULTIMO_MAE_CAT'], df['ULTIMO_MAE_LGB'] = 0.0, 0.0
                return df
                
            except ClientError as e:
                # Se não achou o arquivo desse ano, vai para o anterior silenciosamente
                if 'NoSuchKey' in str(e):
                    continue
                else:
                    logger.error(f"Erro de permissão ou rede ao acessar a Prata de {ano}: {e}")
                    continue
        return None

    def _gerar_scores_com_contexto(self, df, modelos):
        features = self.features_base + self.features_delta + self.meta_features
        multiplicadores = self.cal.obter_multiplicadores()
        
        for persona, mods in modelos.items():
            col_target = f"RISCO_PREDICAO_ATUAL_{persona.upper()}"
            
            p_cat = mods["cat"].predict(df[features]) if mods["cat"] is not None else 0
            p_lgb = mods["lgb"].predict(df[features]) if mods["lgb"] is not None else 0
            
            score_ia = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
            
            df[col_target] = score_ia
            
            # Ajuste de Negócio (Calendário)
            mask_comercial = df['TOTAL_NAO_RES_H3'] > df['INDICE_RESIDENCIAL']
            df.loc[mask_comercial, col_target] *= multiplicadores["comercial"]
            df.loc[~mask_comercial, col_target] *= multiplicadores["residencial"]
            
            df[col_target] = (df[col_target] * multiplicadores["geral"])
            max_val = df[col_target].max() if df[col_target].max() > 0 else 1
            df[col_target] = (df[col_target] / max_val * 100).clip(0, 100)
            
        df['TIMESTAMP_PREDICAO'] = datetime.now()
        return df

    def _upsert_bigquery(self, df):
        tabela_final = f"{self.project_id}.{self.dataset_id}.fato_risco_predicao_atual"
        tabela_stg = f"{tabela_final}_stg"
        
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df, tabela_stg, job_config=job_config).result()

        sql_merge = f"""
        MERGE `{tabela_final}` T
        USING `{tabela_stg}` S
        ON T.H3_INDEX = S.H3_INDEX
        WHEN MATCHED THEN
          UPDATE SET 
            T.RISCO_PREDICAO_ATUAL_MOTORISTA = S.RISCO_PREDICAO_ATUAL_MOTORISTA,
            T.RISCO_PREDICAO_ATUAL_PEDESTRE = S.RISCO_PREDICAO_ATUAL_PEDESTRE,
            T.RISCO_PREDICAO_ATUAL_MOTOCICLISTA = S.RISCO_PREDICAO_ATUAL_MOTOCICLISTA,
            T.TIMESTAMP_PREDICAO = S.TIMESTAMP_PREDICAO,
            T.DELTA_MOTORISTA = S.DELTA_MOTORISTA
        WHEN NOT MATCHED THEN
          INSERT ROW
        """
        self.bq_client.query(sql_merge).result()
        logger.info(f"OURO: Predição Atual sincronizada e atualizada no BigQuery.")

if __name__ == "__main__":
    CamadaOuroSafeDriver().executar_predicao_atual()
