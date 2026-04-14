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
import numpy as np
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
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
        self.pesos = {"catboost": 0.85, "lightgbm": 0.15}
        
        self._conectar_bigquery()

    def _conectar_bigquery(self):
        gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
        try:
            cred_info = json.loads(gcp_json)
            credentials = service_account.Credentials.from_service_account_info(cred_info)
            self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
        except Exception as e:
            logger.error(f"Erro na autenticação GCP: {e}")

    def executar_predicao_atual(self):
        try:
            modelos = self._carregar_modelos_producao()
            df_input = self._obter_contexto_datalake()
            
            if df_input is None: return False

            df_final = self._gerar_scores_ponderados(df_input, modelos)
            
            self._sincronizar_fato_risco(df_final)
            return self._notificar_sucesso_ouro(df_final)

        except Exception as e:
            logger.error(f"OURO: Falha na Camada Ouro: {e}")
            return False

    def _carregar_modelos_producao(self):
        modelos = {}
        for persona in ["motorista", "pedestre"]:
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
        ano_atual = datetime.now().year
        for ano in range(ano_atual, 2021, -1):
            path = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                df = pl.read_parquet(io.BytesIO(resp['Body'].read())).to_pandas()
                
                df['PERFIL_AREA'] = np.where(df['POPULACAO_H3'] > 50, "RESIDENCIAL", 
                                    np.where(df['POPULACAO_H3'] == 0, "COMERCIAL_INDUSTRIAL", "MISTO"))
                
                for col in ['NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']:
                    df[col] = df[col].astype(str).fillna("DESCONHECIDO")
                
                return df.fillna(0)
            except: continue
        return None

    def _gerar_scores_ponderados(self, df, modelos):
        features = ['DENSIDADE', 'POPULACAO_H3', 'DELTA_MOT', 'DELTA_PED', 'NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']
        multiplicadores = self.cal.obter_multiplicadores()
        
        for persona, mods in modelos.items():
            nome_col = f"SCORE_RISCO_{persona.upper()}"
            
            p_cat = mods["cat"].predict(df[features]) if mods["cat"] else 0
            p_lgb = mods["lgb"].predict(df[features]) if mods["lgb"] else 0
            
            score = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
            
            df[nome_col] = score * multiplicadores.get("geral", 1.0)
            
            mask_comercial = df['PERFIL_AREA'] == "COMERCIAL_INDUSTRIAL"
            df.loc[mask_comercial, nome_col] *= multiplicadores.get("comercial", 1.0)
            df.loc[~mask_comercial, nome_col] *= multiplicadores.get("residencial", 1.0)
            
            max_val = df[nome_col].max() if df[nome_col].max() > 0 else 1
            df[nome_col] = (df[nome_col] / max_val * 100).clip(0, 100).round(2)
            
        df['DT_ULTIMA_SINCRONIZACAO'] = datetime.now()
        return df

    def _sincronizar_fato_risco(self, df):
        tabela_destino = f"{self.project_id}.{self.dataset_id}.fato_risco_h3_atual"
        tabela_staging = f"{tabela_destino}_staging"
        
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df, tabela_staging, job_config=job_config).result()

        sql = f"""
        MERGE `{tabela_destino}` T
        USING `{tabela_staging}` S
        ON T.H3_INDEX = S.H3_INDEX
        WHEN MATCHED THEN
          UPDATE SET 
            T.SCORE_RISCO_MOTORISTA = S.SCORE_RISCO_MOTORISTA,
            T.SCORE_RISCO_PEDESTRE = S.SCORE_RISCO_PEDESTRE,
            T.DT_ULTIMA_SINCRONIZACAO = S.DT_ULTIMA_SINCRONIZACAO
        WHEN NOT MATCHED THEN
          INSERT ROW
        """
        self.bq_client.query(sql).result()

    def _notificar_sucesso_ouro(self, df):
        log_discord = {
            "content": "🏆 **SafeDriver - Sincronização Camada Ouro Finalizada**",
            "embeds": [{
                "title": "🛰️ Data Warehouse: BigQuery H3-L9",
                "color": 16766720,
                "fields": [
                    {"name": "📉 Score Médio Motorista", "value": f"{df['SCORE_RISCO_MOTORISTA'].mean():.2f}", "inline": True},
                    {"name": "📉 Score Médio Pedestre", "value": f"{df['SCORE_RISCO_PEDESTRE'].mean():.2f}", "inline": True},
                    {"name": "📍 Hexágonos Atualizados", "value": f"{len(df)}", "inline": False}
                ],
                "footer": {"text": "Inferência baseada em Recuperação Cruzada e Perfil de Área."}
            }]
        }
        return log_discord
