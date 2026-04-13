import polars as pl
import pandas as pd
import boto3
import joblib
import io
import os
import logging
from datetime import datetime
from google.cloud import bigquery
from botocore.exceptions import ClientError
from autobot.comunicador import ComunicadorSafeDriver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class CamadaOuroSafeDriver:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        self.project_id = os.getenv("BQ_PROJECT_ID", "").strip()
        self.dataset_id = os.getenv("BQ_DATASET_ID", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        self.bq_client = bigquery.Client(project=self.project_id)
        self.comunicador = ComunicadorSafeDriver()

    def processar_ouro(self, ano):
        path_prata = f"datalake/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            logger.info(f"OURO: Iniciando sincronizacao ensemble do ano {ano}...")
            
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_prata)
            lf = pl.read_parquet(io.BytesIO(resp['Body'].read()))

            modelos = self._carregar_modelos_ensemble()
            
            df_final = self._gerar_scores_ensemble(lf, modelos)

            self._sincronizar_bigquery(df_final)

            return True

        except Exception as e:
            logger.error(f"Falha na Camada Ouro para o ano {ano}: {e}")
            self.comunicador.relatar_erro(f"Ouro Sync - {ano}", str(e))
            raise e

    def _carregar_modelos_ensemble(self):
        modelos = {}
        for p in ["motorista", "pedestre", "motociclista"]:
            modelos[p] = {}
            for alg in ["catboost", "lightgbm"]:
                path = f"modelos_ml/{alg}_{p}.pkl"
                try:
                    obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                    modelos[p][alg] = joblib.load(io.BytesIO(obj['Body'].read()))
                    logger.info(f"Modelo {alg} {p} carregado.")
                except ClientError:
                    logger.warning(f"Modelo {alg} {p} ausente.")
                    modelos[p][alg] = None
        return modelos

    def _gerar_scores_ensemble(self, lf, modelos):
        df = lf.to_pandas()
        features = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RESIDENCIAL', 'DENSIDADE_ENDERECOS']

        for p, mods in modelos.items():
            col = f"SCORE_RISCO_{p.upper()}"
            preds_final = []
            
            if mods["catboost"]:
                preds_final.append(mods["catboost"].predict(df[features]))
            
            if mods["lightgbm"]:
                preds_final.append(mods["lightgbm"].predict(df[features]))

            if preds_final:
                avg_preds = pd.DataFrame(preds_final).mean().values
                df[col] = (avg_preds / avg_preds.max() * 100).clip(0, 100)
            else:
                df[col] = 50.0

        df['ULTIMA_ATUALIZACAO'] = pd.Timestamp.now()
        return df

    def _sincronizar_bigquery(self, df):
        tabela = f"{self.project_id}.{self.dataset_id}.fato_risco_consolidado"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND",
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
            source_format=bigquery.SourceFormat.PARQUET,
        )

        logger.info(f"Enviando dados para o BigQuery: {tabela}")
        job = self.bq_client.load_table_from_dataframe(df, tabela, job_config=job_config)
        job.result()
        logger.info("Sincronizacao BigQuery concluida.")

if __name__ == "__main__":
    ouro = CamadaOuroSafeDriver()
    ano_inicio = 2022
    ano_fim = datetime.now().year
    
    for a in range(ano_inicio, ano_fim + 1):
        ouro.processar_ouro(a)
