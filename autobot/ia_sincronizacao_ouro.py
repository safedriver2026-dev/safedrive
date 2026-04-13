import polars as pl
import pandas as pd
import boto3
import joblib
import io
import os
import json
import logging
from datetime import datetime
from google.cloud import bigquery
from google.oauth2 import service_account
from google.cloud.exceptions import NotFound
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
        self.comunicador = ComunicadorSafeDriver()

        # Configuração de Pesos do Ensemble
        self.pesos = {"catboost": 0.80, "lightgbm": 0.20}

        # Autenticação Google Cloud (Memória ou Local)
        gcp_json_raw = os.getenv("GCP_SA_JSON", "").strip()
        try:
            if gcp_json_raw and gcp_json_raw.startswith("{"):
                logger.info("OURO: Autenticando via Service Account JSON (Memória).")
                cred_info = json.loads(gcp_json_raw)
                credentials = service_account.Credentials.from_service_account_info(cred_info)
                self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
            else:
                logger.info("OURO: Usando credenciais padrão locais.")
                self.bq_client = bigquery.Client(project=self.project_id)
        except Exception as e:
            logger.error(f"Erro ao configurar cliente BigQuery: {e}")
            raise e

    def processar_ouro(self, ano):
        path_prata = f"datalake/prata/ssp_consolidada_{ano}.parquet"
        try:
            logger.info(f"OURO: Iniciando processamento para o ano {ano}...")
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_prata)
            lf = pl.read_parquet(io.BytesIO(resp['Body'].read()))

            modelos = self._carregar_modelos_ensemble()
            df_final = self._gerar_scores_weighted_ensemble(lf, modelos)

            self._sincronizar_bigquery_delta(df_final)
            return True
        except Exception as e:
            logger.error(f"Falha na Camada Ouro para o ano {ano}: {e}")
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
                except:
                    modelos[p][alg] = None
        return modelos

    def _gerar_scores_weighted_ensemble(self, lf, modelos):
        df = lf.to_pandas()
        features = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS']

        for p, mods in modelos.items():
            col = f"SCORE_RISCO_{p.upper()}"
            
            p_cat = mods["catboost"].predict(df[features]) if mods["catboost"] is not None else None
            p_lgb = mods["lightgbm"].predict(df[features]) if mods["lightgbm"] is not None else None

            if p_cat is not None and p_lgb is not None:
                score_final = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
            else:
                score_final = p_cat if p_cat is not None else (p_lgb if p_lgb is not None else 0.0)

            # Normalização 0-100
            if hasattr(score_final, "max") and score_final.max() > 0:
                df[col] = (score_final / score_final.max() * 100).clip(0, 100)
            else:
                df[col] = 0.0

        df['ULTIMA_ATUALIZACAO'] = pd.Timestamp.now()
        return df

    def _sincronizar_bigquery_delta(self, df):
        tabela_final = f"{self.project_id}.{self.dataset_id}.fato_risco_consolidado"
        tabela_stg = f"{tabela_final}_stg"

        # Garante a existência da tabela destino
        try:
            self.bq_client.get_table(tabela_final)
        except NotFound:
            logger.info(f"OURO: Criando tabela {tabela_final}...")
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            self.bq_client.load_table_from_dataframe(df, tabela_final, job_config=job_config).result()
            return

        # Sincronização via MERGE
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df, tabela_stg, job_config=job_config).result()

        sql_merge = f"""
        MERGE `{tabela_final}` T
        USING `{tabela_stg}` S
        ON T.H3_INDEX = S.H3_INDEX AND T.ANO_REFERENCIA = S.ANO_REFERENCIA
        WHEN MATCHED THEN
          UPDATE SET 
            T.SCORE_RISCO_MOTORISTA = S.SCORE_RISCO_MOTORISTA,
            T.SCORE_RISCO_PEDESTRE = S.SCORE_RISCO_PEDESTRE,
            T.SCORE_RISCO_MOTOCICLISTA = S.SCORE_RISCO_MOTOCICLISTA,
            T.ULTIMA_ATUALIZACAO = S.ULTIMA_ATUALIZACAO
        WHEN NOT MATCHED THEN
          INSERT (H3_INDEX, ANO_REFERENCIA, SCORE_RISCO_MOTORISTA, SCORE_RISCO_PEDESTRE, SCORE_RISCO_MOTOCICLISTA, ULTIMA_ATUALIZACAO, INDICE_RESIDENCIAL, TOTAL_NAO_RES_H3, DENSIDADE_ENDERECOS)
          VALUES (S.H3_INDEX, S.ANO_REFERENCIA, S.SCORE_RISCO_MOTORISTA, S.SCORE_RISCO_PEDESTRE, S.SCORE_RISCO_MOTOCICLISTA, S.ULTIMA_ATUALIZACAO, S.INDICE_RESIDENCIAL, S.TOTAL_NAO_RES_H3, S.DENSIDADE_ENDERECOS)
        """
        self.bq_client.query(sql_merge).result()
        logger.info(f"OURO: Upsert finalizado em {tabela_final}")

if __name__ == "__main__":
    ouro = CamadaOuroSafeDriver()
    for ano in range(2022, datetime.now().year + 1):
        ouro.processar_ouro(ano)
