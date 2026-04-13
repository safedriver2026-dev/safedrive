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
        # Infraestrutura R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # Infraestrutura BigQuery
        self.project_id = os.getenv("BQ_PROJECT_ID", "").strip()
        self.dataset_id = os.getenv("BQ_DATASET_ID", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        self.comunicador = ComunicadorSafeDriver()

        # Configuração de Pesos (Weighted Ensemble)
        self.pesos = {"catboost": 0.80, "lightgbm": 0.20}
        
        # Features do Modelo (Devem ser idênticas às do Treinador)
        self.features_base = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS']
        self.meta_features = ['ULTIMO_MAE_CAT', 'ULTIMO_MAE_LGB']

        # Autenticação GCP
        gcp_json_raw = os.getenv("GCP_SA_JSON", "").strip()
        try:
            if gcp_json_raw and gcp_json_raw.startswith("{"):
                cred_info = json.loads(gcp_json_raw)
                credentials = service_account.Credentials.from_service_account_info(cred_info)
                self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
            else:
                self.bq_client = bigquery.Client(project=self.project_id)
        except Exception as e:
            logger.error(f"Erro na autenticação BigQuery: {e}")
            raise e

    def processar_ouro(self, ano):
        path_prata = f"datalake/prata/ssp_consolidada_{ano}.parquet"
        try:
            logger.info(f"OURO: Iniciando processamento Cold Start para o ano {ano}...")
            
            # 1. Download dos dados da Prata
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_prata)
            lf = pl.read_parquet(io.BytesIO(resp['Body'].read()))

            # 2. Carregar modelos e metadados de performance
            modelos = self._carregar_modelos_latest()
            
            # 3. Gerar Scores com a lógica de Meta-Features
            df_final = self._gerar_scores_evolutivos(lf, modelos)

            # 4. Sincronização Delta (Upsert)
            self._sincronizar_bigquery_delta(df_final)
            return True
            
        except Exception as e:
            logger.error(f"Falha na Camada Ouro ({ano}): {e}")
            raise e

    def _carregar_modelos_latest(self):
        """Busca sempre as versões mais recentes dos modelos no R2."""
        modelos = {}
        for p in ["motorista", "pedestre", "motociclista"]:
            modelos[p] = {}
            for alg in ["cat", "lgb"]:
                path = f"modelos_ml/latest_{alg}_{p}.pkl"
                try:
                    obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                    modelos[p][alg] = joblib.load(io.BytesIO(obj['Body'].read()))
                except:
                    logger.warning(f"Modelo {path} não encontrado. Aguardando primeiro treino.")
                    modelos[p][alg] = None
        return modelos

    def _buscar_meta_features_cold_start(self, persona):
        """
        Implementa o preceito de Cold Start.
        Se não houver treino prévio, as features de performance assumem 0.0.
        """
        path = f"modelos_ml/meta_perf_{persona}.json"
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path)
            meta = json.loads(resp['Body'].read())
            return {
                "ULTIMO_MAE_CAT": meta.get("mae_cat", 0.0),
                "ULTIMO_MAE_LGB": meta.get("mae_lgb", 0.0)
            }
        except:
            # Lógica de Cold Start: Retorna neutro se o arquivo não existir
            return {"ULTIMO_MAE_CAT": 0.0, "ULTIMO_MAE_LGB": 0.0}

    def _gerar_scores_evolutivos(self, lf, modelos):
        df = lf.to_pandas()
        
        # Padronização de colunas
        if "TOTAL_NAO_RESIDENCIAIS_H3" in df.columns:
            df = df.rename(columns={"TOTAL_NAO_RESIDENCIAIS_H3": "TOTAL_NAO_RES_H3"})

        for persona, mods in modelos.items():
            col_score = f"SCORE_RISCO_{persona.upper()}"
            
            # Injeta as Meta-Features de Cold Start para esta persona
            meta = self._buscar_meta_features_cold_start(persona)
            df['ULTIMO_MAE_CAT'] = meta["ULTIMO_MAE_CAT"]
            df['ULTIMO_MAE_LGB'] = meta["ULTIMO_MAE_LGB"]

            features_finais = self.features_base + self.meta_features
            
            # Predição Ponderada (Ensemble)
            p_cat = mods["cat"].predict(df[features_finais]) if mods["cat"] is not None else None
            p_lgb = mods["lgb"].predict(df[features_finais]) if mods["lgb"] is not None else None

            if p_cat is not None and p_lgb is not None:
                score_raw = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
            else:
                score_raw = p_cat if p_cat is not None else (p_lgb if p_lgb is not None else 0.0)

            # Normalização 0-100 por persona
            if hasattr(score_raw, "max") and score_raw.max() > 0:
                df[col_score] = (score_raw / score_raw.max() * 100).clip(0, 100)
            else:
                df[col_score] = 0.0

        df['ULTIMA_ATUALIZACAO'] = pd.Timestamp.now()
        return df

    def _sincronizar_bigquery_delta(self, df):
        tabela_final = f"{self.project_id}.{self.dataset_id}.fato_risco_consolidado"
        tabela_stg = f"{tabela_final}_stg"

        # 1. Verifica existência da tabela (Prevenção de 404)
        try:
            self.bq_client.get_table(tabela_final)
        except NotFound:
            logger.info(f"OURO: Tabela {tabela_final} não existe. Criando schema inicial...")
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            self.bq_client.load_table_from_dataframe(df, tabela_final, job_config=job_config).result()
            return

        # 2. Se existe, realiza o MERGE Delta
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
        logger.info(f"OURO: Sincronização Delta finalizada com sucesso.")

if __name__ == "__main__":
    # Teste de execução isolada para o ano atual
    CamadaOuroSafeDriver().processar_ouro(datetime.now().year)
