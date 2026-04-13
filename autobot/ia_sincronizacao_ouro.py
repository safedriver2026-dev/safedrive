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

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class CamadaOuroSafeDriver:
    def __init__(self):
        # 🛡️ SANITIZAÇÃO TOTAL: .strip() para evitar erros de header no R2 e BigQuery
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        self.project_id = os.getenv("BQ_PROJECT_ID", "").strip()
        self.dataset_id = os.getenv("BQ_DATASET_ID", "").strip()

        # Clientes de Infraestrutura
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        self.bq_client = bigquery.Client(project=self.project_id)
        self.comunicador = ComunicadorSafeDriver()

    def processar_ouro(self, ano):
        """
        Lê a Prata, aplica a IA e sincroniza com o BigQuery.
        """
        path_prata = f"datalake/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            logger.info(f"🥇 [OURO] A iniciar sincronização inteligente do ano {ano}...")
            
            # 1. Carregar dados refinados da Camada Prata
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_prata)
            lf = pl.read_parquet(io.BytesIO(resp['Body'].read()))

            # 2. Carregar Modelos de IA Treinados
            modelos = self._carregar_modelos_ia()
            
            # 3. Gerar Scores de Risco Dinâmicos
            df_final = self._gerar_scores_risco(lf, modelos)

            # 4. Deploy para o BigQuery (Data Warehouse)
            self._sincronizar_bigquery(df_final)

            return True

        except Exception as e:
            logger.error(f"❌ Falha na Camada Ouro para o ano {ano}: {e}")
            self.comunicador.relatar_erro(f"Ouro Sync - {ano}", str(e))
            raise e

    def _carregar_modelos_ia(self):
        """Recupera os ficheiros .pkl do R2"""
        modelos = {}
        for p in ["motorista", "pedestre", "motociclista"]:
            path = f"modelos_ml/catboost_{p}.pkl"
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                modelos[p] = joblib.load(io.BytesIO(obj['Body'].read()))
                logger.info(f"🧠 Modelo {p} carregado.")
            except ClientError:
                logger.warning(f"⚠️ Modelo {p} ausente. A usar fallback de densidade.")
                modelos[p] = None
        return modelos

    def _gerar_scores_risco(self, lf, modelos):
        """Aplica a lógica de inferência da IA"""
        df = lf.to_pandas()
        features = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RESIDENCIAL', 'DENSIDADE_ENDERECOS']

        for p, mod in modelos.items():
            col = f"SCORE_RISCO_{p.upper()}"
            if mod:
                # Predição via CatBoost
                preds = mod.predict(df[features])
                # Normalização 0-100
                df[col] = (preds / preds.max() * 100).clip(0, 100)
            else:
                df[col] = 50.0 # Risco neutro se não houver cérebro treinado

        df['ULTIMA_ATUALIZACAO'] = pd.Timestamp.now()
        return df

    def _sincronizar_bigquery(self, df):
        """Sincronização atómica com o Google BigQuery"""
        tabela = f"{self.project_id}.{self.dataset_id}.fato_risco_consolidado"
        
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_APPEND", # Mantém o histórico e adiciona o novo
            schema_update_options=[bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION],
            source_format=bigquery.SourceFormat.PARQUET,
        )

        logger.info(f"📤 A enviar {len(df)} linhas para o BigQuery...")
        job = self.bq_client.load_table_from_dataframe(df, tabela, job_config=job_config)
        job.result()
        logger.info(f"✅ Sincronização concluída com sucesso.")

# ==========================================
# MOTOR DE EXECUÇÃO DINÂMICO
# ==========================================
if __name__ == "__main__":
    ouro = CamadaOuroSafeDriver()
    ano_inicio = 2022
    ano_fim = datetime.now().year # Dinâmico: deteta 2026, 2027...
    
    logger.info(f"🚀 A iniciar processamento Ouro de {ano_inicio} até {ano_fim}")
    for a in range(ano_inicio, ano_fim + 1):
        ouro.processar_ouro(a)
