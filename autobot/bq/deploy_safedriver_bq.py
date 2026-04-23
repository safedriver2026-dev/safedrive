import os
import io
import json
import boto3
import polars as pl
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from botocore.config import Config

class DeploySafeDriverBigQuery:
    def __init__(self):
        # Conexão BigQuery
        self.project_id = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9")
        self.dataset_id = os.getenv("BQ_DATASET_ID")
        
        bq_json_str = os.getenv("BQ_SERVICE_ACCOUNT_JSON")
        if not bq_json_str:
            raise ValueError("Secret BQ_SERVICE_ACCOUNT_JSON ausente.")
            
        credentials = service_account.Credentials.from_service_account_info(json.loads(bq_json_str))
        self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
        
        # Conexão R2
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint, 
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4')
        )

    def _ler_parquet_r2(self, key):
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pl.read_parquet(io.BytesIO(obj['Body'].read()))

    def _upload_table(self, df_pandas, table_name):
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = self.bq_client.load_table_from_dataframe(df_pandas, table_id, job_config=job_config)
        job.result()
        print(f"✔️ Tabela {table_name} carregada no BigQuery.")

    def executar_deploy(self):
        print("🚀 [MÓDULO 2] Iniciando Deploy do Data Warehouse (Star Schema)...")

        # =================================================================
        # 1. CARGA DA TABELA FATO E DIMENSÃO SHAP (Vindas do Script 1)
        # =================================================================
        df_fato = self._ler_parquet_r2("datalake/ouro/looker_fact_predicoes.parquet").to_pandas()
        self._upload_table(df_fato, "fact_predicoes")

        df_shap = self._ler_parquet_r2("datalake/ouro/looker_dim_shap.parquet").to_pandas()
        self._upload_table(df_shap, "dim_bairros_shap")

        # =================================================================
        # 2. CONSTRUÇÃO DA DIMENSÃO DE INFRAESTRUTURA FÍSICA (H3)
        # =================================================================
        print("🏗️ Extraindo Dimensão Espacial de Infraestrutura...")
        df_ouro = self._ler_parquet_r2("datalake/ouro/safedriver_abt_treino.parquet")
        
        df_dim_infra = df_ouro.unique(subset="H3_INDEX").select([
            "H3_INDEX", "LAT", "LON", "CIDADE", "BAIRRO",
            "MACRO_FINANCEIRO", "MACRO_LAZER_NOTURNO", "MACRO_VAREJO", 
            "MACRO_LOGISTICA_INDUSTRIA", "MACRO_SERVICOS_BASE",
            "CENSO_MEDIA_V0001", "CENSO_MEDIA_V0002"
        ]).to_pandas()
        
        self._upload_table(df_dim_infra, "dim_h3_infra")

        # =================================================================
        # 3. CRIAÇÃO DA MASTER VIEW NO BIGQUERY
        # =================================================================
        print("🌟 Criando Master View para o Looker Studio...")
        
        sql = f"""
        CREATE OR REPLACE VIEW `{self.project_id}.{self.dataset_id}.vw_safedriver_master_star` AS
        SELECT 
            f.CENARIO_ID,
            f.RISCO_SCORE,
            CASE 
                WHEN f.RISCO_SCORE >= 7.0 THEN '🔴 ALTO'
                WHEN f.RISCO_SCORE >= 4.0 THEN '🟡 MÉDIO'
                ELSE '🟢 BAIXO'
            END AS ALERTA_VISUAL,
            
            i.H3_INDEX,
            i.CIDADE,
            i.BAIRRO,
            ST_GEOGPOINT(i.LON, i.LAT) AS GEOMETRIA,
            i.MACRO_FINANCEIRO,
            i.MACRO_LAZER_NOTURNO,
            i.MACRO_VAREJO,
            i.MACRO_LOGISTICA_INDUSTRIA,
            
            s.* EXCEPT(CIDADE, BAIRRO)
            
        FROM `{self.project_id}.{self.dataset_id}.fact_predicoes` f
        INNER JOIN `{self.project_id}.{self.dataset_id}.dim_h3_infra` i 
            ON f.H3_INDEX = i.H3_INDEX
        LEFT JOIN `{self.project_id}.{self.dataset_id}.dim_bairros_shap` s 
            ON i.CIDADE = s.CIDADE AND i.BAIRRO = s.BAIRRO
        """
        
        self.bq_client.query(sql).result()
        print("🏆 [MÓDULO 2 CONCLUÍDO] View 'vw_safedriver_master_star' criada e pronta para uso!")

if __name__ == "__main__":
    DeploySafeDriverBigQuery().executar_deploy()
