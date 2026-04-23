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
        self.project_id = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9")
        self.dataset_id = os.getenv("BQ_DATASET_ID")
        
        bq_json_str = os.getenv("BQ_SERVICE_ACCOUNT_JSON")
        if not bq_json_str:
            raise ValueError("Secret BQ_SERVICE_ACCOUNT_JSON ausente.")
            
        credentials = service_account.Credentials.from_service_account_info(json.loads(bq_json_str))
        self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
        
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # ---> A CORREÇÃO: Mesma limpeza de endpoint do R2 <---
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        if endpoint.endswith(f"/{self.bucket}"):
            endpoint = endpoint[: -len(f"/{self.bucket}")]
            
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint, 
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4')
        )

    def _ler_parquet_r2(self, key):
        print(f"📥 Baixando {key} do R2...", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pl.read_parquet(io.BytesIO(obj['Body'].read())).to_pandas()

    def _upload_table(self, df_pandas, table_name):
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = self.bq_client.load_table_from_dataframe(df_pandas, table_id, job_config=job_config)
        job.result()
        print(f"✔️ Tabela {table_name} cravada no BigQuery.", flush=True)

    def executar_deploy(self):
        print("🚀 [DOSSIÊ MÓDULO 2] Iniciando carga massiva no Data Warehouse...", flush=True)

        # =================================================================
        # 1. SOBE A BASE NA ÍNTEGRA
        # =================================================================
        print("⏳ Subindo Dossiê de Eventos (Isso pode demorar alguns minutos)...", flush=True)
        df_eventos = self._ler_parquet_r2("datalake/ouro/looker_dossie_eventos.parquet")
        
        # Tratamento de segurança para o BigQuery não reclamar de datas e horas zoadas
        if 'DATAOCORRENCIA' in df_eventos.columns:
            df_eventos['DATAOCORRENCIA'] = pd.to_datetime(df_eventos['DATAOCORRENCIA'], errors='coerce')
        if 'HORAOCORRENCIA' in df_eventos.columns:
            df_eventos['HORAOCORRENCIA'] = df_eventos['HORAOCORRENCIA'].astype(str)
            
        self._upload_table(df_eventos, "tb_dossie_eventos")

        # =================================================================
        # 2. SOBE O DNA SHAP
        # =================================================================
        print("🧬 Subindo Dimensão de DNA SHAP...", flush=True)
        df_shap = self._ler_parquet_r2("datalake/ouro/looker_dim_shap.parquet")
        self._upload_table(df_shap, "tb_dim_shap")

        # =================================================================
        # 3. CRIA A MASTER VIEW DO DOSSIÊ
        # =================================================================
        print("🌟 Forjando a Master View do Dossiê para o Looker Studio...", flush=True)
        
        sql = f"""
        CREATE OR REPLACE VIEW `{self.project_id}.{self.dataset_id}.vw_safedriver_dossie_master` AS
        SELECT 
            e.*,
            -- Converte as coordenadas para o formato Geográfico Nativo do BigQuery
            ST_GEOGPOINT(CAST(e.LONGITUDE AS FLOAT64), CAST(e.LATITUDE AS FLOAT64)) AS GEOMETRIA_PONTO,
            
            -- Calculando a diferença entre o Risco Real e o Risco que a IA previu
            (e.RISCO_PREDITO_IA - e.LABEL_PESO_RISCO) AS DELTA_IA_REAL,
            
            s.* EXCEPT(CIDADE, BAIRRO)
            
        FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos` e
        LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_dim_shap` s 
            ON e.CIDADE = s.CIDADE AND e.BAIRRO = s.BAIRRO
        """
        
        self.bq_client.query(sql).result()
        print("🏆 [DOSSIÊ CONCLUÍDO] View 'vw_safedriver_dossie_master' no ar!", flush=True)

if __name__ == "__main__":
    DeploySafeDriverBigQuery().executar_deploy()
