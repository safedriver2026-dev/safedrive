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
    """
    Modulo de Analytics Engineering responsavel pela orquestracao do Data Warehouse.
    Realiza a transferencia dos artefatos processados no Data Lake (Cloudflare R2)
    para o Google BigQuery, estruturando o modelo analitico (Star Schema) atraves
    da criacao de Views para consumo em plataformas de Business Intelligence.
    """
    def __init__(self):
        self.project_id = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9")
        self.dataset_id = os.getenv("BQ_DATASET_ID")
        
        bq_json_str = os.getenv("BQ_SERVICE_ACCOUNT_JSON")
        if not bq_json_str:
            raise ValueError("Credenciais de servico do BigQuery (Secret) ausentes.")
            
        credentials = service_account.Credentials.from_service_account_info(json.loads(bq_json_str))
        self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
        
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # Tratamento de integridade do endpoint do Data Lake
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
        """Leitura de arquivos colunares em nuvem utilizando stream de memoria."""
        print(f"Efetuando download do artefato: {key}", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pl.read_parquet(io.BytesIO(obj['Body'].read())).to_pandas()

    def _upload_table(self, df_pandas, table_name):
        """Executa a carga de dados (Upload Truncate) para o Data Warehouse."""
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        job = self.bq_client.load_table_from_dataframe(df_pandas, table_id, job_config=job_config)
        job.result()
        print(f"Tabela {table_name} sincronizada com sucesso no BigQuery.", flush=True)

    def executar_deploy(self):
        print("Iniciando pipeline de integracao com o Data Warehouse...", flush=True)

        # =================================================================
        # 1. CARGA DA TABELA FATO (EVENTOS PREDITIVOS)
        # =================================================================
        print("Processando Dossiê de Eventos (Tabela Fato)...", flush=True)
        df_eventos = self._ler_parquet_r2("datalake/ouro/looker_dossie_eventos.parquet")
        
        # Normalizacao de tipagem para conformidade com o schema estrito do BigQuery
        if 'DATAOCORRENCIA' in df_eventos.columns:
            df_eventos['DATAOCORRENCIA'] = pd.to_datetime(df_eventos['DATAOCORRENCIA'], errors='coerce')
        if 'HORAOCORRENCIA' in df_eventos.columns:
            df_eventos['HORAOCORRENCIA'] = df_eventos['HORAOCORRENCIA'].astype(str)
            
        self._upload_table(df_eventos, "tb_dossie_eventos")

        # =================================================================
        # 2. CARGA DA TABELA DIMENSÃO (EXPLICABILIDADE SHAP)
        # =================================================================
        print("Processando Dimensao de Inteligencia (Metricas SHAP)...", flush=True)
        df_shap = self._ler_parquet_r2("datalake/ouro/looker_dim_shap.parquet")
        self._upload_table(df_shap, "tb_dim_shap")

        # =================================================================
        # 3. CONSTRUÇÃO DA CAMADA SEMÂNTICA (MASTER VIEW)
        # =================================================================
        print("Construindo a Camada Semantica (Master View) via SQL...", flush=True)
        
        sql = f"""
        CREATE OR REPLACE VIEW `{self.project_id}.{self.dataset_id}.vw_safedriver_dossie_master` AS
        SELECT 
            e.*,
            -- Conversao de coordenadas decimais para o tipo geoespacial nativo do BigQuery
            ST_GEOGPOINT(CAST(e.LONGITUDE AS FLOAT64), CAST(e.LATITUDE AS FLOAT64)) AS GEOMETRIA_PONTO,
            
            -- Metrica de auditoria: Delta entre Risco Preditivo (IA) e Risco Real Observado
            (e.RISCO_PREDITO_IA - e.LABEL_PESO_RISCO) AS DELTA_IA_REAL,
            
            s.* EXCEPT(CIDADE, BAIRRO)
            
        FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos` e
        LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_dim_shap` s 
            ON e.CIDADE = s.CIDADE AND e.BAIRRO = s.BAIRRO
        """
        
        self.bq_client.query(sql).result()
        print("Deploy concluido. View 'vw_safedriver_dossie_master' instanciada com sucesso.", flush=True)

if __name__ == "__main__":
    DeploySafeDriverBigQuery().executar_deploy()
