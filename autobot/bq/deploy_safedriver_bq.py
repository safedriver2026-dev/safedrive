import os
import io
import json
import boto3
import polars as pl
import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account
from botocore.config import Config
from datetime import datetime

class DeploySafeDriverBigQuery:
    """
    Componente de Analytics Engineering responsavel pela sustentacao do Data Warehouse.
    Orquestra a carga dos artefatos processados pelo modelo de Machine Learning
    para o Google BigQuery, estruturando a camada semantica (Star Schema) 
    otimizada para ferramentas de Business Intelligence e analise geoespacial.
    """
    def __init__(self):
        # Prioriza o ID definido pelo usuario nas correcoes de historico
        self.project_id = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9")
        self.dataset_id = os.getenv("BQ_DATASET_ID")
        
        if not self.dataset_id:
            raise ValueError("A variavel BQ_DATASET_ID precisa estar configurada.")
            
        # Inicializacao das credenciais de servico via Secret
        bq_json_str = os.getenv("BQ_SERVICE_ACCOUNT_JSON")
        if not bq_json_str:
            raise ValueError("Credenciais de servico do BigQuery (Secret) ausentes.")
            
        credentials = service_account.Credentials.from_service_account_info(json.loads(bq_json_str))
        self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
        
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # Configuracao de integridade para o endpoint do Cloudflare R2
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        if endpoint.endswith(f"/{self.bucket}"):
            endpoint = endpoint[: -len(f"/{self.bucket}")]
            
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint, 
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4', retries={'max_attempts': 3})
        )

    def _ler_parquet_r2(self, key):
        """Leitura eficiente de arquivos Parquet via buffer de memoria com blindagem NaN."""
        print(f"Acessando artefato no Data Lake: {key}", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        
        # Le com Polars e converte para Pandas
        df_pandas = pl.read_parquet(io.BytesIO(obj['Body'].read())).to_pandas()
        
        # BLINDAGEM: BigQuery nao aceita Float NaN do Pandas em colunas genericas.
        # Converte strings vazias/NaNs de forma aceitavel pelo conector do BQ.
        df_pandas = df_pandas.fillna(pd.NA)
        return df_pandas

    def _upload_table(self, df_pandas, table_name):
        """Executa a persistencia de dados no Data Warehouse (Modo Write Truncate)."""
        table_id = f"{self.project_id}.{self.dataset_id}.{table_name}"
        
        # O autodetect salva o pipeline caso a Camada Ouro crie uma coluna nova
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
            autodetect=True 
        )
        
        print(f"Iniciando job de carga para {table_id}...", flush=True)
        job = self.bq_client.load_table_from_dataframe(df_pandas, table_id, job_config=job_config)
        job.result() # Aguarda a conclusao da API do BQ
        print(f"✅ Carga finalizada: Tabela {table_name} atualizada no BigQuery.", flush=True)

    def executar_deploy(self):
        print("Iniciando pipeline de Deploy Analitico (Sincronizacao R2 -> BQ)...", flush=True)

        # =================================================================
        # 1. CARGA DA TABELA FATO: EVENTOS COM PREDICAO DE RISCO
        # =================================================================
        print("\nProcessando Dossie de Vulnerabilidade (Tabela Fato)...", flush=True)
        df_eventos = self._ler_parquet_r2("datalake/ouro/looker_dossie_eventos.parquet")
        
        # Ajuste de tipos primitivos para conformidade com o BigQuery
        if 'DATAOCORRENCIA' in df_eventos.columns:
            df_eventos['DATAOCORRENCIA'] = pd.to_datetime(df_eventos['DATAOCORRENCIA'], errors='coerce')
        if 'HORAOCORRENCIA' in df_eventos.columns:
            df_eventos['HORAOCORRENCIA'] = df_eventos['HORAOCORRENCIA'].astype(str)
            
        self._upload_table(df_eventos, "tb_dossie_eventos")

        # =================================================================
        # 2. CARGA DA TABELA DIMENSAO: DNA CRIMINAL (VALORES SHAP)
        # =================================================================
        print("\nProcessando Metricas de Explicabilidade SHAP (Tabela Dimensao)...", flush=True)
        df_shap = self._ler_parquet_r2("datalake/ouro/looker_dim_shap.parquet")
        self._upload_table(df_shap, "tb_dim_shap")

        # =================================================================
        # 3. INSTANCIACAO DA CAMADA SEMANTICA (MASTER VIEW)
        # =================================================================
        print("\nGerando Camada Semantica Geoespacial (Star Schema View)...", flush=True)
        
        sql = f"""
        CREATE OR REPLACE VIEW `{self.project_id}.{self.dataset_id}.vw_safedriver_dossie_master` AS
        SELECT 
            e.*,
            -- Conversao dinamica de coordenadas para o tipo GEOGRAPHY nativo
            ST_GEOGPOINT(CAST(e.LONGITUDE AS FLOAT64), CAST(e.LATITUDE AS FLOAT64)) AS GEOMETRIA_PONTO,
            
            -- Calculo de Residuos: Divergencia entre Risco Observado e Risco Preditivo
            (e.RISCO_PREDITO_IA - e.LABEL_PESO_RISCO) AS DELTA_IA_REAL,
            
            -- Join com a Dimensao SHAP para analise de DNA Criminal por localidade
            s.* EXCEPT(CIDADE, BAIRRO)
            
        FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos` e
        LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_dim_shap` s 
            ON e.CIDADE = s.CIDADE AND e.BAIRRO = s.BAIRRO
        """
        
        self.bq_client.query(sql).result()
        print(f"Deploy executado com sucesso em {datetime.now().strftime('%d/%m/%Y %H:%M')}.", flush=True)
        print(f"Objeto '{self.project_id}.{self.dataset_id}.vw_safedriver_dossie_master' pronto para consumo no Looker Studio.", flush=True)

if __name__ == "__main__":
    DeploySafeDriverBigQuery().executar_deploy()
