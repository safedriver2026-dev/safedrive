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

    def _construir_dim_calendario(self):
        """
        Gera uma Dimensão de Calendário robusta nativamente no BigQuery.
        Cobre 10 anos de dados para suportar a Fato sem falhas temporais.
        """
        print("\nGerando Dimensao Calendario nativa no BigQuery...", flush=True)
        
        sql_calendario = f"""
        CREATE OR REPLACE TABLE `{self.project_id}.{self.dataset_id}.tb_dim_calendario` AS
        WITH datas AS (
            SELECT dt AS DATA_BASE
            FROM UNNEST(GENERATE_DATE_ARRAY('2020-01-01', '2030-12-31', INTERVAL 1 DAY)) AS dt
        )
        SELECT
            DATA_BASE,
            CAST(FORMAT_DATE('%Y%m%d', DATA_BASE) AS INT64) AS ID_TEMPO,
            EXTRACT(YEAR FROM DATA_BASE) AS ANO,
            EXTRACT(MONTH FROM DATA_BASE) AS MES_NUM,
            EXTRACT(DAY FROM DATA_BASE) AS DIA,
            EXTRACT(DAYOFWEEK FROM DATA_BASE) AS DIA_SEMANA_NUM,
            FORMAT_DATE('%B', DATA_BASE) AS NOME_MES_EN,
            -- Tradução do mês para o Dash
            CASE EXTRACT(MONTH FROM DATA_BASE)
                WHEN 1 THEN 'Janeiro' WHEN 2 THEN 'Fevereiro' WHEN 3 THEN 'Março'
                WHEN 4 THEN 'Abril' WHEN 5 THEN 'Maio' WHEN 6 THEN 'Junho'
                WHEN 7 THEN 'Julho' WHEN 8 THEN 'Agosto' WHEN 9 THEN 'Setembro'
                WHEN 10 THEN 'Outubro' WHEN 11 THEN 'Novembro' WHEN 12 THEN 'Dezembro'
            END AS NOME_MES_PT,
            -- Identificação de Fim de Semana (1=Domingo, 7=Sábado no BQ)
            CASE WHEN EXTRACT(DAYOFWEEK FROM DATA_BASE) IN (1, 7) THEN 'FIM DE SEMANA' 
                 ELSE 'DIA UTIL' 
            END AS TIPO_DIA,
            EXTRACT(QUARTER FROM DATA_BASE) AS TRIMESTRE
        FROM datas;
        """
        
        self.bq_client.query(sql_calendario).result()
        print("✅ Tabela 'tb_dim_calendario' gerada com sucesso.", flush=True)

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
        # 3. GERAÇÃO DA DIMENSÃO CALENDÁRIO
        # =================================================================
        self._construir_dim_calendario()

        # =================================================================
        # 4. INSTANCIACAO DA CAMADA SEMANTICA (MASTER VIEW)
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
            s.* EXCEPT(CIDADE, BAIRRO),

            -- Dados da Dimensao Tempo (Calendario)
            cal.ANO,
            cal.MES_NUM,
            cal.NOME_MES_PT,
            cal.DIA_SEMANA_NUM,
            cal.TIPO_DIA AS CAL_TIPO_DIA,
            cal.TRIMESTRE
            
        FROM `{self.project_id}.{self.dataset_id}.tb_dossie_eventos` e
        LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_dim_shap` s 
            ON e.CIDADE = s.CIDADE AND e.BAIRRO = s.BAIRRO
        LEFT JOIN `{self.project_id}.{self.dataset_id}.tb_dim_calendario` cal
            ON DATE(e.DATAOCORRENCIA) = cal.DATA_BASE
        """
        
        self.bq_client.query(sql).result()
        print(f"✨ Deploy executado com sucesso em {datetime.now().strftime('%d/%m/%Y %H:%M')}.", flush=True)
        print(f"🗺️ Objeto '{self.project_id}.{self.dataset_id}.vw_safedriver_dossie_master' pronto para consumo no Looker Studio.", flush=True)

if __name__ == "__main__":
    DeploySafeDriverBigQuery().executar_deploy()
