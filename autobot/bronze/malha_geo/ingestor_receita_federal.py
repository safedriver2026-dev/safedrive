import os
import hashlib
import polars as pl
import boto3
from google.cloud import bigquery
from datetime import datetime

class IngestorSafeDriverBronze:
    def __init__(self):
        # Configurações de Nuvem
        self.project_id = os.getenv('BQ_PROJECT_ID', 'safe-driver-fc3a9')
        self.client = bigquery.Client(project=self.project_id)
        
        # Configurações de Segurança
        self.salt = os.getenv('LGPD_SALT', 'default_salt')
        self.pepper = os.getenv('LGPD_PEPPER', 'default_pepper')
        
        # Configurações R2 Cloudflare
        self.s3_client = boto3.client('s3',
            region_name='auto',
            endpoint_url=os.getenv('R2_ENDPOINT_URL', '').strip(),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID', '').strip(),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY', '').strip()
        )
        self.bucket_name = os.getenv('R2_BUCKET_NAME')
        self.r2_path = "datalake/bronze/malha_raw/CNPJ_SP_HISTORICO.parquet"

    def gerar_hash_lgpd(self, cnpj):
        base_string = f"{str(cnpj)}{self.salt}{self.pepper}"
        return hashlib.sha256(base_string.encode()).hexdigest()

    def obter_query_ingestao(self):
        """
        Query Agnóstica: O CAST AS STRING em todas as colunas impede que o motor do BigQuery 
        ou do Pandas tente inferir tipos de dados, garantindo uma Bronze 100% Raw (Texto).
        """
        return """
        SELECT 
            CAST(t1.cnpj AS STRING) AS cnpj,
            CAST(t1.cnae_fiscal_principal AS STRING) AS cnae_fiscal_principal,
            CAST(t1.data_inicio_atividade AS STRING) AS data_inicio_atividade,
            CAST(t1.data_situacao_cadastral AS STRING) AS data_situacao_cadastral,
            CAST(t1.situacao_cadastral AS STRING) AS situacao_cadastral,
            CAST(t1.identificador_matriz_filial AS STRING) AS identificador_matriz_filial,
            CAST(t1.cep AS STRING) AS cep,
            CAST(ST_Y(t2.centroide) AS STRING) AS lat,
            CAST(ST_X(t2.centroide) AS STRING) AS lon
        FROM `basedosdados.br_me_cnpj.estabelecimentos` AS t1
        INNER JOIN `basedosdados.br_bd_diretorios_brasil.cep` AS t2 
            ON t1.cep = t2.cep
        WHERE t1.sigla_uf = 'SP' 
          AND t1.data_inicio_atividade >= '2015-01-01' 
          AND t1.cnae_fiscal_principal IS NOT NULL
          AND t2.centroide IS NOT NULL
        """

    def executar(self):
        print(f"🚀 [BRONZE] Extração Carga Full Agnóstica - Projeto: {self.project_id}")
        
        try:
            # 1. Download veloz do BigQuery
            query_job = self.client.query(self.obter_query_ingestao())
            df_pandas = query_job.to_dataframe()
            
            # 2. Forçar tipagem global para Utf8 (String) no Polars
            df = pl.from_pandas(df_pandas).select(pl.all().cast(pl.Utf8))
            
            print(f"🛡️ [LGPD] Anonimizando {len(df)} registros...")
            
            # 3. Proteção (One-Way Hash)
            df_bronze = df.with_columns([
                pl.col("cnpj").map_elements(self.gerar_hash_lgpd, return_dtype=pl.Utf8).alias("ID_PROTEGIDO")
            ]).drop("cnpj")

            # 4. Salvar e Sobrescrever (Esmagamento) no R2
            temp_file = "bronze_temp.parquet"
            df_bronze.write_parquet(temp_file)
            
            print("📤 [R2] Substituindo histórico anterior pelo snapshot Raw mais recente...")
            self.s3_client.upload_file(temp_file, self.bucket_name, self.r2_path)
            
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            print(f"✅ [SUCESSO] Camada Bronze Agnóstica atualizada: {len(df_bronze)} linhas.")

        except Exception as e:
            print(f"❌ [ERRO CRÍTICO] Falha na pipeline: {str(e)}")
            raise e

if __name__ == "__main__":
    IngestorSafeDriverBronze().executar()
