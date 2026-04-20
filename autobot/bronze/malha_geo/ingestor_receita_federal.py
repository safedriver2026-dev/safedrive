import os
import hashlib
import polars as pl
import boto3
from google.cloud import bigquery
from datetime import datetime

class IngestorSafeDriverBronze:
    def __init__(self):
        # 1. Configurações de Nuvem
        self.project_id = os.getenv('BQ_PROJECT_ID', 'safe-driver-fc3a9')
        self.client = bigquery.Client(project=self.project_id)
        
        # 2. Configurações de Segurança (LGPD)
        self.salt = os.getenv('LGPD_SALT', 'default_salt')
        self.pepper = os.getenv('LGPD_PEPPER', 'default_pepper')
        
        # 3. Configurações do R2 Cloudflare
        self.s3_client = boto3.client('s3',
            region_name='auto',
            endpoint_url=os.getenv('R2_ENDPOINT_URL', '').strip(),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID', '').strip(),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY', '').strip()
        )
        self.bucket_name = os.getenv('R2_BUCKET_NAME')
        self.r2_path = "datalake/bronze/malha_raw/CNPJ_SP_HISTORICO.parquet"

    def gerar_hash_lgpd(self, cnpj):
        """Cria identificador único pseudonimizado (One-Way Hash)"""
        base_string = f"{str(cnpj)}{self.salt}{self.pepper}"
        return hashlib.sha256(base_string.encode()).hexdigest()

    def obter_query_ingestao(self):
        """
        Query otimizada para Custo Zero e Alta Performance.
        - INNER JOIN reduz o volume de dados na origem.
        - Filtros estritos (>= 2015) evitam full-scans desnecessários, mantendo a janela do Delta 2022.
        """
        return """
        SELECT 
            t1.cnpj,
            t1.cnae_fiscal_principal,
            t1.data_inicio_atividade,
            t1.data_situacao_cadastral,
            t1.situacao_cadastral,
            t1.identificador_matriz_filial, -- Essencial para entender a ocupação corporativa vs varejo local
            t1.cep,
            ST_Y(t2.centroide) AS lat,
            ST_X(t2.centroide) AS lon
        FROM `basedosdados.br_me_cnpj.estabelecimentos` AS t1
        INNER JOIN `basedosdados.br_bd_diretorios_brasil.cep` AS t2 
            ON t1.cep = t2.cep
        WHERE t1.sigla_uf = 'SP' 
          AND t1.data_inicio_atividade >= '2015-01-01' 
          AND t1.cnae_fiscal_principal IS NOT NULL
          AND t2.centroide IS NOT NULL
        """

    def executar(self):
        print(f"🚀 [BRONZE] Iniciando extração otimizada do BigQuery (Projeto: {self.project_id})")
        
        try:
            # 1. Busca dados via BigQuery API
            query_job = self.client.query(self.obter_query_ingestao())
            df_pandas = query_job.to_dataframe()
            df = pl.from_pandas(df_pandas)
            
            print(f"🛡️ [LGPD] Protegendo {len(df)} registros com Hashing Irreversível...")
            
            # 2. Proteção e Limpeza (Isolamento de Dados Sensíveis)
            df_bronze = df.with_columns([
                pl.col("cnpj").map_elements(self.gerar_hash_lgpd, return_dtype=pl.Utf8).alias("ID_PROTEGIDO")
            ]).drop("cnpj") # O CNPJ real é destruído da memória neste exato momento

            # 3. Salvamento Temporário
            temp_file = "bronze_temp.parquet"
            df_bronze.write_parquet(temp_file)
            
            print(f"📤 [R2] Enviando Parquet para o Cloudflare...")
            self.s3_client.upload_file(temp_file, self.bucket_name, self.r2_path)
            
            # 4. Higiene do Servidor
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            print(f"✅ [SUCESSO] Camada Bronze atualizada: {len(df_bronze)} linhas.")

        except Exception as e:
            print(f"❌ [ERRO CRÍTICO] Falha na pipeline de ingestão: {str(e)}")
            raise e

if __name__ == "__main__":
    IngestorSafeDriverBronze().executar()
