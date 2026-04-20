import os
import hashlib
import polars as pl
import boto3
from google.cloud import bigquery
from datetime import datetime

class IngestorSafeDriverBronze:
    def __init__(self):
        # 1. Configurações de Nuvem (GCP + Cloudflare R2)
        self.project_id = os.getenv('BQ_PROJECT_ID', 'safe-driver-fc3a9')
        self.client = bigquery.Client(project=self.project_id)
        
        # 2. Segurança LGPD (Salt e Pepper para o Hashing)
        self.salt = os.getenv('LGPD_SALT', 'default_salt')
        self.pepper = os.getenv('LGPD_PEPPER', 'default_pepper')
        
        # 3. Conexão R2 (S3 API)
        self.s3_client = boto3.client('s3',
            region_name='auto',
            endpoint_url=os.getenv('R2_ENDPOINT_URL', '').strip(),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID', '').strip(),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY', '').strip()
        )
        self.bucket_name = os.getenv('R2_BUCKET_NAME')
        self.r2_path = "datalake/bronze/malha_raw/CNPJ_SP_HISTORICO.parquet"

    def gerar_hash_lgpd(self, cnpj):
        """Cria um ID único e irreversível para proteger o dado sensível"""
        base_string = f"{str(cnpj)}{self.salt}{self.pepper}"
        return hashlib.sha256(base_string.encode()).hexdigest()

    def obter_query_ingestao(self):
        """
        Query de Captura Total (Snapshot):
        - CAST AS STRING: Garante agnostismo de tipos para a Prata.
        - Filtro de Situação: Puxa TODAS as Ativas ('02') e quem fechou de 2022 em diante.
        - Otimização: Foca apenas em SP para manter o custo zero na cota mensal de 1TB.
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
          AND (
               t1.situacao_cadastral = '02' -- Empresas abertas (independente da idade)
               OR t1.data_situacao_cadastral >= '2022-01-01' -- Histórico para calcular o Delta
          )
          AND t1.cnae_fiscal_principal IS NOT NULL
          AND t2.centroide IS NOT NULL
        """

    def executar(self):
        print(f"🚀 [BRONZE] Iniciando Snapshot Total (Agnóstico) - Projeto: {self.project_id}")
        
        try:
            # 1. Extração via BigQuery Storage API (Alta Velocidade)
            query_job = self.client.query(self.obter_query_ingestao())
            df_pandas = query_job.to_dataframe()
            
            # 2. Conversão para Polars e Blindagem de Tipos (All to Utf8)
            df = pl.from_pandas(df_pandas).select(pl.all().cast(pl.Utf8))
            
            print(f"🛡️ [LGPD] Anonimizando {len(df)} registros para o Datalake...")
            
            # 3. Aplicação do Hashing e Descarte do CNPJ Real
            df_bronze = df.with_columns([
                pl.col("cnpj").map_elements(self.gerar_hash_lgpd, return_dtype=pl.Utf8).alias("ID_PROTEGIDO")
            ]).drop("cnpj")

            # 4. Persistência em Parquet e Upload para o R2
            temp_file = "bronze_master_snapshot.parquet"
            df_bronze.write_parquet(temp_file)
            
            print(f"📤 [R2] Enviando Snapshot Master para o Cloudflare...")
            self.s3_client.upload_file(temp_file, self.bucket_name, self.r2_path)
            
            # Limpeza de rastro local
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            print(f"✅ [SUCESSO] Bronze completa atualizada: {len(df_bronze)} estabelecimentos capturados.")

        except Exception as e:
            print(f"❌ [ERRO] Falha crítica na ingestão: {str(e)}")
            raise e

if __name__ == "__main__":
    IngestorSafeDriverBronze().executar()
