import os
import boto3
from google.cloud import bigquery
from google.cloud import storage

# Configurações
PROJECT_ID = os.getenv('BQ_PROJECT_ID', 'safe-driver-fc3a9')
GCS_BUCKET = "safedriver"
GCS_PREFIX = "datalake/bronze/malha_raw/"
R2_BUCKET = os.getenv('R2_BUCKET_NAME')
R2_PREFIX = "datalake/bronze/malha_raw/"

class ZeladorSafeDriver:
    def __init__(self):
        self.bq_client = bigquery.Client(project=PROJECT_ID)
        self.gcs_client = storage.Client(project=PROJECT_ID)
        self.r2_client = boto3.client(
            's3',
            endpoint_url=os.getenv('R2_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY')
        )

    def r2_tem_dados(self):
        """Verifica se o R2 já está alimentado."""
        try:
            res = self.r2_client.list_objects_v2(Bucket=R2_BUCKET, Prefix=R2_PREFIX, MaxKeys=5)
            return 'Contents' in res
        except:
            return False

    def disparar_bigquery(self):
        """Manda o BigQuery transformar (LGPD) e exportar para o GCS."""
        pepper = os.getenv('LGPD_PEPPER', 'pepper_padrao')
        salt = os.getenv('LGPD_SALT', 'salt_padrao')

        sql = f"""
        EXPORT DATA OPTIONS(
          uri='gs://{GCS_BUCKET}/{GCS_PREFIX}CNPJ_SP_HISTORICO_*.parquet',
          format='PARQUET', overwrite=true
        ) AS
        SELECT 
            TO_HEX(SHA256(CONCAT(t1.cnpj, '{pepper}', '{salt}'))) AS id_protegido,
            CAST(t1.cnae_fiscal_principal AS STRING) AS cnae_fiscal_principal,
            CAST(t1.data_inicio_atividade AS STRING) AS data_inicio_atividade,
            CAST(t1.data_situacao_cadastral AS STRING) AS data_situacao_cadastral,
            CAST(t1.situacao_cadastral AS STRING) AS situacao_cadastral,
            CAST(t1.identificador_matriz_filial AS STRING) AS identificador_matriz_filial,
            CAST(t1.cep AS STRING) AS cep,
            CAST(ST_Y(t2.centroide) AS STRING) AS lat,
            CAST(ST_X(t2.centroide) AS STRING) AS lon
        FROM `basedosdados.br_me_cnpj.estabelecimentos` AS t1
        INNER JOIN `basedosdados.br_bd_diretorios_brasil.cep` AS t2 ON t1.cep = t2.cep
        WHERE t1.sigla_uf = 'SP' 
          AND (t1.situacao_cadastral = '02' OR t1.data_situacao_cadastral >= '2022-01-01')
          AND t1.cnae_fiscal_principal IS NOT NULL
          AND t2.centroide IS NOT NULL
        """
        print("🛡️ [PY] BigQuery processando 171M de linhas com Hashing...")
        self.bq_client.query(sql).result()
        print("✅ [PY] Exportação protegida concluída no GCS.")

    def limpar_gcs(self):
        """Apaga os arquivos do Google para não gerar custo de storage."""
        bucket = self.gcs_client.bucket(GCS_BUCKET)
        blobs = bucket.list_blobs(prefix=GCS_PREFIX)
        for blob in blobs:
            blob.delete()
        print("🧹 [PY] GCS limpo. Custo zero de armazenamento mantido.")

if __name__ == "__main__":
    import sys
    zelador = ZeladorSafeDriver()
    acao = sys.argv[1] if len(sys.argv) > 1 else "check"

    if acao == "check":
        if not zelador.r2_tem_dados():
            zelador.disparar_bigquery()
        else:
            print("👍 [PY] Dados já existem no R2. Pulando BigQuery.")
    
    elif acao == "clean":
        zelador.limpar_gcs()
