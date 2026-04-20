import os
from google.cloud import bigquery

def executar_exportacao_bronze():
    project_id = os.getenv('BQ_PROJECT_ID', 'safe-driver-fc3a9')
    client = bigquery.Client(project=project_id)
    
    # O SQL que você rodou no console, agora automatizado
    sql = """
    EXPORT DATA OPTIONS(
      uri='gs://safedriver/datalake/bronze/malha_raw/CNPJ_SP_HISTORICO_*.parquet',
      format='PARQUET', overwrite=true
    ) AS
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
    INNER JOIN `basedosdados.br_bd_diretorios_brasil.cep` AS t2 ON t1.cep = t2.cep
    WHERE t1.sigla_uf = 'SP' 
      AND (t1.situacao_cadastral = '02' OR t1.data_situacao_cadastral >= '2022-01-01')
      AND t1.cnae_fiscal_principal IS NOT NULL
      AND t2.centroide IS NOT NULL
    """
    
    print(f"🚀 Disparando Job de Exportação no BigQuery (US)...")
    query_job = client.query(sql)
    query_job.result() # Espera terminar
    print(f"✅ Exportação concluída com sucesso no GCS!")

if __name__ == "__main__":
    executar_exportacao_bronze()
