import os
import sys
import pandas as pd
import boto3
import traceback
from datetime import datetime
from .comunicador import RoboComunicador
from google.cloud import bigquery
import json

def executar_sincronizacao_ouro():
    robo = RoboComunicador(
        webhook_sucesso=os.environ.get("DISCORD_SUCESSO"),
        webhook_erro=os.environ.get("DISCORD_ERRO")
    )

    try:
        robo.enviar_relatorio_operacional("🚀 A iniciar a Sincronização Ouro da IA...")
        print("--- PASSO 3: Camada Ouro (Sincronização IA) ---")

        # Configuração do R2
        s3 = boto3.client('s3',
                          endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                          aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                          aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))
        bucket = os.environ.get("R2_BUCKET_NAME")
        ano = datetime.now().year
        path_silver = f"safedriver/datalake/silver/prata_{ano}.parquet"
        path_gold = f"safedriver/datalake/gold/ouro_{ano}.parquet"

        # Configuração do BigQuery
        bq_service_account_json = os.environ.get("BQ_SERVICE_ACCOUNT_JSON")
        bq_project_id = os.environ.get("BQ_PROJECT_ID")
        bq_dataset_id = os.environ.get("BQ_DATASET_ID")

        if not bq_service_account_json:
            raise ValueError("BQ_SERVICE_ACCOUNT_JSON não configurado.")
        if not bq_project_id:
            raise ValueError("BQ_PROJECT_ID não configurado.")
        if not bq_dataset_id:
            raise ValueError("BQ_DATASET_ID não configurado.")

        # Salvar credenciais temporariamente para o BigQuery
        with open("bq_credentials.json", "w") as f:
            f.write(bq_service_account_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "bq_credentials.json"

        bq_client = bigquery.Client(project=bq_project_id)
        table_id = f"{bq_project_id}.{bq_dataset_id}.ouro_{ano}"

        # 1. Baixar dados da camada Prata do R2
        print(f"DEBUG OURO: Baixando dados da camada Prata do R2: {path_silver}")
        s3.download_file(bucket, path_silver, "prata_data_temp.parquet")
        df_prata = pd.read_parquet("prata_data_temp.parquet")
        os.remove("prata_data_temp.parquet")
        print(f"DEBUG OURO: Dados da camada Prata carregados. {len(df_prata)} registros.")

        # 2. Lógica de Processamento da Camada Ouro (Exemplo: Treinamento/Inferência de IA)
        print("DEBUG OURO: Executando lógica de IA e gerando camada Ouro...")
        df_ouro = df_prata.copy()
        df_ouro['PREVISAO_RISCO'] = 0.75 # Exemplo de coluna gerada por IA
        df_ouro['DATA_PROCESSAMENTO_OURO'] = datetime.now().isoformat()

        # 3. Salvar camada Ouro no R2
        print(f"DEBUG OURO: Salvando camada Ouro no R2: {path_gold}")
        df_ouro.to_parquet("ouro_data_temp.parquet", index=False)
        s3.upload_file("ouro_data_temp.parquet", bucket, path_gold)
        os.remove("ouro_data_temp.parquet")
        print("DEBUG OURO: Camada Ouro salva com sucesso no R2.")

        # 4. Carregar camada Ouro no BigQuery
        print(f"DEBUG OURO: Carregando camada Ouro no BigQuery: {table_id}")
        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("ID_ANONIMO", "STRING"),
                bigquery.SchemaField("NUM_BO", "INTEGER"),
                bigquery.SchemaField("ANO_BO", "INTEGER"),
                bigquery.SchemaField("MES_BO", "INTEGER"),
                bigquery.SchemaField("DATA_OCORRENCIA_BO", "DATE"),
                bigquery.SchemaField("HORA_OCORRENCIA_BO", "TIME"),
                bigquery.SchemaField("PERIODO_OCORRENCIA", "STRING"),
                bigquery.SchemaField("FLAG_FLAGRANTE", "STRING"),
                bigquery.SchemaField("FLAG_VITIMA_FATAL", "STRING"),
                bigquery.SchemaField("NATUREZA_APURADA", "STRING"),
                bigquery.SchemaField("DESCRICAO_TIPO_CRIME", "STRING"),
                bigquery.SchemaField("DESCRICAO_SUBTIPO_CRIME", "STRING"),
                bigquery.SchemaField("CIDADE_OCORRENCIA", "STRING"),
                bigquery.SchemaField("BAIRRO_OCORRENCIA", "STRING"),
                bigquery.SchemaField("LOGRADOURO_OCORRENCIA", "STRING"),
                bigquery.SchemaField("NUMERO_OCORRENCIA", "STRING"),
                bigquery.SchemaField("LATITUDE", "FLOAT"),
                bigquery.SchemaField("LONGITUDE", "FLOAT"),
                bigquery.SchemaField("INDICE_H3", "STRING"),
                bigquery.SchemaField("DENSIDADE_LOGRADOUROS", "FLOAT"),
                bigquery.SchemaField("PROPORCAO_RESIDENCIAL_H3", "FLOAT"),
                bigquery.SchemaField("TOTAL_EDIFICACOES_H3", "INTEGER"),
                bigquery.SchemaField("EH_FERIADO", "BOOLEAN"),
                bigquery.SchemaField("DIA_PAGAMENTO", "BOOLEAN"),
                bigquery.SchemaField("PESO_CRIME", "INTEGER"),
                bigquery.SchemaField("PREVISAO_RISCO", "FLOAT"),
                bigquery.SchemaField("DATA_PROCESSAMENTO_OURO", "TIMESTAMP"),
            ],
            write_disposition="WRITE_TRUNCATE", # Sobrescreve a tabela a cada execução
        )

        job = bq_client.load_table_from_dataframe(df_ouro, table_id, job_config=job_config)
        job.result()
        print(f"DEBUG OURO: Carregado {job.output_rows} linhas para {table_id}")

        # Limpar credenciais temporárias
        os.remove("bq_credentials.json")
        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

        robo.enviar_relatorio_operacional("✅ Sincronização Ouro da IA concluída com sucesso!",
                                   {"Registros Ouro": len(df_ouro),
                                    "Camada": "Ouro"})

    except Exception:
        robo.enviar_alerta_tecnico("Sincronização Ouro da IA", traceback.format_exc())
        print(f"❌ Erro crítico na Sincronização Ouro da IA: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    executar_sincronizacao_ouro()
