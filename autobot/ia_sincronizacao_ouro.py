import polars as pl
import pandas as pd
import joblib
import boto3
import io
import os
from google.cloud import bigquery
import logging

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CamadaOuroSafeDriver:
    def __init__(self):
        # Conexões R2 e BigQuery (Project: safe-driver-fc3a9)
        self.s3 = boto3.client('s3',
                                endpoint_url=os.getenv("R2_ENDPOINT_URL"),
                                aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
                                aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"))
        self.bq_client = bigquery.Client(project='safe-driver-fc3a9')
        self.bucket = os.getenv("R2_BUCKET_NAME")
        self.dataset_id = "analise_risco_sp"

    def _carregar_modelos(self):
        logger.info("🧠 Carregando modelos CatBoost (.pkl) do Data Lake...")
        modelos = {}
        for persona in ["motorista", "pedestre", "motociclista"]:
            path = f"modelos_ml/catboost_{persona}.pkl"
            obj = self.s3.get_object(Bucket=self.bucket, Key=path)
            modelos[persona] = joblib.load(io.BytesIO(obj['Body'].read()))
        return modelos

    def processar_ouro(self, ano_ref):
        logger.info(f"🏆 INICIANDO CAMADA OURO - ANO: {ano_ref}")
        
        # 1. Carregar Dados Enriquecidos da Prata
        path_prata = f"datalake/prata/ssp_consolidada_{ano_ref}.parquet"
        resp = self.s3.get_object(Bucket=self.bucket, Key=path_prata)
        lf_prata = pl.scan_parquet(io.BytesIO(resp['Body'].read()))

        # 2. Carregar Modelos
        modelos = self._carregar_modelos()

        # 3. Preparação para Inferência
        # Agrupamos por H3 para garantir que temos 1 linha por local para o dashboard
        df_base = lf_prata.collect().to_pandas()
        
        # 4. Execução de Predições e Lógica de Risco Base
        logger.info("🔮 Executando Inferência por Persona e calculando Risco Base...")
        
        for persona, modelo in modelos.items():
            p_upper = persona.upper()
            
            # Predição da IA
            # Nota: A IA usa as features do IBGE (Taxa Residencial, etc) que já estão na Prata
            df_base[f'PRED_IA_{p_upper}'] = modelo.predict(df_base[['DENSIDADE_TOTAL_ENDERECOS', 'TAXA_RESIDENCIAL']])
            
            # Lógica de Risco Base (Onde não há crimes)
            # Se PESO_CRIME for 0, usamos a Heurística de Solo
            df_base[f'SCORE_{p_upper}'] = df_base.apply(
                lambda row: self._calcular_risco_base(row) if row['TOTAL_CRIMES'] == 0 else row[f'PRED_IA_{p_upper}'],
                axis=1
            )
            
            # Atribuição de Rótulos de Motivo (Explicabilidade Simplificada)
            df_base[f'MOTIVO_{p_upper}'] = df_base.apply(
                lambda row: self._gerar_rotulo_motivo(row, p_upper), axis=1
            )

        # 5. Exportação para BigQuery
        self._upload_bigquery(df_base, f"risco_consolidado_{ano_ref}")

    def _calcular_risco_base(self, row):
        # Heurística: Áreas com baixa taxa residencial (Comerciais/Industriais) têm risco base maior
        if row['TAXA_RESIDENCIAL'] < 0.6: return 3.0 # Zona Industrial/Comercial
        if row['DENSIDADE_TOTAL_ENDERECOS'] < 50: return 2.5 # Zona de Baixa Ocupação
        return 1.5 # Residencial Padrão

    def _gerar_rotulo_motivo(self, row, persona):
        if row[f'SCORE_{persona}'] > 7.0:
            if row['TAXA_RESIDENCIAL'] < 0.5: return "Zona Industrial com baixo fluxo habitacional"
            return "Alta concentração histórica de ocorrências"
        return "Área de monitoramento preventivo"

    def _upload_bigquery(self, df, table_name):
        table_id = f"safe-driver-fc3a9.{self.dataset_id}.{table_name}"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        
        logger.info(f"📤 Subindo {len(df)} linhas para o BigQuery: {table_id}")
        self.bq_client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
        logger.info("✅ Dados disponíveis no BigQuery.")

if __name__ == "__main__":
    ouro = CamadaOuroSafeDriver()
    ouro.processar_ouro(2024)
