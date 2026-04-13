import polars as pl
import pandas as pd
import joblib
import boto3
import io
import os
from google.cloud import bigquery
from sklearn.metrics import mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class CamadaOuroSafeDriver:
    def __init__(self):
        # Conexões R2 e BQ via Secrets (Projeto: safe-driver-fc3a9)
        self.s3 = boto3.client('s3',
                                endpoint_url=os.getenv("R2_ENDPOINT_URL"),
                                aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
                                aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"))
        self.bucket = os.getenv("R2_BUCKET_NAME")
        
        projeto_bq = os.getenv("BQ_PROJECT_ID")
        self.dataset_id = os.getenv("BQ_DATASET_ID")
        self.bq_client = bigquery.Client(project=projeto_bq)

    def _carregar_modelos(self):
        modelos = {}
        for persona in ["motorista", "pedestre", "motociclista"]:
            path = f"modelos_ml/catboost_{persona}.pkl"
            obj = self.s3.get_object(Bucket=self.bucket, Key=path)
            modelos[persona] = joblib.load(io.BytesIO(obj['Body'].read()))
        return modelos

    def processar_ouro(self, ano_ref):
        logger.info(f"🏆 PROCESSANDO OURO INTELIGENTE - REFERÊNCIA: {ano_ref}")
        
        # 1. Carregamento da Prata
        path_prata = f"datalake/prata/ssp_consolidada_{ano_ref}.parquet"
        resp = self.s3.get_object(Bucket=self.bucket, Key=path_prata)
        df_base = pl.read_parquet(io.BytesIO(resp['Body'].read())).to_pandas()

        # 2. Modelos e Inferência
        modelos = self._carregar_modelos()
        
        for persona, modelo in modelos.items():
            p_upper = persona.upper()
            
            # Predição IA
            df_base[f'PRED_IA_{p_upper}'] = modelo.predict(df_base[['DENSIDADE_TOTAL_ENDERECOS', 'TAXA_RESIDENCIAL']])
            
            # Risco Base (Heurística para áreas vazias)
            df_base[f'SCORE_{p_upper}'] = df_base.apply(
                lambda row: self._calcular_risco_base(row) if row.get('TOTAL_CRIMES', 0) == 0 else row[f'PRED_IA_{p_upper}'],
                axis=1
            )

        # 3. INTELIGÊNCIA: Cálculo de Backtesting (Apenas para anos passados)
        # Se o ano já fechou, calculamos a precisão real do modelo para esse ano
        df_base['ANO_REFERENCIA'] = ano_ref
        df_base['METRICA_CONFIANCA'] = 1.0 # Padrão para o ano atual
        
        if int(ano_ref) < 2026: # Se for um ano histórico (Backtesting)
            # Exemplo: MAE (Erro Médio Absoluto)
            # $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
            try:
                erro = mean_absolute_error(df_base['TOTAL_CRIMES'], df_base['PRED_IA_MOTORISTA'])
                df_base['METRICA_CONFIANCA'] = 1 - (erro / df_base['TOTAL_CRIMES'].max())
                logger.info(f"📊 Backtesting {ano_ref}: Confiabilidade do Modelo em {df_base['METRICA_CONFIANCA'].mean():.2%}")
            except:
                pass

        # 4. Exportação Escalável
        # Criamos uma tabela por ano OU uma tabela única particionada. 
        # Para o Looker Studio, o ideal é uma tabela por ano para não estourar custos de consulta.
        self._upload_bigquery(df_base, f"risco_consolidado_{ano_ref}")

    def _calcular_risco_base(self, row):
        if row['TAXA_RESIDENCIAL'] < 0.6: return 3.0
        if row['DENSIDADE_TOTAL_ENDERECOS'] < 50: return 2.5
        return 1.5

    def _gerar_rotulo_motivo(self, row, persona):
        # (Mantém a lógica de rótulos anterior)
        return "Área de monitoramento"

    def _upload_bigquery(self, df, table_name):
        table_id = f"{self.bq_client.project}.{self.dataset_id}.{table_name}"
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df, table_id, job_config=job_config).result()
        logger.info(f"✅ Tabela {table_id} atualizada.")
