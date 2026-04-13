import os
import pandas as pd
import boto3
import traceback
import io
import joblib
import h3
import json
from datetime import datetime
from google.cloud import bigquery
from google.oauth2 import service_account

# Importações Absolutas para evitar ModuleNotFoundError
from autobot.comunicador import RoboComunicador

class SincronizacaoOuro:
    def __init__(self, robo: RoboComunicador):
        self.robo = robo
        self.s3 = boto3.client('s3',
                                endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                                aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                                aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))
        self.bucket = os.environ.get("R2_BUCKET_NAME")
        self.projeto_id = os.environ.get("BQ_PROJECT_ID")
        self.dataset_id = os.environ.get("BQ_DATASET_ID")
        self.credenciais_json = os.environ.get("BQ_SERVICE_ACCOUNT_JSON")

    def carregar_modelos_ia(self):
        """Carrega os pesos do Ensemble do R2"""
        try:
            obj_lgb = self.s3.get_object(Bucket=self.bucket, Key="modelos/lgbm_v1.pkl")
            obj_cat = self.s3.get_object(Bucket=self.bucket, Key="modelos/catboost_v1.pkl")
            return joblib.load(io.BytesIO(obj_lgb['Body'].read())), joblib.load(io.BytesIO(obj_cat['Body'].read()))
        except:
            self.robo.enviar_alerta_tecnico("Modelos não encontrados", "Usando heurística básica para a Ouro.")
            return None, None

    def suavizar_risco_espacial(self, df):
        """Distribui o risco entre vizinhos H3 para evitar efeito tabuleiro"""
        # Agrupa média de risco por hexágono
        mapa_risco = df.groupby('H3_INDEX')['PREVISAO_PONTUAL'].mean().to_dict()
        
        def calcular_vizinhanca(hex_id):
            vizinhos = h3.grid_disk(hex_id, 1) # Hexágono + 6 vizinhos
            valores = [mapa_risco.get(v, 0) for v in vizinhos]
            return (sum(valores) / len(valores)) if valores else 0

        df['RISCO_FINAL'] = df['H3_INDEX'].apply(calcular_vizinhanca)
        return df

    def executar_predicao_ensemble(self, df):
        lgb, cat = self.carregar_modelos_ia()
        
        # Features completas vindas da Prata (incluindo Residências)
        recursos = [
            'DENSIDADE_LOGRADOUROS', 
            'TOTAL_RESIDENCIAS_H3', 
            'TOTAL_EDIFICACOES_H3', 
            'TOTAL_NAO_RESIDENCIAIS_H3', 
            'PROPORCAO_RESIDENCIAL_H3', 
            'DIVERSIDADE_LOGRADOUROS_H3',
            'PESO_CRIME'
        ]
        
        x = df[recursos].fillna(0)

        if lgb and cat:
            # Ponderação do Ensemble: 40% LGBM / 60% CatBoost
            p_lgb = lgb.predict_proba(x)[:, 1]
            p_cat = cat.predict_proba(x)[:, 1]
            df['PREVISAO_PONTUAL'] = (p_lgb * 0.4) + (p_cat * 0.6)
        else:
            # Fallback caso a IA não esteja disponível
            df['PREVISAO_PONTUAL'] = (df['PESO_CRIME'] * 0.3) + (df['DENSIDADE_LOGRADOUROS'] * 0.7)
            
        # Aplica suavização e arredonda para 1 casa decimal (Ex: 8.4)
        df = self.suavizar_risco_espacial(df)
        df['RISCO_FINAL'] = df['RISCO_FINAL'].astype(float).round(1)
        
        return df

    def executar(self, ano):
        caminho_prata = f"datalake/prata/ssp_{ano}_prata.parquet"
        tabela_bq = f"{self.projeto_id}.{self.dataset_id}.ouro_{ano}"

        try:
            self.robo.enviar_relatorio_operacional(f"Gerando Inteligência Ouro {ano}")

            # Lê da Prata
            resp = self.s3.get_object(Bucket=self.bucket, Key=caminho_prata)
            df_prata = pd.read_parquet(io.BytesIO(resp['Body'].read()))

            # Processa Predição
            df_ouro = self.executar_predicao_ensemble(df_prata)
            df_ouro['DATA_PROCESSAMENTO'] = datetime.now()

            # Sincronização BigQuery (Delta Sync via Overwrite de Tabela Anual)
            info_bq = json.loads(self.credenciais_json)
            creds = service_account.Credentials.from_service_account_info(info_bq)
            cliente = bigquery.Client(project=self.projeto_id, credentials=creds)
            
            job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            job = cliente.load_table_from_dataframe(df_ouro, tabela_bq, job_config=job_config)
            job.result()

            self.robo.enviar_relatorio_operacional(f"Ouro Finalizada {ano}", {"Registros": len(df_ouro)})
            return True

        except Exception:
            self.robo.enviar_alerta_tecnico(f"Erro Camada Ouro {ano}", traceback.format_exc())
            return False
