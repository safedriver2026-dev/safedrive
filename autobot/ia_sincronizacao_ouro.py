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
from .comunicador import RoboComunicador

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
        try:
            obj_lgb = self.s3.get_object(Bucket=self.bucket, Key="modelos/lgbm_safedriver_v1.pkl")
            obj_cat = self.s3.get_object(Bucket=self.bucket, Key="modelos/catboost_safedriver_v1.pkl")
            return joblib.load(io.BytesIO(obj_lgb['Body'].read())), joblib.load(io.BytesIO(obj_cat['Body'].read()))
        except:
            return None, None

    def aplicar_suavizacao_espacial(self, df):
        mapa_risco = df.groupby('H3_INDEX')['RISCO_PONTUAL'].mean().to_dict()
        
        def calcular_vizinhanca(hex_id):
            vizinhos = h3.grid_disk(hex_id, 1)
            valores = [mapa_risco.get(v, 0) for v in vizinhos]
            return sum(valores) / len(valores) if valores else 0

        df['RISCO_PREDITIVO_FINAL'] = df['H3_INDEX'].apply(calcular_vizinhanca)
        return df

    def executar_ensemble(self, df):
        lgb, cat = self.carregar_modelos_ia()
        
        features = [
            'DENSIDADE_LOGRADOUROS', 
            'TOTAL_RESIDENCIAS_H3', 
            'TOTAL_EDIFICACOES_H3', 
            'TOTAL_NAO_RESIDENCIAIS_H3', 
            'PROPORCAO_RESIDENCIAL_H3', 
            'DIVERSIDADE_LOGRADOUROS_H3',
            'PESO_CRIME'
        ]
        
        x = df[features].fillna(0)

        if lgb and cat:
            p_lgb = lgb.predict_proba(x)[:, 1]
            p_cat = cat.predict_proba(x)[:, 1]
            df['RISCO_PONTUAL'] = (p_lgb * 0.4) + (p_cat * 0.6)
        else:
            df['RISCO_PONTUAL'] = (df['PESO_CRIME'] * 0.3) + (df['DENSIDADE_LOGRADOUROS'] * 0.7)
            
        return self.aplicar_suavizacao_espacial(df)

    def executar(self, ano):
        caminho_prata = f"datalake/prata/ssp_{ano}_prata.parquet"
        caminho_ouro_r2 = f"datalake/ouro/ssp_{ano}_ouro.parquet"
        tabela_bq = f"{self.projeto_id}.{self.dataset_id}.ouro_{ano}"

        try:
            self.robo.enviar_relatorio_operacional(f"Iniciando Ensemble Ouro {ano}")

            resp = self.s3.get_object(Bucket=self.bucket, Key=caminho_prata)
            df_prata = pd.read_parquet(io.BytesIO(resp['Body'].read()))

            df_ouro = self.executar_ensemble(df_prata)
            df_ouro['DATA_PROCESSAMENTO'] = datetime.now()

            buffer = io.BytesIO()
            df_ouro.to_parquet(buffer, index=False)
            self.s3.put_object(Bucket=self.bucket, Key=caminho_ouro_r2, Body=buffer.getvalue())

            info_bq = json.loads(self.credenciais_json)
            creds = service_account.Credentials.from_service_account_info(info_bq)
            cliente_bq = bigquery.Client(project=self.projeto_id, credentials=creds)
            
            job = cliente_bq.load_table_from_dataframe(
                df_ouro, 
                tabela_bq, 
                job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            )
            job.result()

            self.robo.enviar_relatorio_operacional(f"Ouro Finalizada {ano}", {"Registros": len(df_ouro)})
            return True

        except Exception:
            self.robo.enviar_alerta_tecnico(f"Erro Camada Ouro {ano}", traceback.format_exc())
            return False

def iniciar_ouro(robo, ano):
    servico = SincronizacaoOuro(robo)
    return servico.executar(ano)
