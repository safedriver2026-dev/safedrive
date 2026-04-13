import polars as pl
import pandas as pd
import boto3
import joblib
import io
import os
import logging
from datetime import datetime
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class TreinadorEvolutivo:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        
        # Definição das Personas e seus alvos (targets) na Prata
        self.personas = {
            "motorista": "TOTAL_CRIMES_MOTORISTA",
            "pedestre": "TOTAL_CRIMES_PEDESTRE",
            "motociclista": "TOTAL_CRIMES_MOTOCICLISTA"
        }
        
        # Features baseadas nos indicadores geográficos que criamos
        self.features = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS']

    def treinar_modelo_mestre(self):
        logger.info("IA: Iniciando ciclo de treinamento ensemble.")
        df_treino = self._carregar_base_historica()
        
        if df_treino is None or df_treino.shape[0] < 100:
            logger.warning("IA: Base histórica insuficiente para treinamento.")
            return False

        for persona, target in self.personas.items():
            logger.info(f"Treinando ensemble para: {persona.upper()}")
            
            X = df_treino[self.features]
            y = df_treino[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_test_size=0.2, random_state=42)

            # 1. CatBoost (O seu melhor modelo até agora)
            model_cat = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0)
            model_cat.fit(X_train, y_train)
            mae_cat = mean_absolute_error(y_test, model_cat.predict(X_test))

            # 2. LightGBM (O segundo elemento do ensemble)
            model_lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, verbosity=-1)
            model_lgb.fit(X_train, y_train)
            mae_lgb = mean_absolute_error(y_test, model_lgb.predict(X_test))

            logger.info(f"Persona {persona.upper()} - MAE CatBoost: {mae_cat:.4f} | MAE LightGBM: {mae_lgb:.4f}")

            # Exportação direta para o R2
            self._exportar_modelo(model_cat, f"catboost_{persona}.pkl")
            self._exportar_modelo(model_lgb, f"lightgbm_{persona}.pkl")

        return True

    def _carregar_base_historica(self):
        lista_dfs = []
        ano_atual = datetime.now().year
        
        for ano in range(2022, ano_atual + 1):
            path = f"datalake/prata/ssp_consolidada_{ano}.parquet"
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                df_ano = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                
                # Garante que a coluna de Não Residenciais tenha o nome correto
                if "TOTAL_NAO_RESIDENCIAIS_H3" in df_ano.columns:
                    df_ano = df_ano.rename({"TOTAL_NAO_RESIDENCIAIS_H3": "TOTAL_NAO_RES_H3"})
                
                lista_dfs.append(df_ano)
                logger.info(f"Dados de {ano} carregados para treino.")
            except:
                continue
        
        if lista_dfs:
            # Consolida tudo num único DataFrame do Pandas para o Scikit/CatBoost
            full_df = pl.concat(lista_dfs, how="diagonal").to_pandas()
            return full_df.fillna(0)
        return None

    def _exportar_modelo(self, modelo, nome_arquivo):
        path_r2 = f"modelos_ml/{nome_arquivo}"
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=path_r2,
                Body=buffer.getvalue()
            )
            logger.info(f"Modelo {nome_arquivo} exportado para o R2.")
        except Exception as e:
            logger.error(f"Falha ao exportar {nome_arquivo}: {e}")

if __name__ == "__main__":
    TreinadorEvolutivo().treinar_modelo_mestre()
