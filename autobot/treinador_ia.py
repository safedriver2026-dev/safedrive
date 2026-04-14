import polars as pl
import pandas as pd
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import joblib
import io
import os
import json
import logging
import numpy as np
from datetime import datetime
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class TreinadorEvolutivo:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        
        self.versao_modelo = datetime.now().strftime("%Y%m%d_%H%M")
        self.personas = {"geral": "TOTAL_CRIMES"}
        self.features_numericas = ['DENSIDADE']
        self.features_categoricas = ['NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']
        self.metricas_detalhadas = []

    def treinar_modelo_mestre(self):
        df_treino = self._carregar_datalake_consolidado()
        
        if df_treino is None or df_treino.shape[0] < 100:
            logger.error("IA: Dados insuficientes para treinamento.")
            return False

        colunas_ia = self.features_numericas + self.features_categoricas

        for persona, target in self.personas.items():
            try:
                X = df_treino[colunas_ia].copy()
                y = df_treino[target]
                
                # Conversão para categorias para compatibilidade LightGBM
                for col in self.features_categoricas:
                    X[col] = X[col].astype(str).fillna("DESCONHECIDO").astype('category')

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Treino CatBoost com Distribuição Tweedie (Mitigação de Zero-Inflated)
                model_cat = CatBoostRegressor(
                    iterations=1000, 
                    depth=8, 
                    learning_rate=0.03,
                    cat_features=self.features_categoricas, 
                    loss_function='Tweedie:variance_power=1.5',
                    verbose=0
                )
                model_cat.fit(X_train, y_train)
                
                # Treino LightGBM com Distribuição Tweedie
                model_lgb = LGBMRegressor(
                    n_estimators=500, 
                    learning_rate=0.02, 
                    num_leaves=64, 
                    objective='tweedie',
                    tweedie_variance_power=1.5,
                    verbosity=-1
                )
                model_lgb.fit(X_train, y_train)

                # Auditoria de Erro
                mae_cat = mean_absolute_error(y_test, model_cat.predict(X_test))
                mae_lgb = mean_absolute_error(y_test, model_lgb.predict(X_test))
                
                self.metricas_detalhadas.append({
                    "persona": persona,
                    "mae_cat": round(float(mae_cat), 4),
                    "mae_lgb": round(float(mae_lgb), 4),
                    "total_treino": len(X_train),
                    "distribuicao": "Tweedie"
                })

                self._exportar_artefactos(model_cat, f"cat_{persona}")
                self._exportar_artefactos(model_lgb, f"lgb_{persona}")
                
            except Exception as e:
                logger.error(f"Erro no treinamento evolutivo: {e}")
                return False

        return True

    def obter_metricas_finais(self):
        """Retorna os KPIs de performance para o orquestrador/Discord."""
        return self.metricas_detalhadas

    def _carregar_datalake_consolidado(self):
        lista_dfs = []
        for ano in range(2022, datetime.now().year + 1):
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=f"datalake/prata/ssp_consolidada_{ano}.parquet")
                df = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                # Garante cast numérico e preenchimento de nulos para evitar erros no Tweedie
                lista_dfs.append(df.with_columns([pl.all().cast(pl.Float64, strict=False).fill_null(0.0)]))
            except Exception: 
                continue
        
        if not lista_dfs: return None
        df = pl.concat(lista_dfs, how="diagonal").to_pandas()
        
        df['PERFIL_AREA'] = np.where(df['DENSIDADE'] > 5000, "RESIDENCIAL", 
                            np.where(df['DENSIDADE'] == 0, "COMERCIAL_INDUSTRIAL", "MISTO"))
        return df

    def _exportar_artefactos(self, modelo, nome):
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()
        self.s3.put_object(Bucket=self.bucket, Key=f"datalake/modelos_ml/versions/v{self.versao_modelo}_{nome}.pkl", Body=payload)
        self.s3.put_object(Bucket=self.bucket, Key=f"datalake/modelos_ml/latest_{nome}.pkl", Body=payload)
