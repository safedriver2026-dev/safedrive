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
        self.personas = {
            "motorista": "TOTAL_CRIMES_MOTORISTA",
            "pedestre": "TOTAL_CRIMES_PEDESTRE"
        }
        
        self.features_numericas = ['DENSIDADE', 'POPULACAO_H3', 'DELTA_MOT', 'DELTA_PED']
        self.features_categoricas = ['NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']
        self.metricas_detalhadas = []

    def treinar_modelo_mestre(self):
        df_treino = self._carregar_datalake_consolidado()
        
        if df_treino is None or df_treino.shape[0] < 100:
            return None

        colunas_ia = self.features_numericas + self.features_categoricas

        for persona, target in self.personas.items():
            try:
                X = df_treino[colunas_ia].copy()
                y = df_treino[target]
                
                for col in self.features_categoricas:
                    X[col] = X[col].astype(str).fillna("DESCONHECIDO")

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model_cat = CatBoostRegressor(
                    iterations=1000, depth=8, learning_rate=0.03,
                    cat_features=self.features_categoricas, loss_function='MAE', verbose=0
                )
                model_cat.fit(X_train, y_train)
                
                model_lgb = LGBMRegressor(
                    n_estimators=500, learning_rate=0.02, num_leaves=64, verbosity=-1
                )
                model_lgb.fit(X_train, y_train)

                mae_cat = mean_absolute_error(y_test, model_cat.predict(X_test))
                mae_lgb = mean_absolute_error(y_test, model_lgb.predict(X_test))
                
                self.metricas_detalhadas.append({
                    "persona": persona,
                    "mae_cat": round(float(mae_cat), 4),
                    "mae_lgb": round(float(mae_lgb), 4),
                    "total_treino": len(X_train)
                })

                self._exportar_artefactos(model_cat, f"cat_{persona}")
                self._exportar_artefactos(model_lgb, f"lgb_{persona}")
                
            except Exception as e:
                logger.error(f"Erro no treinamento: {e}")
                continue

        return self._formatar_log_discord()

    def _formatar_log_discord(self):
        if not self.metricas_detalhadas: return None
        
        log = {
            "content": "🧠 **SafeDriver - Pipeline de Inteligência Artificial Finalizado**",
            "embeds": [{
                "title": f"🤖 MLOps: Treinamento Evolutivo v{self.versao_modelo}",
                "color": 3447003,
                "fields": [
                    {"name": "📍 Resolução Espacial", "value": "H3 Level 9 (Clusters de 0.1km²)", "inline": True},
                    {"name": "📚 Massa de Dados", "value": f"{self.metricas_detalhadas[0]['total_treino']} amostras", "inline": True}
                ],
                "footer": {"text": "Modelos versionados e persistidos no R2 Bucket."}
            }]
        }
        
        for m in self.metricas_detalhadas:
            log["embeds"][0]["fields"].append({
                "name": f"👤 Persona: {m['persona'].upper()}",
                "value": f"📉 CatBoost MAE: `{m['mae_cat']}`\n📉 LightGBM MAE: `{m['mae_lgb']}`",
                "inline": False
            })
            
        return log

    def _carregar_datalake_consolidado(self):
        lista_dfs = []
        for ano in range(2022, datetime.now().year + 1):
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet")
                df = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                lista_dfs.append(df.with_columns([pl.all().cast(pl.Float64, strict=False).fill_null(0.0)]))
            except: continue
        
        if not lista_dfs: return None
        df = pl.concat(lista_dfs, how="diagonal").to_pandas()
        
        df['PERFIL_AREA'] = np.where(df['POPULACAO_H3'] > 50, "RESIDENCIAL", 
                            np.where(df['POPULACAO_H3'] == 0, "COMERCIAL_INDUSTRIAL", "MISTO"))
        return df

    def _exportar_artefactos(self, modelo, nome):
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()
        self.s3.put_object(Bucket=self.bucket, Key=f"safedriver/modelos_ml/versions/v{self.versao_modelo}_{nome}.pkl", Body=payload)
        self.s3.put_object(Bucket=self.bucket, Key=f"safedriver/modelos_ml/latest_{nome}.pkl", Body=payload)
