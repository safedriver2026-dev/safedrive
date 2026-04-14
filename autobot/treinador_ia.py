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
        self.metricas_acumuladas = []

    def treinar_modelo_mestre(self):
        logger.info(f"IA: Iniciando Pipeline H3-L9 [v{self.versao_modelo}]")
        
        df_treino = self._carregar_datalake_consolidado()
        
        if df_treino is None or df_treino.shape[0] < 100:
            logger.warning("IA: Massa de dados insuficiente para H3-L9.")
            return False

        colunas_ia = self.features_numericas + self.features_categoricas

        for persona, target in self.personas.items():
            try:
                logger.info(f"--- Treinando Persona: {persona.upper()} ---")
                
                X = df_treino[colunas_ia].copy()
                y = df_treino[target]
                
                for col in self.features_categoricas:
                    X[col] = X[col].astype(str).fillna("DESCONHECIDO")

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model_cat = CatBoostRegressor(
                    iterations=1000,
                    depth=8,
                    learning_rate=0.03,
                    cat_features=self.features_categoricas,
                    loss_function='MAE',
                    verbose=0
                )
                model_cat.fit(X_train, y_train)
                preds_cat = model_cat.predict(X_test)
                
                model_lgb = LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.02,
                    num_leaves=64,
                    importance_type='gain',
                    verbosity=-1
                )
                model_lgb.fit(X_train, y_train)
                preds_lgb = model_lgb.predict(X_test)

                mae_cat = mean_absolute_error(y_test, preds_cat)
                mae_lgb = mean_absolute_error(y_test, preds_lgb)
                
                self.metricas_acumuladas.append({"persona": persona, "mae_cat": mae_cat, "mae_lgb": mae_lgb})

                self._exportar_artefactos(model_cat, f"cat_{persona}")
                self._exportar_artefactos(model_lgb, f"lgb_{persona}")
                self._atualizar_meta_perf(persona, mae_cat, mae_lgb)
                
            except Exception as e:
                logger.error(f"Erro no treinamento {persona}: {e}")
                continue

        return True

    def obter_metricas_finais(self):
        return self.metricas_acumuladas

    def _carregar_datalake_consolidado(self):
        lista_dfs = []
        ano_atual = datetime.now().year
        
        for ano in range(2022, ano_atual + 1):
            path = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                df_ano = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                df_ano = df_ano.with_columns([pl.all().cast(pl.Float64, strict=False).fill_null(0.0)])
                lista_dfs.append(df_ano)
            except ClientError:
                continue
        
        if not lista_dfs: return None
            
        df_consolidado = pl.concat(lista_dfs, how="diagonal").to_pandas()
        
        if 'POPULACAO_H3' in df_consolidado.columns and 'TOTAL_RESIDENCIAS' in df_consolidado.columns:
            df_consolidado['PERFIL_AREA'] = np.where(
                (df_consolidado['POPULACAO_H3'] > 50), "RESIDENCIAL",
                np.where(df_consolidado['POPULACAO_H3'] == 0, "COMERCIAL_INDUSTRIAL", "MISTO")
            )
        else:
            df_consolidado['PERFIL_AREA'] = "MISTO"
            
        return df_consolidado

    def _atualizar_meta_perf(self, persona, mae_cat, mae_lgb):
        path = f"safedriver/modelos_ml/meta_perf_{persona}.json"
        dados = {"mae_cat": float(mae_cat), "mae_lgb": float(mae_lgb), "timestamp": str(datetime.now())}
        self.s3.put_object(Bucket=self.bucket, Key=path, Body=json.dumps(dados))

    def _exportar_artefactos(self, modelo, nome_base):
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()
        
        self.s3.put_object(Bucket=self.bucket, Key=f"safedriver/modelos_ml/versions/v{self.versao_modelo}_{nome_base}.pkl", Body=payload)
        self.s3.put_object(Bucket=self.bucket, Key=f"safedriver/modelos_ml/latest_{nome_base}.pkl", Body=payload)

if __name__ == "__main__":
    TreinadorEvolutivo().treinar_modelo_mestre()
