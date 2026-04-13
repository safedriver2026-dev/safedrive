import polars as pl
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import boto3
import os
import io
import logging
from datetime import datetime
from autobot.comunicador import ComunicadorSafeDriver

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
        
        self.comunicador = ComunicadorSafeDriver()
        self.features = [
            'INDICE_RESIDENCIAL', 
            'TOTAL_NAO_RES_H3', 
            'DENSIDADE_ENDERECOS'
        ]

    def treinar_modelo_mestre(self):
        logger.info("IA: Iniciando ciclo de treinamento ensemble.")
        
        try:
            dfs_prata = []
            ano_atual = datetime.now().year
            
            for ano in range(2022, ano_atual + 1):
                path = f"datalake/prata/ssp_consolidada_{ano}.parquet"
                try:
                    resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                    dfs_prata.append(pl.read_parquet(io.BytesIO(resp['Body'].read())))
                    logger.info(f"Dados de {ano} carregados para treino.")
                except Exception:
                    continue

            if not dfs_prata:
                logger.error("Nenhuma base encontrada para treinamento.")
                return False

            df_total = pl.concat(dfs_prata).to_pandas()
            
            if not all(f in df_total.columns for f in self.features):
                raise ValueError(f"Features ausentes na Prata. Verifique as colunas geradas.")

            ano_max = df_total['ANO_REFERENCIA'].max()
            
            train = df_total[df_total['ANO_REFERENCIA'] < ano_max].copy()
            test = df_total[df_total['ANO_REFERENCIA'] == ano_max].copy()
            
            if train.empty:
                from sklearn.model_selection import train_test_split
                train, test = train_test_split(df_total, test_size=0.2, random_state=42)

            for persona in ["MOTORISTA", "PEDESTRE", "MOTOCICLISTA"]:
                logger.info(f"Treinando ensemble para: {persona}")
                target = f'TOTAL_CRIMES_{persona}'
                
                train[target] = train[target].fillna(0)
                test[target] = test[target].fillna(0)
                
                model_cat = CatBoostRegressor(
                    iterations=1000,
                    learning_rate=0.03,
                    depth=6,
                    l2_leaf_reg=3,
                    loss_function='MAE',
                    verbose=False
                )
                model_cat.fit(train[self.features], train[target], eval_set=(test[self.features], test[target]))
                
                model_lgb = LGBMRegressor(
                    n_estimators=1000,
                    learning_rate=0.03,
                    num_leaves=31,
                    importance_type='gain',
                    objective='regression_l1',
                    verbose=-1
                )
                model_lgb.fit(train[self.features], train[target], eval_set=[(test[self.features], test[target])])

                preds_cat = model_cat.predict(test[self.features])
                preds_lgb = model_lgb.predict(test[self.features])
                
                mae_cat = mean_absolute_error(test[target], preds_cat)
                mae_lgb = mean_absolute_error(test[target], preds_lgb)
                
                logger.info(f"Persona {persona} - MAE CatBoost: {mae_cat:.4f} | MAE LightGBM: {mae_lgb:.4f}")

                self._salvar_modelo_r2(model_cat, f"catboost_{persona.lower()}")
                self._salvar_modelo_r2(model_lgb, f"lightgbm_{persona.lower()}")

            return True

        except Exception as e:
            logger.error(f"Falha no treinamento ensemble: {e}")
            self.comunicador.relatar_erro("Treinador IA", str(e))
            raise e

    def _salvar_modelo_r2(self, model, nome):
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        
        path_modelo = f"modelos_ml/{nome}.pkl"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=path_modelo,
            Body=buffer.getvalue()
        )
        logger.info(f"Modelo {nome} exportado para o R2.")

if __name__ == "__main__":
    TreinadorEvolutivo().treinar_modelo_mestre()
