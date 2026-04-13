import polars as pl
import pandas as pd
import boto3
import joblib
import io
import os
import json
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
        self.version = datetime.now().strftime("%Y%m%d_%H%M")
        self.personas = {
            "motorista": "TOTAL_CRIMES_MOTORISTA",
            "pedestre": "TOTAL_CRIMES_PEDESTRE",
            "motociclista": "TOTAL_CRIMES_MOTOCICLISTA"
        }
        self.features_base = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS']
        self.meta_features = ['ULTIMO_MAE_CAT', 'ULTIMO_MAE_LGB']

    def treinar_modelo_mestre(self):
        logger.info(f"IA: Iniciando Ciclo [{self.version}]")
        df_treino = self._carregar_base_com_meta_features()
        
        if df_treino is None or df_treino.shape[0] < 20: # Reduzi para 20 para facilitar o Cold Start
            logger.warning(f"IA: Base insuficiente ({df_treino.shape[0] if df_treino is not None else 0} linhas).")
            return False

        features_finais = self.features_base + self.meta_features
        for persona, target in self.personas.items():
            if target not in df_treino.columns:
                logger.warning(f"IA: Alvo {target} nao encontrado nos dados. Pulando persona.")
                continue

            X = df_treino[features_finais]
            y = df_treino[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model_cat = CatBoostRegressor(iterations=500, depth=6, verbose=0).fit(X_train, y_train)
            mae_cat = mean_absolute_error(y_test, model_cat.predict(X_test))
            
            model_lgb = LGBMRegressor(n_estimators=100, verbosity=-1).fit(X_train, y_train)
            mae_lgb = mean_absolute_error(y_test, model_lgb.predict(X_test))

            logger.info(f"📊 {persona.upper()} | MAE Cat: {mae_cat:.4f} | MAE LGB: {mae_lgb:.4f}")
            self._exportar_modelo(model_cat, f"cat_{persona}")
            self._exportar_modelo(model_lgb, f"lgb_{persona}")
            self._persistir_metadados_performance(persona, mae_cat, mae_lgb)

        return True

    def _carregar_base_com_meta_features(self):
        df = self._carregar_base_historica_debug() 
        if df is None: return None
        for persona in self.personas.keys():
            meta = self._buscar_metadados_performance(persona)
            df['ULTIMO_MAE_CAT'] = meta.get('mae_cat', 0.0)
            df['ULTIMO_MAE_LGB'] = meta.get('mae_lgb', 0.0)
        return df

    def _carregar_base_historica_debug(self):
        lista_dfs = []
        # Lista arquivos no bucket para depuracao
        try:
            objs = self.s3.list_objects_v2(Bucket=self.bucket, Prefix="datalake/prata/")
            keys = [obj['Key'] for obj in objs.get('Contents', [])]
            logger.info(f"🔍 Arquivos encontrados na Prata: {keys}")
        except Exception as e:
            logger.error(f"Erro ao listar bucket: {e}")

        for ano in range(2022, datetime.now().year + 1):
            path = f"datalake/prata/ssp_consolidada_{ano}.parquet"
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                df_temp = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                
                # Mapeamento flexivel de colunas
                mapping = {
                    "TOTAL_NAO_RESIDENCIAIS_H3": "TOTAL_NAO_RES_H3",
                    "DENSIDADE": "DENSIDADE_ENDERECOS"
                }
                for old, new in mapping.items():
                    if old in df_temp.columns:
                        df_temp = df_temp.rename({old: new})
                
                lista_dfs.append(df_temp)
                logger.info(f"✅ Ano {ano} carregado com {df_temp.shape[0]} linhas.")
            except:
                continue
        
        if not lista_dfs: return None
        return pl.concat(lista_dfs, how="diagonal").to_pandas().fillna(0)

    def _buscar_metadados_performance(self, persona):
        path = f"modelos_ml/meta_perf_{persona}.json"
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path)
            return json.loads(resp['Body'].read())
        except:
            return {"mae_cat": 0.0, "mae_lgb": 0.0}

    def _persistir_metadados_performance(self, persona, mae_cat, mae_lgb):
        path = f"modelos_ml/meta_perf_{persona}.json"
        meta = {"mae_cat": mae_cat, "mae_lgb": mae_lgb, "versao": self.version}
        self.s3.put_object(Bucket=self.bucket, Key=path, Body=json.dumps(meta))

    def _exportar_modelo(self, modelo, nome_base):
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()
        self.s3.put_object(Bucket=self.bucket, Key=f"modelos_ml/versions/v{self.version}_{nome_base}.pkl", Body=payload)
        self.s3.put_object(Bucket=self.bucket, Key=f"modelos_ml/latest_{nome_base}.pkl", Body=payload)
        logger.info(f"💾 Modelo {nome_base} atualizado.")

if __name__ == "__main__":
    TreinadorEvolutivo().treinar_modelo_mestre()
