import polars as pl
import pandas as pd
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import joblib
import io
import os
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
        
        # --- LÓGICA DE LOCALIZAÇÃO AUTOMÁTICA ---
        self.base_path = self._descobrir_prefixo_datalake()
        logger.info(f"IA: Raiz do Data Lake detectada em: '{self.base_path}'")
        
        self.versao_modelo = datetime.now().strftime("%Y%m%d_%H%M")
        self.personas = {"geral": "TOTAL_CRIMES"}
        
        self.features_numericas = ['DENSIDADE', 'TAXA_VACANCIA']
        self.features_categoricas = ['NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']
        self.metricas_detalhadas = []

    def _descobrir_prefixo_datalake(self):
        """Localiza a pasta datalake independentemente da estrutura do bucket."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, MaxKeys=15)
            if 'Contents' in response:
                keys = [obj['Key'] for obj in response['Contents']]
                for key in keys:
                    if "safedriver/datalake" in key: return "safedriver/datalake"
                    if "datalake" in key: return "datalake"
            return "datalake"
        except: return "datalake"

    def _get_path(self, camada, subpasta, filename):
        return f"{self.base_path}/{camada}/{subpasta}/{filename}".replace("//", "/")

    def treinar_modelo_mestre(self):
        df_treino = self._carregar_datalake_consolidado()
        
        if df_treino is None or df_treino.shape[0] < 50: # Reduzi o limiar para testes
            logger.error(f"IA: Dados insuficientes ({0 if df_treino is None else df_treino.shape[0]} linhas).")
            return False

        colunas_ia = self.features_numericas + self.features_categoricas

        for persona, target in self.personas.items():
            try:
                X = df_treino[colunas_ia].copy()
                y = df_treino[target]
                
                for col in self.features_categoricas:
                    X[col] = X[col].astype(str).fillna("DESCONHECIDO").astype('category')
                
                for col in self.features_numericas:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                logger.info(f"IA: Treinando CatBoost Tweedie ({persona})...")
                model_cat = CatBoostRegressor(
                    iterations=1000, depth=8, learning_rate=0.03,
                    cat_features=self.features_categoricas, 
                    loss_function='Tweedie:variance_power=1.5', verbose=0
                )
                model_cat.fit(X_train, y_train)
                
                logger.info(f"IA: Treinando LightGBM Tweedie ({persona})...")
                model_lgb = LGBMRegressor(
                    n_estimators=500, learning_rate=0.02, num_leaves=64, 
                    objective='tweedie', tweedie_variance_power=1.5, verbosity=-1
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
                logger.error(f"IA: Erro no treinamento ({persona}): {e}")
                return False

        return True

    def _carregar_datalake_consolidado(self):
        lista_dfs = []
        ano_atual = datetime.now().year
        
        for ano in range(2022, ano_atual + 1):
            key = self._get_path("prata", "", f"ssp_consolidada_{ano}.parquet")
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=key)
                df = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                
                # Normalização Polars
                df = df.with_columns([
                    pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) 
                    for c in ["TOTAL_CRIMES", "DENSIDADE", "TAXA_VACANCIA"] if c in df.columns
                ])
                lista_dfs.append(df)
            except: continue
        
        if not lista_dfs: return None
            
        df = pl.concat(lista_dfs, how="diagonal").to_pandas()
        
        # Feature Engineering: Perfil da Área
        df['PERFIL_AREA'] = np.select(
            [(df['TAXA_VACANCIA'] > 0.3), (df['DENSIDADE'] > 5000), (df['DENSIDADE'] == 0)],
            ["ALTA_VACANCIA", "RESIDENCIAL_DENSO", "COMERCIAL_INDUSTRIAL"], 
            default="MISTO"
        )
        return df

    def _exportar_artefactos(self, modelo, nome):
        key_version = self._get_path("modelos_ml/versions", "", f"v{self.versao_modelo}_{nome}.pkl")
        key_latest = self._get_path("modelos_ml", "", f"latest_{nome}.pkl")
        
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()
        
        self.s3.put_object(Bucket=self.bucket, Key=key_version, Body=payload)
        self.s3.put_object(Bucket=self.bucket, Key=key_latest, Body=payload)

    def obter_metricas_finais(self):
        return self.metricas_detalhadas
