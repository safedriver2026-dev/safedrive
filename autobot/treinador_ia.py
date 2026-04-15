import polars as pl
import pandas as pd
import boto3
from botocore.config import Config
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
    def __init__(self, dev_mode=False):
    
        self.dev_mode = dev_mode 
        
       
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

     
        self.s3 = boto3.client('s3', endpoint_url=self.endpoint,
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key,
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}, max_pool_connections=50))
        
        self.base_path = self._localizar_datalake_real()
        logger.info(f"IA: Datalake mestre localizado em: '{self.base_path}'")
        
        self.versao_modelo = datetime.now().strftime("%Y%m%d_%H%M")
        
        self.target = "TOTAL_CRIMES"
        self.features_numericas = ['DENSIDADE', 'TAXA_VACANCIA', 'RANKING_RISCO_LOCAL', 'INDICE_EXPOSICAO']
        self.features_categoricas = ['NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA', 'PERIODO_DIA', 'PERFIL_ALVO', 'TIPO_LOCAL']
        self.metricas_detalhadas = []

    def _localizar_datalake_real(self):
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, MaxKeys=100):
                for obj in page.get('Contents', []):
                    if "datalake/prata/" in obj['Key']:
                        return obj['Key'].split("datalake/")[0] + "datalake"
            return "datalake"
        except: return "datalake"

    def _get_path(self, camada, filename):
        return f"{self.base_path}/{camada}/{filename}".replace("//", "/")

    def treinar_modelo_mestre(self):
        df_treino = self._carregar_datalake_consolidado()
        
        if df_treino is None or len(df_treino) < 50:
            logger.error("IA: Dados insuficientes para treino.")
            return False

        logger.info(f"IA: Iniciando Treino com {len(df_treino)} registros (Dev Mode: {self.dev_mode}).")

       
        colunas_ia = self.features_numericas + self.features_categoricas
        X = df_treino[colunas_ia].copy()
        y = df_treino[self.target]

        for col in self.features_categoricas:
            X[col] = X[col].astype(str).fillna("INDEFINIDO").astype('category')
        for col in self.features_numericas:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)

       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       
        logger.info("IA: [1/2] Treinando CatBoost Tweedie...")
        model_cat = CatBoostRegressor(
            iterations=1000, 
            depth=6, 
            learning_rate=0.05, 
            cat_features=self.features_categoricas,
            loss_function='Tweedie:variance_power=1.5',
            thread_count=-1, 
            early_stopping_rounds=50, 
            verbose=100
        )
        
        model_cat.fit(X_train, y_train, eval_set=(X_test, y_test))

   
        logger.info("IA: [2/2] Treinando LightGBM Tweedie...")
        model_lgb = LGBMRegressor(
            n_estimators=300, 
            learning_rate=0.05, 
            num_leaves=31,
            objective='tweedie', 
            tweedie_variance_power=1.5,
            n_jobs=-1, 
            verbosity=-1 
        )
        model_lgb.fit(X_train, y_train)

     
        mae_cat = mean_absolute_error(y_test, model_cat.predict(X_test))
        mae_lgb = mean_absolute_error(y_test, model_lgb.predict(X_test))
        
        logger.info(f"IA: MAE CatBoost: {mae_cat:.4f} | MAE LightGBM: {mae_lgb:.4f}")

        self._exportar_modelo(model_cat, "cat_geral")
        self._exportar_modelo(model_lgb, "lgb_geral")
        
        self.metricas_detalhadas.append({"mae_cat": mae_cat, "mae_lgb": mae_lgb, "total": len(X_train)})
        return True

    def _carregar_datalake_consolidado(self):
        lista_dfs = []
        anos_para_processar = [2026] if self.dev_mode else range(2022, datetime.now().year + 1)
        
        for ano in anos_para_processar:
            key = self._get_path("prata", f"ssp_consolidada_{ano}.parquet")
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=key)
                df = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                
            
                df = df.with_columns([
                    pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) 
                    for c in self.features_numericas if c in df.columns
                ])
                lista_dfs.append(df)
            except Exception as e:
                logger.warning(f"IA: Arquivo de {ano} ignorado: {e}")
                continue
        
        if not lista_dfs: return None
        
    
        df_completo_polars = pl.concat(lista_dfs, how="diagonal")
        
     
        df_completo_polars = df_completo_polars.with_columns([
            pl.when(pl.col('DENSIDADE') > 5000).then(pl.lit("RESIDENCIAL_DENSO"))
              .when(pl.col('DENSIDADE') == 0).then(pl.lit("COMERCIAL_INDUSTRIAL"))
              .otherwise(pl.lit("MISTO")).alias('PERFIL_AREA')
        ])
        
       
        return df_completo_polars.to_pandas()

    def _exportar_modelo(self, modelo, nome):
        key_ver = self._get_path("modelos_ml/versions", f"v{self.versao_modelo}_{nome}.pkl")
        key_lat = self._get_path("modelos_ml", f"latest_{nome}.pkl")
        
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()
        
        self.s3.put_object(Bucket=self.bucket, Key=key_ver, Body=payload)
        self.s3.put_object(Bucket=self.bucket, Key=key_lat, Body=payload)
        logger.info(f"IA: Modelo {nome} salvo no R2.")

    def obter_metricas_finais(self):
        return self.metricas_detalhadas

if __name__ == "__main__":
    
    treinador = TreinadorEvolutivo(dev_mode=False) 
    treinador.treinar_modelo_mestre()
