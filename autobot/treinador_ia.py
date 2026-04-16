import polars as pl
import pandas as pd
import boto3
from botocore.config import Config
import joblib
import io
import os
import gc  # Garbage Collector
import logging
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
        self.versao_modelo = datetime.now().strftime("%Y%m%d_%H%M")
        
        self.target = "TOTAL_CRIMES"
        
        # --- ALINHAMENTO HEAVY SILVER ---
        # Recebendo as novas colunas temporais de alta inteligência da Prata
        self.features_numericas = [
            'DENSIDADE', 'TAXA_VACANCIA', 'RANKING_RISCO_LOCAL', 'INDICE_EXPOSICAO',
            'CONTAGIO_PONDERADO', 'PRESSAO_RISCO_LOCAL', 
            'MES_OCORRENCIA', 'DIA_SEMANA', 'IS_PAGAMENTO', 'IS_FDS'
        ]
        
        self.features_categoricas = ['NM_BAIRRO', 'NM_MUN', 'PERIODO_DIA', 'PERFIL_ALVO', 'TIPO_LOCAL']
        self.stats_treino = {}

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
        
        if df_treino is None or len(df_treino) < 10:
            logger.error("IA: Massa de dados insuficiente para o ciclo evolutivo.")
            return False

        logger.info(f"IA: Iniciando Treinamento Ensemble Otimizado com {len(df_treino)} registros.")

        X = df_treino[self.features_numericas + self.features_categoricas].copy()
        y = df_treino[self.target].fillna(0).astype('float32') 

        # Elimina o dataframe original pesado da memória RAM
        del df_treino 
        gc.collect()

        for col in self.features_categoricas:
            X[col] = X[col].astype(str).fillna("INDEFINIDO").astype('category')
            
        for col in self.features_numericas:
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0).astype('float32')

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 1. CatBoost (Tuned para Amostragem)
        logger.info("IA: [1/2] Lapidando CatBoost (Hyper-Tuned)...")
        model_cat = CatBoostRegressor(
            iterations=600, 
            depth=6, 
            learning_rate=0.06, 
            l2_leaf_reg=5,             # TUNING: Impede Overfitting na amostra de 600k
            cat_features=self.features_categoricas,
            loss_function='Tweedie:variance_power=1.5',
            early_stopping_rounds=30, 
            verbose=100,
            thread_count=2
        )
        model_cat.fit(X_train, y_train, eval_set=(X_test, y_test))

        # 2. LightGBM (Tuned para Extração de Features)
        logger.info("IA: [2/2] Lapidando LightGBM (Hyper-Tuned)...")
        model_lgb = LGBMRegressor(
            n_estimators=400, 
            learning_rate=0.06, 
            objective='tweedie', 
            tweedie_variance_power=1.5,
            feature_fraction=0.8,      # TUNING: Obriga a IA a usar as features novas e não focar só na densidade
            importance_type='gain',
            n_jobs=2,
            verbosity=-1 
        )
        model_lgb.fit(X_train, y_train)

        mae_cat = mean_absolute_error(y_test, model_cat.predict(X_test))
        mae_lgb = mean_absolute_error(y_test, model_lgb.predict(X_test))
        
        logger.info(f"IA: MAE CatBoost: {mae_cat:.4f} | MAE LightGBM: {mae_lgb:.4f}")

        self._exportar_modelo(model_cat, "cat_geral")
        self._exportar_modelo(model_lgb, "lgb_geral")
        
        self.stats_treino = {
            "mae": round(min(mae_cat, mae_lgb), 4),
            "modelo_vencedor": "CatBoost" if mae_cat < mae_lgb else "LightGBM",
            "volume_treino": len(X_train),
            "data_versao": self.versao_modelo
        }
        return True

    def _carregar_datalake_consolidado(self):
        lista_dfs = []
        anos = [datetime.now().year] if self.dev_mode else range(2022, datetime.now().year + 1)
        
        for ano in anos:
            key = self._get_path("prata", f"ssp_consolidada_{ano}.parquet")
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=key)
                df = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                lista_dfs.append(df)
            except:
                logger.warning(f"IA: Dados de {ano} não encontrados na Prata.")
                continue
        
        if not lista_dfs: return None
        
        df_pl = pl.concat(lista_dfs, how="diagonal")
        
        del lista_dfs
        gc.collect()
        
        mapeamento = {"NM_MUN_FINAL": "NM_MUN", "NM_BAIRRO_FINAL": "NM_BAIRRO"}
        df_pl = df_pl.rename({old: new for old, new in mapeamento.items() if old in df_pl.columns})

        for col in self.features_categoricas:
            if col not in df_pl.columns:
                df_pl = df_pl.with_columns(pl.lit("INDEFINIDO").alias(col))

        for col in self.features_numericas:
            if col in df_pl.columns:
                df_pl = df_pl.with_columns(pl.col(col).cast(pl.Float32, strict=False).fill_null(0.0))
            else:
                df_pl = df_pl.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(col))

        if self.target not in df_pl.columns:
            df_pl = df_pl.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(self.target))

        # SALVAÇÃO DO OOM KILLER (Mantido)
        LIMITE_LINHAS = 600000 
        if df_pl.height > LIMITE_LINHAS:
            logger.warning(f"IA: Amostrando dataset de {df_pl.height} para {LIMITE_LINHAS} registros para proteger a memória.")
            df_pl = df_pl.sample(n=LIMITE_LINHAS, seed=42)

        return df_pl.to_pandas()

    def _exportar_modelo(self, modelo, nome):
        key_ver = self._get_path("modelos_ml/versions", f"v{self.versao_modelo}_{nome}.pkl")
        key_lat = self._get_path("modelos_ml", f"latest_{nome}.pkl")
        
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()
        
        self.s3.put_object(Bucket=self.bucket, Key=key_ver, Body=payload)
        self.s3.put_object(Bucket=self.bucket, Key=key_lat, Body=payload)
        logger.info(f"IA: Modelo {nome} persistido no R2 (Versão: {self.versao_modelo}).")

    def obter_stats(self):
        return self.stats_treino

if __name__ == "__main__":
    treinador = TreinadorEvolutivo(dev_mode=False) 
    treinador.treinar_modelo_mestre()
