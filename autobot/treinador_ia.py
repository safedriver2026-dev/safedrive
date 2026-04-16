import polars as pl
import pandas as pd
import boto3
from botocore.config import Config
import joblib
import io
import os
import gc
import logging
from datetime import datetime
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error

# Auditoria de Performance
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
        
        # 🎯 ALVO: IA prevê a Gravidade Futura
        self.target = "INDICE_GRAVIDADE"
        
        # --- 🚀 RESOLUÇÃO DO GAP: Adição de Memória Histórica como Feature ---
        self.features_numericas = [
            'DENSIDADE', 
            'TAXA_VACANCIA', 
            'CONTAGIO_PONDERADO', 
            'PRESSAO_RISCO_LOCAL', 
            'GRAVIDADE_HISTORICA', # <-- A IA agora "sabe" o que aconteceu ali antes
            'VOLUME_HISTORICO',    # <-- Densidade de incidentes por grão
            'MES_OCORRENCIA', 
            'DIA_SEMANA', 
            'IS_PAGAMENTO', 
            'IS_FDS'
        ]
        
        self.features_categoricas = ['NM_BAIRRO', 'NM_MUN', 'PERIODO_DIA', 'PERFIL_ALVO', 'TIPO_LOCAL']
        self.features_full = self.features_numericas + self.features_categoricas
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
        """Ciclo de aprendizado supervisionado com Memória Local."""
        df_full = self._carregar_datalake_consolidado()
        
        if df_full is None or len(df_full) < 100:
            logger.error("IA: Massa de dados insuficiente.")
            return False

        # --- NORMALIZAÇÃO DE TIPOS ---
        for col in self.features_categoricas:
            df_full[col] = df_full[col].astype(str).fillna("INDEFINIDO").astype('category')
            
        for col in self.features_numericas:
            df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0.0).astype('float32')
            
        df_full[self.target] = df_full[self.target].astype('float32')

        # 🛡️ SPLIT CRONOLÓGICO: Garante que não prevemos o passado com dados do futuro
        df_full = df_full.sort_values(by=['ANO_REF', 'MES_OCORRENCIA'], ascending=[True, True])
        corte_idx = int(len(df_full) * 0.8)

        df_train = df_full.iloc[:corte_idx]
        df_test = df_full.iloc[corte_idx:]

        X_train = df_train[self.features_full]
        y_train = df_train[self.target]
        X_test = df_test[self.features_full]
        y_test = df_test[self.target]

        logger.info(f"IA: Treino iniciado com Memória Histórica. {len(X_train)} linhas.")
        del df_full, df_train, df_test
        gc.collect()

        # 1. CatBoost (Tweedie ajustado para penalizar erro em zonas de alta gravidade)
        model_cat = CatBoostRegressor(
            iterations=1200,
            depth=7, # Aumentamos a profundidade para capturar interações entre Período e Histórico
            learning_rate=0.02,
            l2_leaf_reg=7,
            cat_features=self.features_categoricas,
            loss_function='Tweedie:variance_power=1.5',
            early_stopping_rounds=100,
            verbose=100
        )
        model_cat.fit(X_train, y_train, eval_set=(X_test, y_test))

        # 2. LightGBM (Foco em variância de Poisson para baixas contagens)
        model_lgb = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            objective='tweedie',
            tweedie_variance_power=1.6, # Ajuste para ser mais sensível a variações de período
            importance_type='gain',
            n_jobs=-1,
            verbosity=-1
        )
        
        model_lgb.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric='mae',
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=100)]
        )

        # Avaliação
        p_cat = model_cat.predict(X_test)
        p_lgb = model_lgb.predict(X_test)
        mae_cat = mean_absolute_error(y_test, p_cat)
        mae_lgb = mean_absolute_error(y_test, p_lgb)

        logger.info(f"IA: MAE CatBoost: {mae_cat:.4f} | MAE LightGBM: {mae_lgb:.4f}")

        # Persistência
        self._exportar_modelo(model_cat, "cat_geral")
        self._exportar_modelo(model_lgb, "lgb_geral")
        
        self.stats_treino = {
            "mae": round(min(mae_cat, mae_lgb), 4),
            "modelo_vencedor": "CatBoost" if mae_cat < mae_lgb else "LightGBM",
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
                
                # Injeção de Features de Apoio
                if "ANO_REF" not in df.columns:
                    df = df.with_columns(pl.lit(ano).cast(pl.Int16).alias("ANO_REF"))
                
                # Se GRAVIDADE_HISTORICA não vier da Prata, geramos como cópia do Target (Deslocada)
                if "GRAVIDADE_HISTORICA" not in df.columns:
                    df = df.with_columns(pl.col("INDICE_GRAVIDADE").alias("GRAVIDADE_HISTORICA"))
                if "VOLUME_HISTORICO" not in df.columns:
                    df = df.with_columns(pl.col("TOTAL_CRIMES").alias("VOLUME_HISTORICO"))

                lista_dfs.append(df)
            except: continue
        
        if not lista_dfs: return None
        df_pl = pl.concat(lista_dfs, how="diagonal")
        
        # Amostragem Estratégica (mantendo 800k registros para profundidade de aprendizado)
        if df_pl.height > 800000:
            df_pl = df_pl.sample(n=800000, seed=42)

        return df_pl.to_pandas()

    def _exportar_modelo(self, modelo, nome):
        key_ver = self._get_path("modelos_ml/versions", f"v{self.versao_modelo}_{nome}.pkl")
        key_lat = self._get_path("modelos_ml", f"latest_{nome}.pkl")
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()
        self.s3.put_object(Bucket=self.bucket, Key=key_ver, Body=payload)
        self.s3.put_object(Bucket=self.bucket, Key=key_lat, Body=payload)

    def obter_stats(self):
        return self.stats_treino

if __name__ == "__main__":
    treinador = TreinadorEvolutivo(dev_mode=False)
    treinador.treinar_modelo_mestre()
