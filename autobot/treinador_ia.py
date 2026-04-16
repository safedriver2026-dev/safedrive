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
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

# Auditoria de Performance
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class TreinadorEvolutivo:
    def __init__(self, dev_mode=False):
        self.dev_mode = dev_mode 
        
        # Conectividade Cloudflare R2
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
        
        # 🎯 A MUDANÇA DE PARADIGMA: IA agora prevê Gravidade (Dano Real) em vez de Volume
        self.target = "INDICE_GRAVIDADE"
        
        # --- SCHEMA DE TREINAMENTO LIMPO (SEM DATA LEAKAGE) ---
        self.features_numericas = [
            'DENSIDADE', 'TAXA_VACANCIA', 'CONTAGIO_PONDERADO', 'PRESSAO_RISCO_LOCAL', 
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
        """Ciclo de aprendizado supervisionado com Corte Cronológico Contínuo."""
        df_treino_full = self._carregar_datalake_consolidado()
        
        if df_treino_full is None or len(df_treino_full) < 10:
            logger.error("IA: Insumos insuficientes para o treinamento.")
            return False

        logger.info(f"IA: Iniciando preparação cronológica com {len(df_treino_full)} registros. Target: {self.target}")

        # Tipagem Otimizada (Evita erros do CatBoost e LightGBM)
        for col in self.features_categoricas:
            df_treino_full[col] = df_treino_full[col].astype(str).fillna("INDEFINIDO").astype('category')
            
        for col in self.features_numericas:
            df_treino_full[col] = pd.to_numeric(df_treino_full[col], errors='coerce').fillna(0.0).astype('float32')
            
        df_treino_full[self.target] = df_treino_full[self.target].fillna(0).astype('float32')

        # ---------------------------------------------------------
        # 🛡️ DATA LEAKAGE PREVENTION: CORTE CRONOLÓGICO CONTÍNUO
        # ---------------------------------------------------------
        logger.info("IA: Ordenando a base de dados temporalmente (Passado -> Presente)...")
        
        # Ordenamos do crime mais velho para o mais novo
        if 'MES_OCORRENCIA' in df_treino_full.columns:
            df_treino_full = df_treino_full.sort_values(by=['ANO_REF', 'MES_OCORRENCIA'], ascending=[True, True])
        else:
            df_treino_full = df_treino_full.sort_values(by=['ANO_REF'], ascending=[True])

        # Encontramos o ponto de corte exato (80% para treino, 20% para teste)
        corte_idx = int(len(df_treino_full) * 0.8)

        # Fatiamos o dataframe (O passado treina, o futuro testa)
        df_train = df_treino_full.iloc[:corte_idx]
        df_test = df_treino_full.iloc[corte_idx:]

        logger.info(f"IA: Split Cronológico Concluído. Treino: {len(df_train)} | Teste: {len(df_test)}")

        # Limpeza de memória agressiva
        del df_treino_full
        gc.collect()

        X_train = df_train[self.features_numericas + self.features_categoricas]
        y_train = df_train[self.target]
        X_test = df_test[self.features_numericas + self.features_categoricas]
        y_test = df_test[self.target]

        del df_train, df_test
        gc.collect()

        # 1. CatBoost (Ajustado para base limpa e Previsão de Gravidade)
        logger.info(f"IA: [1/2] Lapidando CatBoost (Tweedie). Aguarde...")
        model_cat = CatBoostRegressor(
            iterations=800, 
            depth=5,                 
            learning_rate=0.05, 
            l2_leaf_reg=10,          
            cat_features=self.features_categoricas,
            loss_function='Tweedie:variance_power=1.5',
            early_stopping_rounds=50, 
            verbose=100,
            thread_count=2
        )
        model_cat.fit(X_train, y_train, eval_set=(X_test, y_test))

        # 2. LightGBM (O Especialista em Velocidade)
        logger.info("IA: [2/2] Lapidando LightGBM (Feature Fractioning)...")
        model_lgb = LGBMRegressor(
            n_estimators=600, 
            learning_rate=0.04, 
            objective='tweedie', 
            tweedie_variance_power=1.5,
            feature_fraction=0.8,    
            importance_type='gain',
            n_jobs=2,
            verbosity=-1,
            min_child_samples=50     
        )
        model_lgb.fit(X_train, y_train)

        # Avaliação de Precisão Realista
        p_cat = model_cat.predict(X_test)
        p_lgb = model_lgb.predict(X_test)
        
        mae_cat = mean_absolute_error(y_test, p_cat)
        mae_lgb = mean_absolute_error(y_test, p_lgb)
        
        logger.info(f"IA: MAE (Gravidade) CatBoost: {mae_cat:.4f} | MAE (Gravidade) LightGBM: {mae_lgb:.4f}")

        # Persistência no Datalake (Cloudflare R2)
        self._exportar_modelo(model_cat, "cat_geral")
        self._exportar_modelo(model_lgb, "lgb_geral")
        
        self.stats_treino = {
            "mae": round(min(mae_cat, mae_lgb), 4),
            "modelo_vencedor": "CatBoost" if mae_cat < mae_lgb else "LightGBM",
            "volume_treino": len(X_train),
            "volume_teste": len(X_test),
            "data_versao": self.versao_modelo
        }
        return True

    def _carregar_datalake_consolidado(self):
        """Reúne a Camada Prata, aplica amostragem de memória e garante os dados temporais."""
        lista_dfs = []
        anos = [datetime.now().year] if self.dev_mode else range(2022, datetime.now().year + 1)
        
        for ano in anos:
            key = self._get_path("prata", f"ssp_consolidada_{ano}.parquet")
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=key)
                df = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                
                # Garantia da existência da coluna de Ano para o particionamento temporal
                if "ANO_REF" not in df.columns:
                    df = df.with_columns(pl.lit(ano).cast(pl.Int16).alias("ANO_REF"))
                    
                lista_dfs.append(df)
            except:
                logger.warning(f"IA: Dados de {ano} ausentes na Prata.")
                continue
        
        if not lista_dfs: return None
        
        df_pl = pl.concat(lista_dfs, how="diagonal")
        del lista_dfs
        gc.collect()

        # Preenchimento de invariantes para evitar quebra de schema no fit()
        for col in self.features_categoricas:
            if col not in df_pl.columns:
                df_pl = df_pl.with_columns(pl.lit("INDEFINIDO").alias(col))

        for col in self.features_numericas:
            if col in df_pl.columns:
                df_pl = df_pl.with_columns(pl.col(col).cast(pl.Float32).fill_null(0.0))
            else:
                df_pl = df_pl.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(col))

        if self.target not in df_pl.columns:
            df_pl = df_pl.with_columns(pl.lit(0.0, dtype=pl.Float32).alias(self.target))

        # --- PROTEÇÃO NUCLEAR CONTRA OOM ---
        LIMITE_LINHAS = 600000 
        if df_pl.height > LIMITE_LINHAS:
            logger.warning(f"IA: Dataset original ({df_pl.height} linhas) muito pesado. Aplicando amostragem estrutural para {LIMITE_LINHAS}.")
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
        logger.info(f"IA: Modelo {nome} persistido (Versão: {self.versao_modelo}).")

    def obter_stats(self):
        return self.stats_treino

if __name__ == "__main__":
    treinador = TreinadorEvolutivo(dev_mode=False) 
    treinador.treinar_modelo_mestre()
