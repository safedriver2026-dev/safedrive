import os
import io
import json
import boto3
import polars as pl
import pandas as pd
import numpy as np
import time
import warnings
from datetime import datetime
from catboost import CatBoostRegressor, Pool
from botocore.config import Config

warnings.filterwarnings("ignore", category=FutureWarning)

class TreinadorSafeDriver:
    """
    Motor de Treinamento Anti-Diluição com Auditoria Detalhada.
    Implementa Tweedie Regression para dados zero-inflados (muitos locais sem crime).
    
    A função de perda Tweedie é definida por:
    $$P(y| \mu, \phi, p) = \exp \left( \frac{y \cdot \frac{\mu^{1-p}}{1-p} - \frac{\mu^{2-p}}{2-p}}{\phi} + a(y, \phi, p) \right)$$
    """
    def __init__(self):
        # 1. SINCRONIA: Lê o nome do projeto para bater com a Camada Ouro
        self.projeto = os.getenv("NOME_PROJETO", "safedriver").strip().lower()
        
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint,
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4', retries={'max_attempts': 5})
        )
        
        self.modelo_nome = "modelo_safedriver_catboost.cbm"
        self.log_nome = "AUDITORIA_TREINAMENTO_MODELO.json"
        self.target = "LABEL_PESO_RISCO"
        self.cat_features = [
            "H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", 
            "FEAT_PERFIL_VITIMA", "FEAT_CONTEXTO_CRITICO", "FEAT_TIPO_DIA"
        ]
        self.telemetria = {
            "projeto": "SafeDriver",
            "id_instancia": self.projeto,
            "timestamp": str(datetime.now()),
            "metricas_final": {},
            "importancia_features": {}
        }

    def _sincronizar_r2(self, local_path, s3_key):
        print(f"☁️ Enviando {local_path} para o Data Lake...", flush=True)
        self.s3.upload_file(local_path, self.bucket, s3_key)

    def carregar_dados(self):
        # CONSTRUÇÃO DA CHAVE: Agora idêntica à que a Camada Ouro usa
        key_ouro = f"datalake/ouro/{self.projeto}_abt_treino.parquet"
        print(f"📥 Extraindo ABT Ouro: {key_ouro}", flush=True)
        
        obj = self.s3.get_object(Bucket=self.bucket, Key=key_ouro)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        zeros = df.filter(pl.col(self.target) == 0).height
        total = df.height
        self.telemetria["distribuicao_target"] = {
            "total_registros": total,
            "registros_zero": zeros,
            "percentual_zero": round((zeros / total) * 100, 2),
            "media_target": round(df[self.target].mean(), 4)
        }
        return df

    def treinar(self):
        inicio_geral = time.time()
        df = self.carregar_dados()
        df_pd = df.to_pandas()

        features = [col for col in df_pd.columns if col not in [self.target, "DATAOCORRENCIA", "ANO_JOIN"]]
        X = df_pd[features]
        y = df_pd[self.target]
        
        # PESOS: Amplifica o sinal de crimes reais para o modelo não ficar "preguiçoso"
        pesos = np.where(y > 0, 10.0, 1.0)

        # Saneamento categórico
        for col in self.cat_features:
            if col in X.columns:
                X[col] = X[col].fillna("DESCONHECIDO").astype(str).str.replace(r'\.0$', '', regex=True)

        print(f"🚀 Iniciando CatBoost Tweedie (Project: {self.projeto})...", flush=True)
        train_pool = Pool(X, y, cat_features=self.cat_features, weight=pesos)

        params = {
            'iterations': 3000,
            'learning_rate': 0.02,
            'depth': 8,
            'l2_leaf_reg': 1.5,      # Regularização baixa para permitir picos de risco
            'loss_function': 'Tweedie:variance_power=1.6', 
            'eval_metric': 'MAE',
            'random_seed': 42,
            'verbose': 250,
            'task_type': 'CPU',
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.85
        }

        modelo = CatBoostRegressor(**params)
        modelo.fit(train_pool)

        duracao = round(time.time() - inicio_geral, 2)
        
        # Extração de Inteligência
        importancias = modelo.get_feature_importance()
        feat_imp = sorted(zip(features, importancias), key=lambda x: x[1], reverse=True)
        self.telemetria["importancia_features"] = {f: round(i, 2) for f, i in feat_imp}

        self.telemetria["metricas_final"] = {
            "tempo_treinamento_seg": duracao,
            "best_loss": modelo.get_best_score().get('learn', {}).get('Tweedie', 0),
            "mae_final": modelo.get_best_score().get('learn', {}).get('MAE', 0)
        }

        # Salvamento e Sincronização
        modelo.save_model(self.modelo_nome)
        with open(self.log_nome, 'w', encoding='utf-8') as f:
            json.dump(self.telemetria, f, indent=4, ensure_ascii=False)

        self._sincronizar_r2(self.modelo_nome, f"modelos/{self.modelo_nome}")
        self._sincronizar_r2(self.log_nome, f"modelos/{self.log_nome}")

        print(f"\n🏆 TREINAMENTO CONCLUÍDO EM {duracao}s")
        print(f"📊 Top 3 Features: {list(self.telemetria['importancia_features'].keys())[:3]}")

if __name__ == "__main__":
    TreinadorSafeDriver().treinar()
