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
    Motor de Treinamento Anti-Diluição.
    Totalmente sincronizado com a Camada Ouro (ArquitetoSafeDriverOuro).
    """
    def __init__(self):
        # 1. IDENTIDADE SINCRONIZADA: Busca o mesmo Secret usado na Ouro
        self.projeto = os.getenv("NOME_PROJETO", "safedriver").strip().lower()
        
        # 2. INFRAESTRUTURA R2
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint,
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4', retries={'max_attempts': 5})
        )
        
        # 3. PARÂMETROS DE MODELAGEM
        self.modelo_nome = "modelo_safedriver_catboost.cbm"
        self.log_nome = "AUDITORIA_TREINAMENTO_MODELO.json"
        self.target = "LABEL_PESO_RISCO"
        
        # Features categóricas que a Ouro já normalizou em UPPERCASE
        self.cat_features = [
            "H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", 
            "FEAT_PERFIL_VITIMA", "FEAT_CONTEXTO_CRITICO", "FEAT_TIPO_DIA"
        ]
        
        self.telemetria = {
            "projeto": "SafeDriver",
            "instancia_ouro": self.projeto,
            "timestamp": str(datetime.now()),
            "metricas_final": {},
            "importancia_features": {}
        }

    def _sincronizar_r2(self, local_path, s3_key):
        print(f"☁️ Sincronizando artefato com o Data Lake: {s3_key}...", flush=True)
        self.s3.upload_file(local_path, self.bucket, s3_key)

    def carregar_dados(self):
        """Lê a ABT que a Camada Ouro acabou de salvar."""
        key_ouro = f"datalake/ouro/{self.projeto}_abt_treino.parquet"
        print(f"📥 Extraindo ABT Ouro Equalizada: {key_ouro}", flush=True)
        
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key_ouro)
            # Usamos Polars para leitura rápida, convertendo apenas o necessário para o CatBoost
            df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
            
            # Auditoria de Densidade: Fundamental para monitorar a 'honestidade' do risco
            zeros = df.filter(pl.col(self.target) == 0).height
            total = df.height
            self.telemetria["distribuicao_target"] = {
                "total_registros": total,
                "percentual_zero": round((zeros / total) * 100, 2),
                "media_risco": round(df[self.target].mean(), 4)
            }
            return df
        except Exception as e:
            print(f"❌ ERRO: Arquivo {key_ouro} não encontrado. Verifique o Job da Ouro.")
            raise e

    def treinar(self):
        inicio_geral = time.time()
        df = self.carregar_dados()
        df_pd = df.to_pandas()

        # SELEÇÃO DE FEATURES: 
        # Removemos o target e metadados que a Ouro deixou (DATAOCORRENCIA, ANO_JOIN, etc)
        colunas_excluir = [self.target, "DATAOCORRENCIA", "ANO_JOIN", "CIDADE", "BAIRRO", "BAIRRO_right"]
        features = [col for col in df_pd.columns if col not in colunas_excluir]
        
        X = df_pd[features]
        y = df_pd[self.target]
        
        # PESOS (O 'Soco' de 10x): 
        # Como a Ouro usa dosimetria (10, 5, 2), aqui amplificamos o sinal para evitar diluição.
        pesos = np.where(y > 0, 10.0, 1.0)

        # Tratamento preventivo de Nulos em Categorias (Blindagem extra)
        for col in self.cat_features:
            if col in X.columns:
                X[col] = X[col].fillna("DESCONHECIDO").astype(str)

        print(f"🚀 Iniciando CatBoost Tweedie (Poder de Variância: 1.6)...", flush=True)
        train_pool = Pool(X, y, cat_features=self.cat_features, weight=pesos)

        # Hiperparâmetros ajustados para Micro-Bairros (H3)
        params = {
            'iterations': 3000,
            'learning_rate': 0.02,
            'depth': 8,
            'l2_leaf_reg': 2.0,      # Regularização moderada para evitar overfitting no H3
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
        
        # --- PÓS-TREINAMENTO: MÉTRICAS E IMPORTÂNCIA ---
        importancias = modelo.get_feature_importance()
        self.telemetria["importancia_features"] = {f: round(i, 2) for f, i in sorted(zip(features, importancias), key=lambda x: x[1], reverse=True)}
        
        self.telemetria["metricas_final"] = {
            "tempo_treinamento_seg": duracao,
            "mae_final": modelo.get_best_score().get('learn', {}).get('MAE', 0)
        }

        # Salvamento e Upload
        modelo.save_model(self.modelo_nome)
        with open(self.log_nome, 'w', encoding='utf-8') as f:
            json.dump(self.telemetria, f, indent=4)

        self._sincronizar_r2(self.modelo_nome, f"modelos/{self.modelo_nome}")
        self._sincronizar_r2(self.log_nome, f"modelos/{self.log_nome}")

        print(f"\n🏆 TREINAMENTO EQUALIZADO CONCLUÍDO EM {duracao}s")
        print(f"📊 Features Dominantes: {list(self.telemetria['importancia_features'].keys())[:3]}")

if __name__ == "__main__":
    TreinadorSafeDriver().treinar()
