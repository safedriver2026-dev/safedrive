import os
import boto3
import polars as pl
import io
import pandas as pd
import requests
import time
import shap
import json
import numpy as np
import shutil
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from botocore.config import Config
from datetime import datetime

class TreinadorSafeDriver:
    def __init__(self):
        # Configurações de conexão R2
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        if endpoint.endswith(f"/{self.bucket}"):
            endpoint = endpoint[: -len(f"/{self.bucket}")]
            
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint,
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4', retries={'max_attempts': 3})
        )

        self.webhook_url = os.getenv("DISCORD_SUCESSO")
        self.caminho_abt = "datalake/ouro/safedriver_abt_treino.parquet"
        self.modelo_local = "modelo_safedriver_catboost.cbm"
        
        self.auditoria = {
            "fase": "Treinamento de Modelo",
            "data": str(datetime.now()),
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except: pass

    def executar_treino(self):
        inicio_processo = time.time()
        print(f"🚀 [TREINO] Iniciando carregamento da ABT Ouro...", flush=True)
        
        # 1. CARREGAMENTO DOS DADOS (Streaming do R2 para Memória)
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.caminho_abt)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # 2. SELEÇÃO INTELIGENTE DE FEATURES
        # Pegamos todas as features criadas na Ouro, ignorando IDs ou nomes de ruas
        cols_features = [c for c in df.columns if any(c.startswith(pre) for pre in ["FEAT_", "INFRA_", "CENSO_", "MICRO_", "SAZON_"])]
        # H3_INDEX é nossa categoria espacial de alta cardinalidade
        cols_features.append("H3_INDEX")
        
        target = "LABEL_PESO_RISCO"
        
        print(f"📊 Dataset: {df.height} linhas | {len(cols_features)} features.")

        # 3. PREPARAÇÃO (Conversão para Pandas apenas no final para o CatBoost Pool)
        pdf = df.select(cols_features + [target]).to_pandas()
        
        # Definimos colunas categóricas para o CatBoost não tratar como números
        cat_features = ["H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES"]
        cat_features = [c for c in cat_features if c in pdf.columns]
        
        for col in cat_features:
            pdf[col] = pdf[col].astype(str)

        # 4. SPLIT TREINO/TESTE (85/15)
        X = pdf[cols_features]
        y = pdf[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        # 5. CONFIGURAÇÃO E TREINO
        print(f"🧠 Treinando CatBoost (Iterações: 1500)...", flush=True)
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        test_pool = Pool(X_test, y_test, cat_features=cat_features)

        modelo = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=8,
            l2_leaf_reg=3,
            loss_function='RMSE', # Foco em penalizar erros em crimes severos
            eval_metric='MAE',
            od_type='Iter', # Early stopping
            od_wait=50,
            random_seed=42,
            verbose=100
        )

        modelo.fit(train_pool, eval_set=test_pool, use_best_model=True)
        
        # 6. EXPLICABILIDADE SHAP (Por que o modelo dá risco alto?)
        print("🔍 Calculando explicabilidade SHAP (Amostra de 5k)...", flush=True)
        explainer = shap.TreeExplainer(modelo)
        sample_test = X_test.sample(min(5000, len(X_test)))
        shap_values = explainer.shap_values(sample_test)
        
        shap_importance = pd.DataFrame({
            'feature': X.columns,
            'impacto_medio': np.abs(shap_values).mean(0)
        }).sort_values(by='impacto_medio', ascending=False)

        top_5 = shap_importance.head(5)
        resumo_shap = "\n".join([f"• {r['feature']}: {r['impacto_medio']:.4f}" for _, r in top_5.iterrows()])

        # 7. PERFORMANCE FINAL
        y_pred = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        duracao = time.time() - inicio_processo

        # 8. SALVAMENTO E UPLOAD
        print("💾 Salvando modelo e subindo para o R2...", flush=True)
        modelo.save_model(self.modelo_local)
        
        # Modelo Binário
        with open(self.modelo_local, "rb") as f:
            self.s3.put_object(Bucket=self.bucket, Key=f"modelos/{self.modelo_local}", Body=f.read())
        
        # JSON de Explicabilidade para o Front-end
        shap_json = json.dumps(shap_importance.to_dict(orient='records'), indent=4)
        self.s3.put_object(Bucket=self.bucket, Key="modelos/SHAP_IMPORTANCE.json", Body=shap_json.encode())

        # 9. NOTIFICAÇÃO DISCORD
        report = (
            f"🤖 **[SafeDriver] Modelo de IA Atualizado**\n"
            f"```ml\n"
            f"MÉTRICAS:\n"
            f"• R² Score: {r2:.4f}\n"
            f"• Erro Médio (MAE): {mae:.4f}\n"
            f"-------------------------------------\n"
            f"DRIVERS DE RISCO (SHAP TOP 5):\n"
            f"{resumo_shap}\n"
            f"-------------------------------------\n"
            f"Tempo: {duracao/60:.2f} min | ABT: {len(pdf)} linhas\n"
            f"Status: DEPLOY PRONTO NO R2\n"
            f"```"
        )
        print(report)
        self._notificar_discord(report)
        
        # Cleanup Local
        if os.path.exists(self.modelo_local):
            os.remove(self.modelo_local)

if __name__ == "__main__":
    trainer = TreinadorSafeDriver()
    trainer.executar_treino()
