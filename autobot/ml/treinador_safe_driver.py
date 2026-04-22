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
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from botocore.config import Config

class TreinadorSafeDriver:
    def __init__(self):
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

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg})
            except: pass

    def executar_treino(self):
        inicio_processo = time.time()
        print(f"🚀 Iniciando Treino com a ABT Ouro: {self.caminho_abt}", flush=True)
        
        # 1. CARREGAMENTO DOS DADOS
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.caminho_abt)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # 2. SELEÇÃO DE FEATURES (Automática baseada nos prefixos da Ouro)
        # Selecionamos tudo o que é variável preditiva
        cols_features = [c for c in df.columns if any(c.startswith(pre) for pre in ["FEAT_", "INFRA_", "CENSO_", "MICRO_", "SAZON_"])]
        # Adicionamos o H3 como categoria principal
        cols_features.append("H3_INDEX")
        
        target = "LABEL_PESO_RISCO"
        
        print(f"📊 Features identificadas: {len(cols_features)} | Registros: {df.height}")

        # 3. PREPARAÇÃO PARA PANDAS (CatBoost prefere para o Pool de categorias)
        pdf = df.select(cols_features + [target]).to_pandas()
        
        # Definimos quais colunas são categóricas
        cat_features = ["H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES"]
        cat_features = [c for c in cat_features if c in pdf.columns]
        
        for col in cat_features:
            pdf[col] = pdf[col].astype(str)

        # 4. SPLIT TREINO/TESTE
        X = pdf[cols_features]
        y = pdf[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

        # 5. TREINAMENTO (Otimizado para Regressão de Severidade)
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        test_pool = Pool(X_test, y_test, cat_features=cat_features)

        modelo = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=8,
            loss_function='RMSE', # Root Mean Squared Error para penalizar grandes erros de gravidade
            od_type='Iter',
            od_wait=50,
            random_seed=42,
            verbose=100
        )

        print("🧠 Treinando motor CatBoost...")
        modelo.fit(train_pool, eval_set=test_pool, use_best_model=True)
        
        duracao = time.time() - inicio_processo

        # 6. EXPLICABILIDADE (SHAP) - O que está a causar o crime?
        print("🔍 Calculando interpretação SHAP (Amostra de 5k)...", flush=True)
        # Usamos uma amostra para o SHAP não demorar horas no GitHub Actions
        explainer = shap.TreeExplainer(modelo)
        sample_test = X_test.sample(min(5000, len(X_test)))
        shap_values = explainer.shap_values(sample_test)
        
        # Impacto Médio Global
        shap_importance = pd.DataFrame({
            'feature': X.columns,
            'impacto_medio': np.abs(shap_values).mean(0)
        }).sort_values(by='impacto_medio', ascending=False)

        top_5_features = shap_importance.head(5)
        resumo_shap = "\n".join([f"• {row['feature']}: {row['impacto_medio']:.4f}" for _, row in top_5_features.iterrows()])

        # 7. MÉTRICAS DE PERFORMANCE
        y_pred = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # 8. RELATÓRIO E NOTIFICAÇÃO
        report_msg = (
            f"🤖 **[SafeDriver] Modelo de Risco Treinado**\n"
            f"```ml\n"
            f"DESEMPENHO:\n"
            f"• R² Score: {r2:.4f}\n"
            f"• Erro Médio (MAE): {mae:.4f}\n"
            f"-------------------------------------\n"
            f"DRIVERS DE RISCO (SHAP):\n"
            f"{resumo_shap}\n"
            f"-------------------------------------\n"
            f"Tempo: {duracao:.2f}s | Linhas: {len(pdf)}\n"
            f"Status: MODELO ATUALIZADO NO R2\n"
            f"```"
        )

        print(report_msg)
        self._notificar_discord(report_msg)

        # 9. SALVAMENTO E UPLOAD
        modelo.save_model(self.modelo_local)
        with open(self.modelo_local, "rb") as f:
            self.s3.put_object(Bucket=self.bucket, Key=f"modelos/{self.modelo_local}", Body=f.read())
        
        # Salva também a importância SHAP como JSON para o dashboard
        shap_dict = shap_importance.to_dict(orient='records')
        self.s3.put_object(
            Bucket=self.bucket, 
            Key="modelos/explicabilidade_shap.json", 
            Body=json.dumps(shap_dict, indent=4).encode()
        )

if __name__ == "__main__":
    trainer = TreinadorSafeDriver()
    trainer.executar_treino()
