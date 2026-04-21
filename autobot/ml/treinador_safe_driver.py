import os
import boto3
import polars as pl
import io
import pandas as pd
import requests
import time
import shap
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, d2_tweedie_score
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
        self.caminho_abt = "datalake/ouro/safedriver_abt_eventos.parquet"
        self.modelo_local = "modelo_safedriver_catboost.cbm"

    def _notificar_discord(self, msg):
        if self.webhook_url:
            requests.post(self.webhook_url, json={"content": msg})

    def executar_treino(self):
        inicio_processo = time.time()
        
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.caminho_abt)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        colunas_agrupamento = ["H3_INDEX", "SAZON_DIA_SEMANA", "SAZON_HORA", "SAZON_MES", "SAZON_PONTO_FACULTATIVO"]
        cols_meta = [c for c in df.columns if c.startswith("META_") or c.startswith("INFRA_DIV_") or c == "MICRO_POPULACAO_H3"]
        
        df_treino = df.group_by(colunas_agrupamento).agg([
            pl.len().alias("TARGET_CONTAGEM"),
            pl.col("META_PESO_GRAVIDADE").mean().alias("PESO_AMOSTRA"),
            *[pl.col(c).mean() for c in cols_meta]
        ])

        pdf = df_treino.to_pandas()
        cat_features = ["H3_INDEX", "SAZON_DIA_SEMANA", "SAZON_HORA", "SAZON_MES", "SAZON_PONTO_FACULTATIVO"]
        for col in cat_features:
            pdf[col] = pdf[col].astype(str)

        X = pdf.drop(["TARGET_CONTAGEM", "PESO_AMOSTRA"], axis=1)
        y = pdf["TARGET_CONTAGEM"]
        w = pdf["PESO_AMOSTRA"]

        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, w, test_size=0.2, random_state=42)

        train_pool = Pool(X_train, y_train, weight=w_train, cat_features=cat_features)
        test_pool = Pool(X_test, y_test, weight=w_test, cat_features=cat_features)

        modelo = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            loss_function='Tweedie:variance_power=1.5',
            eval_metric='Tweedie',
            early_stopping_rounds=50,
            random_seed=42,
            verbose=False
        )

        modelo.fit(train_pool, eval_set=test_pool, use_best_model=True)
        duracao = time.time() - inicio_processo

        # --- MOTOR DE EXPLICABILIDADE (SHAP) ---
        print("Calculando valores SHAP para interpretabilidade...", flush=True)
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X_test)
        
        # Calculamos o impacto médio de cada feature (Global SHAP)
        shap_importance = pd.DataFrame({
            'feature': X.columns,
            'impacto_medio': shap.np.abs(shap_values).mean(0)
        }).sort_values(by='impacto_medio', ascending=False)

        top_shap = "\n".join([f"• {row['feature']}: {row['impacto_medio']:.4f}" for _, row in shap_importance.head(5).iterrows()])

        # Métricas
        y_pred = modelo.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        tweedie_d2 = d2_tweedie_score(y_test, y_pred, power=1.5)

        report_msg = (
            f"🧠 **[SafeDriver] IA Explicável (XAI) Ativada**\n"
            f"```ml\n"
            f"MÉTRICAS:\n"
            f"• Tweedie D² Score: {tweedie_d2:.4f}\n"
            f"• MAE: {mae:.4f}\n"
            f"-------------------------------------\n"
            f"CONTRIBUIÇÃO SHAP (O que define o risco):\n"
            f"{top_shap}\n"
            f"-------------------------------------\n"
            f"Tempo: {duracao:.2f}s | Registros: {len(pdf)}\n"
            f"```"
        )

        print(report_msg)
        self._notificar_discord(report_msg)

        modelo.save_model(self.modelo_local)
        with open(self.modelo_local, "rb") as f:
            self.s3.put_object(Bucket=self.bucket, Key=f"modelos/{self.modelo_local}", Body=f.read())

if __name__ == "__main__":
    trainer = TreinadorSafeDriver()
    trainer.executar_treino()
