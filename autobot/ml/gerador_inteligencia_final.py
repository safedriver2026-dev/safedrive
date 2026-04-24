import os
import boto3
import polars as pl
import io
import pandas as pd
import shap
import time
import requests
import json
import numpy as np
import unicodedata
import re
from datetime import datetime
from catboost import CatBoostRegressor
from botocore.config import Config

class GeradorDossieSafeDriver:
    def __init__(self):
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        if endpoint.endswith(f"/{self.bucket}"):
            endpoint = endpoint[: -len(f"/{self.bucket}")]
            
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint, 
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4')
        )
        
        self.webhook_url = os.getenv("DISCORD_SUCESSO")
        self.modelo_local = "modelo_safedriver_catboost.cbm"
        self.auditoria = {
            "projeto": "SafeDriver",
            "fase": "Dossiê de Inteligência Geográfica",
            "data_processamento": str(datetime.now()),
            "metricas": {}
        }

    def _normalizar_texto_pl(self, coluna):
        """
        Normalização performática em Polars: Caixa Alta e remoção de espaços.
        Para acentos, usamos uma substituição manual via Regex (mais rápido que map_elements).
        """
        return (
            pl.col(coluna)
            .str.to_uppercase()
            .str.strip_chars()
            # Substituições básicas de acentuação para garantir integridade do JOIN
            .str.replace_all(r"[ÁÀÂÃÄ]", "A")
            .str.replace_all(r"[ÉÈÊË]", "E")
            .str.replace_all(r"[ÍÌÎÏ]", "I")
            .str.replace_all(r"[ÓÒÔÕÖ]", "O")
            .str.replace_all(r"[ÚÙÛÜ]", "U")
            .str.replace_all(r"[Ç]", "C")
            .str.replace_all(r"[^A-Z0-9\s]", "") # Remove qualquer caractere especial restante
        )

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except: pass

    def gerar_dados(self):
        inicio_processo = time.time()
        print("🧠 [DOSSIÊ] Iniciando motor de inteligência...", flush=True)
        
        if not os.path.exists(self.modelo_local):
            print(f"📥 Baixando {self.modelo_local} do bucket...", flush=True)
            self.s3.download_file(self.bucket, f"modelos/{self.modelo_local}", self.modelo_local)
        
        modelo = CatBoostRegressor().load_model(self.modelo_local)
        
        print("📥 Lendo a base Ouro...", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key="datalake/ouro/safedriver_abt_treino.parquet")
        df_ouro = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        total_linhas = df_ouro.height
        
        # =================================================================
        # NORMALIZAÇÃO DE STRINGS (CIDADE E BAIRRO)
        # =================================================================
        print("🧹 Normalizando strings para integridade do Data Warehouse...", flush=True)
        df_ouro = df_ouro.with_columns([
            self._normalizar_texto_pl("CIDADE").alias("CIDADE"),
            self._normalizar_texto_pl("BAIRRO").alias("BAIRRO")
        ])

        print("⚙️ Recriando Feature Cross (Sazonalidade x Perfil)...", flush=True)
        df_ouro = df_ouro.with_columns(
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        )
        
        print("⚡ Rodando predição massiva...", flush=True)
        X_all = df_ouro.select(modelo.feature_names_).to_pandas()
        for col in X_all.select_dtypes(['object', 'category']).columns: 
            X_all[col] = X_all[col].astype(str)
            
        preds = modelo.predict(X_all)
        df_dossie = df_ouro.with_columns(
            pl.Series("RISCO_PREDITO_IA", preds).round(2)
        )

        print("🧬 Analisando DNA criminal (SHAP) por Bairro...", flush=True)
        df_shap_sample = df_ouro.sample(n=min(50000, df_ouro.height), seed=42)
        X_shap = df_shap_sample.select(modelo.feature_names_).to_pandas()
        for col in X_shap.select_dtypes(['object', 'category']).columns: 
            X_shap[col] = X_shap[col].astype(str)
        
        explainer = shap.TreeExplainer(modelo)
        shap_vals = explainer.shap_values(X_shap)
        
        df_shap_geo = pd.concat([
            df_shap_sample.select(["CIDADE", "BAIRRO"]).to_pandas(),
            pd.DataFrame(shap_vals, columns=[f"SHAP_{f}" for f in modelo.feature_names_])
        ], axis=1).groupby(["CIDADE", "BAIRRO"]).mean().reset_index()

        print("📦 Sincronizando resultados com o R2...", flush=True)
        buf_eventos = io.BytesIO()
        df_dossie.write_parquet(buf_eventos, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dossie_eventos.parquet", Body=buf_eventos.getvalue())
        
        buf_shap = io.BytesIO()
        pl.from_pandas(df_shap_geo).write_parquet(buf_shap, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dim_shap.parquet", Body=buf_shap.getvalue())

        # Finalização e Relatório
        duracao = time.time() - inicio_processo
        media_risco = float(np.mean(preds))
        max_risco = float(np.max(preds))
        
        self.auditoria["metricas"] = {
            "total_processado": total_linhas,
            "media_risco_predito": round(media_risco, 4),
            "max_risco_detectado": round(max_risco, 4),
            "bairros_analisados": len(df_shap_geo),
            "tempo_execucao_s": round(duracao, 2)
        }

        buf_log = io.BytesIO(json.dumps(self.auditoria, indent=4).encode())
        self.s3.put_object(Bucket=self.bucket, Key="modelos/AUDITORIA_DOSSIE_INTELIGENCIA.json", Body=buf_log.getvalue())

        report = (
            f"==============================================================\n"
            f" 🛡️ RELATÓRIO DE INTELIGÊNCIA - SAFEDRIVER \n"
            f"==============================================================\n"
            f"Dossiê normalizado com sucesso. Status: Pronto para BigQuery.\n"
            f"==============================================================\n"
        )
        print(report)
        self._notificar_discord(f"```text\n{report}\n```")

if __name__ == "__main__":
    GeradorDossieSafeDriver().gerar_dados()
