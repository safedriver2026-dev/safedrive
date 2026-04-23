import os
import boto3
import polars as pl
import io
import pandas as pd
import shap
from catboost import CatBoostRegressor
from botocore.config import Config

class GeradorDossieSafeDriver:
    def __init__(self):
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # --- CORREÇÃO: Limpeza do Endpoint R2 idêntica à do Treinador ---
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        if endpoint.endswith(f"/{self.bucket}"):
            endpoint = endpoint[: -len(f"/{self.bucket}")]
            
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint, 
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4')
        )
        self.modelo_local = "modelo_safedriver_catboost.cbm"

    def gerar_dados(self):
        print("🧠 [DOSSIÊ MÓDULO 1] Baixando IA e lendo a base Ouro na íntegra...")
        
        # Faz o download do modelo treinado do R2
        if not os.path.exists(self.modelo_local):
            print(f"📥 Baixando {self.modelo_local} do bucket...")
            self.s3.download_file(self.bucket, f"modelos/{self.modelo_local}", self.modelo_local)
        
        modelo = CatBoostRegressor().load_model(self.modelo_local)
        
        print("📥 Lendo a ABT Ouro...")
        obj = self.s3.get_object(Bucket=self.bucket, Key="datalake/ouro/safedriver_abt_treino.parquet")
        df_ouro = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # =================================================================
        # 1. PREDICAÇÃO MASSIVA (Linha a Linha)
        # =================================================================
        print("⚡ Rodando o modelo em todos os registros reais...")
        X_all = df_ouro.select(modelo.feature_names_).to_pandas()
        
        for col in X_all.select_dtypes(['object', 'category']).columns: 
            X_all[col] = X_all[col].astype(str)
            
        df_dossie = df_ouro.with_columns(
            pl.Series("RISCO_PREDITO_IA", modelo.predict(X_all)).round(2)
        )

        # =================================================================
        # 2. DNA DE RISCO (SHAP POR BAIRRO)
        # =================================================================
        print("🧬 Extraindo o DNA criminal (SHAP) por Bairro...")
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

        # =================================================================
        # 3. EXPORTAÇÃO PARA O R2
        # =================================================================
        print("📦 Salvando o Dossiê completo no Cloudflare R2...")
        
        buf_eventos = io.BytesIO()
        df_dossie.write_parquet(buf_eventos, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dossie_eventos.parquet", Body=buf_eventos.getvalue())
        
        buf_shap = io.BytesIO()
        pl.from_pandas(df_shap_geo).write_parquet(buf_shap, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dim_shap.parquet", Body=buf_shap.getvalue())
        
        print("✅ [DOSSIÊ MÓDULO 1 CONCLUÍDO] Base bruta com inteligência gerada!")

if __name__ == "__main__":
    GeradorDossieSafeDriver().gerar_dados()
