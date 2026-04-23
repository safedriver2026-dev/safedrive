import os
import boto3
import polars as pl
import io
import pandas as pd
import shap
from catboost import CatBoostRegressor
from botocore.config import Config

class GeradorInteligenciaSafeDriver:
    def __init__(self):
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint, 
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4')
        )
        self.modelo_local = "modelo_safedriver_catboost.cbm"

    def gerar_dados(self):
        print("🧠 [MÓDULO 1] Carregando modelo e dados estruturados...")
        modelo = CatBoostRegressor().load_model(self.modelo_local)
        
        obj = self.s3.get_object(Bucket=self.bucket, Key="datalake/ouro/safedriver_abt_treino.parquet")
        df_ouro = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # =================================================================
        # 1. EXTRAÇÃO DE DNA DE RISCO (SHAP POR BAIRRO)
        # =================================================================
        print("🧬 Analisando DNA de risco via valores SHAP...")
        # Amostragem para não estourar a RAM no cálculo de SHAP
        df_shap_sample = df_ouro.sample(n=min(50000, df_ouro.height), seed=42)
        X_shap = df_shap_sample.select(modelo.feature_names_).to_pandas()
        
        # Cast de variáveis categóricas
        for col in X_shap.select_dtypes(['object', 'category']).columns: 
            X_shap[col] = X_shap[col].astype(str)
        
        explainer = shap.TreeExplainer(modelo)
        shap_vals = explainer.shap_values(X_shap)
        
        df_shap_geo = pd.concat([
            df_shap_sample.select(["CIDADE", "BAIRRO"]).to_pandas(),
            pd.DataFrame(shap_vals, columns=[f"SHAP_{f}" for f in modelo.feature_names_])
        ], axis=1).groupby(["CIDADE", "BAIRRO"]).mean().reset_index()

        # =================================================================
        # 2. GERAÇÃO DE CENÁRIOS MULTIDIMENSIONAIS (FATO PREDIÇÃO)
        # =================================================================
        print("🔮 Simulando a resposta da cidade aos cenários de risco...")
        df_base_geo = df_ouro.unique(subset="H3_INDEX")
        
        cenarios = [
            {"ID": "Terca_Tarde_Pedestre", "DIA": "2", "PERIODO": "TARDE", "ALVO": "PEDESTRE", "FDS": "0"},
            {"ID": "Sexta_Noite_Motorista", "DIA": "5", "PERIODO": "NOITE", "ALVO": "MOTORISTA", "FDS": "0"},
            {"ID": "Sabado_Madruga_Geral", "DIA": "7", "PERIODO": "MADRUGADA", "ALVO": "GERAL", "FDS": "1"}
        ]

        preds_final = []
        for c in cenarios:
            df_c = df_base_geo.with_columns([
                pl.lit(c["DIA"]).alias("FEAT_DIA_SEMANA"),
                pl.lit(c["PERIODO"]).alias("SAZON_PERIODO"),
                pl.lit(c["ALVO"]).alias("FEAT_PERFIL_VITIMA"),
                pl.lit(f"{c['PERIODO']}_{c['ALVO']}").alias("FEAT_CONTEXTO_CRITICO"),
                pl.lit(c["FDS"]).alias("FEAT_IS_FIM_DE_SEMANA"),
                pl.lit("0").alias("FEAT_IS_FERIADO"),
                pl.lit("DIA_UTIL").alias("FEAT_TIPO_FERIADO"),
                pl.lit("6").alias("FEAT_MES")
            ])
            
            X_c = df_c.select(modelo.feature_names_).to_pandas()
            for col in X_c.select_dtypes(['object', 'category']).columns: 
                X_c[col] = X_c[col].astype(str)
            
            # Exportamos apenas a Chave (H3) e os Dados Dinâmicos para a Tabela Fato
            df_res = df_c.select(["H3_INDEX"]).with_columns([
                pl.lit(c["ID"]).alias("CENARIO_ID"),
                pl.Series("RISCO_SCORE", modelo.predict(X_c)).round(2)
            ])
            preds_final.append(df_res)

        # =================================================================
        # 3. EXPORTAÇÃO PARA O DATA LAKE (R2)
        # =================================================================
        print("📦 Salvando artefatos de inteligência no Cloudflare R2...")
        buf_mapa = io.BytesIO()
        pl.concat(preds_final).write_parquet(buf_mapa, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_fact_predicoes.parquet", Body=buf_mapa.getvalue())
        
        buf_shap = io.BytesIO()
        pl.from_pandas(df_shap_geo).write_parquet(buf_shap, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dim_shap.parquet", Body=buf_shap.getvalue())
        
        print("✅ [MÓDULO 1 CONCLUÍDO] Inteligência gerada e salva com sucesso.")

if __name__ == "__main__":
    GeradorInteligenciaSafeDriver().gerar_dados()
