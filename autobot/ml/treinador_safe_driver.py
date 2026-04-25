import os
import io
import json
import boto3
import polars as pl
import pandas as pd
import numpy as np
import time
import requests
import warnings
import shap
from datetime import datetime
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, r2_score
from botocore.config import Config

warnings.filterwarnings("ignore", category=FutureWarning)

class TreinadorSafeDriver:
    """
    Motor de Treinamento SafeDriver.
    Plumbing de conexão R2 ultra-estável (inspirado na v1) + Engine Tweedie (v2).
    """
    def __init__(self):
        self.projeto = os.getenv("NOME_PROJETO", "safedriver").strip().lower()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # 1. A MÁGICA DA CONEXÃO QUE FUNCIONA
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
        # 2. CAMINHO ABSOLUTO ESTÁVEL
        self.caminho_abt = f"datalake/ouro/{self.projeto}_abt_treino.parquet"
        self.modelo_local = "modelo_safedriver_catboost.cbm"
        
        self.auditoria = {
            "projeto": self.projeto,
            "fase": "Treinamento CatBoost (Tweedie Regression)",
            "data_processamento": str(datetime.now()),
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except: pass

    def executar_treino(self):
        inicio_processo = time.time()
        print(f"[INFO] Buscando ABT Ouro em: {self.caminho_abt}", flush=True)
        
        # =================================================================
        # 1. CARREGAMENTO E SANEAMENTO
        # =================================================================
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.caminho_abt)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        linhas_originais = df.height
        
        # Filtra colunas dinamicamente para garantir que a IA não leia IDs de texto
        cols_features = [c for c in df.columns if any(c.startswith(pre) for pre in ["FEAT_", "MACRO_", "CENSO_", "MICRO_", "SAZON_", "FS_"])]
        if "H3_INDEX" not in cols_features: cols_features.append("H3_INDEX")
        target = "LABEL_PESO_RISCO"

        # =================================================================
        # 2. SPLIT TEMPORAL
        # =================================================================
        print("[INFO] Executando split temporal (85/15)...", flush=True)
        df = df.sort(["DATAOCORRENCIA", "H3_INDEX"])
        split_idx = int(df.height * 0.85)
        
        train_df = df.slice(0, split_idx)
        test_df = df.slice(split_idx, df.height - split_idx)
        del df

        # =================================================================
        # 3. BALANCEAMENTO (UNDERSAMPLING)
        # =================================================================
        print("[INFO] Aplicando undersampling no treino...", flush=True)
        df_treino_graves = train_df.filter(pl.col(target) >= 4.0)
        df_treino_leves = train_df.filter(pl.col(target) < 4.0)
        
        n_amostra = min(df_treino_graves.height * 3, df_treino_leves.height)
        df_treino_leves_amostra = df_treino_leves.sample(n=n_amostra, seed=42)
        
        train_df = pl.concat([df_treino_graves, df_treino_leves_amostra]).sample(fraction=1.0, seed=42)
        
        linhas_balanceadas_treino = train_df.height
        linhas_teste_real = test_df.height

        # =================================================================
        # 4. PREPARAÇÃO PARA O CATBOOST
        # =================================================================
        pdf_train = train_df.select(cols_features + [target]).to_pandas()
        pdf_test = test_df.select(cols_features + [target]).to_pandas()
        
        cat_features_declaradas = [
            "H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", 
            "FEAT_PERFIL_VITIMA", "FEAT_CONTEXTO_CRITICO", "FEAT_TIPO_FERIADO", 
            "FEAT_IS_FIM_DE_SEMANA", "FEAT_TIPO_DIA"
        ]
        cat_features = [c for c in cat_features_declaradas if c in pdf_train.columns]
        
        for col in cat_features:
            pdf_train[col] = pdf_train[col].fillna("DESCONHECIDO").astype(str)
            pdf_test[col] = pdf_test[col].fillna("DESCONHECIDO").astype(str)

        # Pesos amplificados para crimes graves
        pesos_treino = np.where(pdf_train[target] > 0, 10.0, 1.0)

        # =================================================================
        # 5. TREINAMENTO (ENGINE TWEEDIE)
        # =================================================================
        print("[INFO] Iniciando CatBoost (Tweedie Regression)...")
        train_pool = Pool(pdf_train[cols_features], pdf_train[target], cat_features=cat_features, weight=pesos_treino)
        test_pool = Pool(pdf_test[cols_features], pdf_test[target], cat_features=cat_features)

        modelo = CatBoostRegressor(
            iterations=3000,
            learning_rate=0.01,
            depth=8,
            l2_leaf_reg=2.0,
            loss_function='Tweedie:variance_power=1.6', # Ideal para dados zero-inflados
            eval_metric='MAE',
            od_type='Iter',
            od_wait=300,
            random_seed=42,
            verbose=250
        )

        modelo.fit(train_pool, eval_set=test_pool)

        # =================================================================
        # 6. EXPLICABILIDADE E MÉTRICAS
        # =================================================================
        print("[INFO] Calculando SHAP e Métricas...")
        explainer = shap.TreeExplainer(modelo)
        sample_test = pdf_test[cols_features].sample(min(5000, len(pdf_test)), random_state=42)
        shap_values = explainer.shap_values(sample_test)
        
        shap_importance = pd.DataFrame({
            'feature': cols_features,
            'impacto_medio': np.abs(shap_values).mean(0)
        }).sort_values(by='impacto_medio', ascending=False)

        y_pred = modelo.predict(pdf_test[cols_features])
        mae = mean_absolute_error(pdf_test[target], y_pred)
        r2 = r2_score(pdf_test[target], y_pred)
        duracao = time.time() - inicio_processo

        # =================================================================
        # 7. UPLOAD BLINDADO (VIA PUT_OBJECT)
        # =================================================================
        print("[INFO] Exportando artefatos para o Cloudflare R2...")
        modelo.save_model(self.modelo_local)
        
        # Faz upload lendo o arquivo em bytes (resolve problemas de permissão multipart)
        with open(self.modelo_local, "rb") as f:
            self.s3.put_object(Bucket=self.bucket, Key=f"modelos/{self.modelo_local}", Body=f.read())
        
        shap_json = json.dumps(shap_importance.to_dict(orient='records'), indent=4)
        self.s3.put_object(Bucket=self.bucket, Key="modelos/SHAP_IMPORTANCE.json", Body=shap_json.encode())

        self.auditoria["metricas"] = {
            "linhas_originais": linhas_originais,
            "r2_score_validacao": round(r2, 4),
            "mae_validacao": round(mae, 4),
            "tempo_execucao_seg": round(duracao, 2)
        }
        
        buf_auditoria = io.BytesIO(json.dumps(self.auditoria, indent=4).encode())
        self.s3.put_object(Bucket=self.bucket, Key="modelos/AUDITORIA_TREINO.json", Body=buf_auditoria.getvalue())

        # =================================================================
        # 8. DISCORD REPORT
        # =================================================================
        top_15 = shap_importance.head(15)
        resumo_shap = "\n".join([f"   [{str(i+1).zfill(2)}] {r['feature'].ljust(30)} : {r['impacto_medio']:.4f}" for i, r in top_15.iterrows()])
        
        report = (
            f"==============================================================\n"
            f" RELATÓRIO DE TREINAMENTO - {self.projeto.upper()} (TWEEDIE) \n"
            f"==============================================================\n"
            f"1. VOLUMETRIA E BALANCEAMENTO\n"
            f"   • Base Original           : {linhas_originais} registros\n"
            f"   • Base Treinamento        : {linhas_balanceadas_treino} registros (Undersampled)\n"
            f"   • Base Validação          : {linhas_teste_real} registros\n\n"
            f"2. PERFORMANCE (ALVO: PESO DE RISCO)\n"
            f"   • R² Score                : {r2:.4f}\n"
            f"   • Erro Médio (MAE)        : {mae:.4f}\n\n"
            f"3. DNA DO CRIME (TOP 15 SHAP VALUES)\n"
            f"{resumo_shap}\n"
            f"==============================================================\n"
            f"Status: Modelo Salvo no R2 | Tempo: {duracao/60:.2f} min\n"
            f"==============================================================\n"
        )
        print(report)
        self._notificar_discord(f"```text\n{report}\n```")
        
        if os.path.exists(self.modelo_local): os.remove(self.modelo_local)

if __name__ == "__main__":
    TreinadorSafeDriver().executar_treino()
