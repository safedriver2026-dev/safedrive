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
            "fase": "Treinamento Preditivo (Pesos + Perfil Vítima)",
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
        
        # 1. CARREGAMENTO DOS DADOS
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.caminho_abt)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # 2. SELEÇÃO DE FEATURES (Agora pega automaticamente o FEAT_PERFIL_VITIMA)
        cols_features = [c for c in df.columns if any(c.startswith(pre) for pre in ["FEAT_", "INFRA_", "CENSO_", "MICRO_", "SAZON_"])]
        cols_features.append("H3_INDEX")
        target = "LABEL_PESO_RISCO"

        # 3. SPLIT TEMPORAL (Blindagem contra Overfitting)
        print("📅 Ordenando dados e executando Split Temporal...")
        df = df.sort("DATAOCORRENCIA")
        total_rows = df.height
        split_idx = int(total_rows * 0.85)
        
        train_df = df.slice(0, split_idx)
        test_df = df.slice(split_idx, total_rows - split_idx)

        # Liberando memória do dataframe original (Boa prática para Big Data)
        del df 

        # 4. PREPARAÇÃO (Conversão e Tipagem Segura)
        pdf_train = train_df.select(cols_features + [target]).to_pandas()
        pdf_test = test_df.select(cols_features + [target]).to_pandas()
        
        # ✨ AQUI ESTÁ A CORREÇÃO CRUCIAL: FEAT_PERFIL_VITIMA adicionado na lista!
        cat_features_declaradas = ["H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", "FEAT_PERFIL_VITIMA"]
        cat_features = [c for c in cat_features_declaradas if c in pdf_train.columns]
        
        # Conversão segura para string (evita que 'NaN' vire float e quebre o CatBoost)
        for col in cat_features:
            pdf_train[col] = pdf_train[col].fillna("DESCONHECIDO").astype(str)
            pdf_test[col] = pdf_test[col].fillna("DESCONHECIDO").astype(str)

        # 5. CÁLCULO DE PESOS (Foco em Severidade Extrema)
        print("⚖️ Calculando Pesos de Amostra...")
        pesos_treino = np.where(pdf_train[target] == 10, 50,
                       np.where(pdf_train[target] == 5, 10, 1))

        pesos_teste = np.where(pdf_test[target] == 10, 50,
                      np.where(pdf_test[target] == 5, 10, 1))

        # 6. CONFIGURAÇÃO E TREINAMENTO
        print("🧠 Treinando CatBoost com Contexto Espacial e Perfil...")
        train_pool = Pool(pdf_train[cols_features], pdf_train[target], cat_features=cat_features, weight=pesos_treino)
        test_pool = Pool(pdf_test[cols_features], pdf_test[target], cat_features=cat_features, weight=pesos_teste)

        modelo = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.05,       
            depth=7,                  
            l2_leaf_reg=3,            
            
            loss_function='RMSE',     
            eval_metric='MAE',
            
            od_type='Iter',
            od_wait=100,              
            use_best_model=True,
            
            max_ctr_complexity=2,
            random_seed=42,
            verbose=100
        )

        modelo.fit(train_pool, eval_set=test_pool)
        
        # 7. EXPLICABILIDADE (SHAP)
        print("🔍 Calculando explicabilidade SHAP...")
        explainer = shap.TreeExplainer(modelo)
        sample_test = pdf_test[cols_features].sample(min(5000, len(pdf_test)))
        shap_values = explainer.shap_values(sample_test)
        
        shap_importance = pd.DataFrame({
            'feature': cols_features,
            'impacto_medio': np.abs(shap_values).mean(0)
        }).sort_values(by='impacto_medio', ascending=False)

        top_5 = shap_importance.head(5)
        resumo_shap = "\n".join([f"• {r['feature']}: {r['impacto_medio']:.4f}" for _, r in top_5.iterrows()])

        # 8. MÉTRICAS E PERFORMANCE
        y_pred = modelo.predict(pdf_test[cols_features])
        mae = mean_absolute_error(pdf_test[target], y_pred)
        r2 = r2_score(pdf_test[target], y_pred)
        duracao = time.time() - inicio_processo

        # 9. SALVAMENTO E UPLOAD
        print("💾 Deploy do modelo preditivo no R2...")
        modelo.save_model(self.modelo_local)
        with open(self.modelo_local, "rb") as f:
            self.s3.put_object(Bucket=self.bucket, Key=f"modelos/{self.modelo_local}", Body=f.read())
        
        shap_json = json.dumps(shap_importance.to_dict(orient='records'), indent=4)
        self.s3.put_object(Bucket=self.bucket, Key="modelos/SHAP_IMPORTANCE.json", Body=shap_json.encode())

        # 10. RELATÓRIO DISCORD
        report = (
            f"🛡️ **[SafeDriver] IA Preditiva Otimizada (Perfis + Pesos)**\n"
            f"```ml\n"
            f"MÉTRICAS DE RISCO:\n"
            f"• R² Score: {r2:.4f}\n"
            f"• MAE (Erro Médio): {mae:.4f}\n"
            f"-------------------------------------\n"
            f"DRIVERS DE RISCO (SHAP):\n"
            f"{resumo_shap}\n"
            f"-------------------------------------\n"
            f"Tempo: {duracao/60:.2f} min | ABT: {len(pdf_train)} linhas\n"
            f"Status: MODELO BALANCEADO E ATIVO NO R2\n"
            f"```"
        )
        print(report)
        self._notificar_discord(report)
        
        if os.path.exists(self.modelo_local):
            os.remove(self.modelo_local)

if __name__ == "__main__":
    trainer = TreinadorSafeDriver()
    trainer.executar_treino()
