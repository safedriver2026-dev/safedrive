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
        
        # Dicionário de Auditoria Inicializado
        self.auditoria = {
            "projeto": "SafeDriver - Treinamento de IA",
            "fase": "Treinamento Preditivo (Undersampling + Perfil)",
            "data_processamento": str(datetime.now()),
            "parametros_modelo": {
                "iterations": 1500,
                "learning_rate": 0.03,
                "depth": 7,
                "l2_leaf_reg": 5
            },
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
        linhas_originais = df.height
        
        # 2. SELEÇÃO DE FEATURES (Pega automaticamente o FEAT_PERFIL_VITIMA)
        cols_features = [c for c in df.columns if any(c.startswith(pre) for pre in ["FEAT_", "INFRA_", "CENSO_", "MICRO_", "SAZON_"])]
        cols_features.append("H3_INDEX")
        target = "LABEL_PESO_RISCO"

        # ✨ UNDERSAMPLING FÍSICO
        print("⚖️ Balanceando a base (Undersampling Físico da classe majoritária)...")
        
        # Separamos os crimes de alto risco (pesos 5 e 10)
        df_graves = df.filter(pl.col(target) > 1)
        qtd_graves = df_graves.height
        
        # Pegamos os crimes leves e sorteamos um volume equivalente (3x o número de graves)
        df_leves = df.filter(pl.col(target) == 1)
        n_amostra = min(qtd_graves * 3, df_leves.height)
        df_leves_amostra = df_leves.sample(n=n_amostra, seed=42)
        
        # Unimos a base e reordenamos (importante embaralhar após o concat)
        df = pl.concat([df_graves, df_leves_amostra]).sample(fraction=1.0, seed=42)
        print(f"📉 Base reduzida de {linhas_originais} para {df.height} linhas equilibradas.")

        # 3. SPLIT TEMPORAL (Blindagem contra Overfitting)
        print("📅 Ordenando dados e executando Split Temporal...")
        df = df.sort("DATAOCORRENCIA")
        total_rows = df.height
        split_idx = int(total_rows * 0.85)
        
        train_df = df.slice(0, split_idx)
        test_df = df.slice(split_idx, total_rows - split_idx)

        del df  # Liberando memória

        # 4. PREPARAÇÃO (Conversão e Tipagem Segura)
        pdf_train = train_df.select(cols_features + [target]).to_pandas()
        pdf_test = test_df.select(cols_features + [target]).to_pandas()
        
        cat_features_declaradas = ["H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", "FEAT_PERFIL_VITIMA"]
        cat_features = [c for c in cat_features_declaradas if c in pdf_train.columns]
        
        for col in cat_features:
            pdf_train[col] = pdf_train[col].fillna("DESCONHECIDO").astype(str)
            pdf_test[col] = pdf_test[col].fillna("DESCONHECIDO").astype(str)

        # 5. CONFIGURAÇÃO E TREINAMENTO
        print("🧠 Treinando CatBoost na base balanceada...")
        train_pool = Pool(pdf_train[cols_features], pdf_train[target], cat_features=cat_features)
        test_pool = Pool(pdf_test[cols_features], pdf_test[target], cat_features=cat_features)

        modelo = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,       
            depth=7,                  
            l2_leaf_reg=5,            
            
            loss_function='RMSE',     
            eval_metric='R2',         
            
            od_type='Iter',
            od_wait=100,              
            use_best_model=True,
            
            max_ctr_complexity=2,
            random_seed=42,
            verbose=100
        )

        modelo.fit(train_pool, eval_set=test_pool)
        
        # 6. EXPLICABILIDADE (SHAP)
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

        # 7. MÉTRICAS E PERFORMANCE
        y_pred = modelo.predict(pdf_test[cols_features])
        mae = mean_absolute_error(pdf_test[target], y_pred)
        r2 = r2_score(pdf_test[target], y_pred)
        duracao = time.time() - inicio_processo

        # 8. SALVAMENTO, UPLOAD E AUDITORIA EM JSON
        print("💾 Salvando artefatos e log de auditoria no R2...")
        
        # A. Salvar Modelo Binário
        modelo.save_model(self.modelo_local)
        with open(self.modelo_local, "rb") as f:
            self.s3.put_object(Bucket=self.bucket, Key=f"modelos/{self.modelo_local}", Body=f.read())
        
        # B. Salvar Importância SHAP
        shap_json = json.dumps(shap_importance.to_dict(orient='records'), indent=4)
        self.s3.put_object(Bucket=self.bucket, Key="modelos/SHAP_IMPORTANCE.json", Body=shap_json.encode())

        # C. Salvar JSON de Auditoria de Treinamento
        self.auditoria["metricas"] = {
            "linhas_originais_abt": linhas_originais,
            "linhas_pos_undersampling": int(pdf_train.shape[0] + pdf_test.shape[0]),
            "r2_score_validacao": round(r2, 4),
            "mae_validacao": round(mae, 4),
            "tempo_execucao_segundos": round(duracao, 2),
            "top_features": shap_importance['feature'].head(10).tolist()
        }
        
        buf_json_auditoria = io.BytesIO(json.dumps(self.auditoria, indent=4).encode())
        self.s3.put_object(
            Bucket=self.bucket, 
            Key="modelos/AUDITORIA_TREINO_CATBOOST.json", 
            Body=buf_json_auditoria.getvalue()
        )

        # 9. RELATÓRIO DISCORD
        report = (
            f"🛡️ **[SafeDriver] IA Preditiva (Undersampling + Auditoria)**\n"
            f"```ml\n"
            f"MÉTRICAS DE RISCO:\n"
            f"• R² Score: {r2:.4f}\n"
            f"• MAE (Erro Médio): {mae:.4f}\n"
            f"-------------------------------------\n"
            f"DRIVERS DE RISCO (SHAP):\n"
            f"{resumo_shap}\n"
            f"-------------------------------------\n"
            f"Tempo: {duracao/60:.2f} min | ABT Balanceada: {len(pdf_train)} linhas\n"
            f"Status: MODELO FORÇADO A APRENDER\n"
            f"```"
        )
        print(report)
        self._notificar_discord(report)
        
        if os.path.exists(self.modelo_local):
            os.remove(self.modelo_local)

if __name__ == "__main__":
    trainer = TreinadorSafeDriver()
    trainer.executar_treino()
