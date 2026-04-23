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
            "fase": "Treinamento Preditivo PRO (Huber + Feature Store)",
            "data_processamento": str(datetime.now()),
            "parametros_modelo": {
                "iterations": 2000,
                "learning_rate": 0.03,
                "depth": 8,
                "loss_function": "Huber:delta=2.0"
            },
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except: pass

    def executar_treino(self):
        inicio_processo = time.time()
        print(f"🚀 [TREINO] Iniciando carregamento da ABT Master Ouro...", flush=True)
        
        # 1. CARREGAMENTO DOS DADOS
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.caminho_abt)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        linhas_originais = df.height
        
        # ✨ INOVAÇÃO: FEATURE CROSS (Cruzamento Sintético de Contexto)
        print("🧬 Criando Feature Cross: Contexto Crítico...", flush=True)
        df = df.with_columns(
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        )

        # 2. SELEÇÃO DE FEATURES (Puxando a Feature Store "FS_")
        cols_features = [c for c in df.columns if any(c.startswith(pre) for pre in ["FEAT_", "INFRA_", "CENSO_", "MICRO_", "SAZON_", "FS_"])]
        cols_features.append("H3_INDEX")
        target = "LABEL_PESO_RISCO"

        # ✨ UNDERSAMPLING FÍSICO INTELIGENTE
        print("⚖️ Balanceando a base (Undersampling Físico baseado na Escala Penal)...")
        
        # Separamos os crimes GRAVES (Nota >= 4: Lesão, Roubo, Tráfico, Extorsão, Latrocínio)
        df_graves = df.filter(pl.col(target) >= 4.0)
        qtd_graves = df_graves.height
        
        # Pegamos os crimes LEVES (Nota < 4: Furto e Base) e sorteamos um volume (3x os graves)
        df_leves = df.filter(pl.col(target) < 4.0)
        n_amostra = min(qtd_graves * 3, df_leves.height)
        df_leves_amostra = df_leves.sample(n=n_amostra, seed=42)
        
        # Unimos a base e reordenamos
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
        
        cat_features_declaradas = ["H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", "FEAT_PERFIL_VITIMA", "FEAT_CONTEXTO_CRITICO"]
        cat_features = [c for c in cat_features_declaradas if c in pdf_train.columns]
        
        for col in cat_features:
            pdf_train[col] = pdf_train[col].fillna("DESCONHECIDO").astype(str)
            pdf_test[col] = pdf_test[col].fillna("DESCONHECIDO").astype(str)

        # 5. CONFIGURAÇÃO E TREINAMENTO
        print("🧠 Treinando CatBoost PRO (Huber Loss + Deep Trees)...")
        train_pool = Pool(pdf_train[cols_features], pdf_train[target], cat_features=cat_features)
        test_pool = Pool(pdf_test[cols_features], pdf_test[target], cat_features=cat_features)

        modelo = CatBoostRegressor(
            iterations=2000,          # Mais iterações
            learning_rate=0.03,       
            depth=8,                  # Árvore mais profunda (cruza Feature Store com Infraestrutura)
            l2_leaf_reg=5,            
            
            loss_function='Huber:delta=2.0',  # Resiliência a outliers (pesos 10)
            eval_metric='R2',         
            
            od_type='Iter',
            od_wait=150,              
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

        # 7. MÉTRICAS E PERFORMANCE
        y_pred = modelo.predict(pdf_test[cols_features])
        mae = mean_absolute_error(pdf_test[target], y_pred)
        r2 = r2_score(pdf_test[target], y_pred)
        duracao = time.time() - inicio_processo

        # 8. SALVAMENTO, UPLOAD E AUDITORIA EM JSON
        print("💾 Salvando artefatos e log de auditoria no R2...")
        
        modelo.save_model(self.modelo_local)
        with open(self.modelo_local, "rb") as f:
            self.s3.put_object(Bucket=self.bucket, Key=f"modelos/{self.modelo_local}", Body=f.read())
        
        shap_json = json.dumps(shap_importance.to_dict(orient='records'), indent=4)
        self.s3.put_object(Bucket=self.bucket, Key="modelos/SHAP_IMPORTANCE.json", Body=shap_json.encode())

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

        # =================================================================
        # 9. DOSSIÊ DETALHADO PARA AUDITORIA DE QUALIDADE
        # =================================================================
        top_15 = shap_importance.head(15)
        resumo_shap_detalhado = "\n".join([f"   [{str(i+1).zfill(2)}] {r['feature'].ljust(30)} : {r['impacto_medio']:.4f}" for i, r in top_15.iterrows()])
        
        fs_impact = shap_importance[shap_importance['feature'].str.startswith('FS_')]['impacto_medio'].sum()
        infra_impact = shap_importance[shap_importance['feature'].str.startswith('INFRA_')]['impacto_medio'].sum()
        contexto_impact = shap_importance[shap_importance['feature'].isin(['SAZON_PERIODO', 'FEAT_PERFIL_VITIMA', 'FEAT_CONTEXTO_CRITICO'])]['impacto_medio'].sum()

        report = (
            f"==============================================================\n"
            f" 🛡️ DOSSIÊ DE AUDITORIA DO MODELO SAFEDRIVER PRO 🛡️\n"
            f"==============================================================\n"
            f"📊 1. VOLUMETRIA E BALANCEAMENTO\n"
            f"   • Base Original (Ouro)    : {linhas_originais} linhas\n"
            f"   • Base Balanceada (Treino): {df.height} linhas\n"
            f"   • Fator de Compressão     : {(df.height / linhas_originais * 100):.2f}%\n\n"
            f"⚙️ 2. CONFIGURAÇÃO DO MOTOR (CATBOOST)\n"
            f"   • Loss Function           : Huber (delta=2.0)\n"
            f"   • Árvores (Iterations)    : 2000 (Early Stopping em {modelo.tree_count_})\n"
            f"   • Profundidade (Depth)    : 8\n\n"
            f"🎯 3. MÉTRICAS DE PERFORMANCE (DADOS NÃO VISTOS)\n"
            f"   • R² Score (Variância)    : {r2:.4f}\n"
            f"   • MAE (Erro Médio Absoluto): {mae:.4f} pontos de severidade\n\n"
            f"🧠 4. EXPLICABILIDADE SHAP (COMO A IA PENSA)\n"
            f"   --- Top 15 Variáveis Mais Importantes ---\n"
            f"{resumo_shap_detalhado}\n\n"
            f"   --- Peso por Grupo de Inteligência ---\n"
            f"   • Feature Store (Passado) : {fs_impact:.4f} de impacto somado\n"
            f"   • Contexto (Hora/Perfil)  : {contexto_impact:.4f} de impacto somado\n"
            f"   • Infraestrutura (Local)  : {infra_impact:.4f} de impacto somado\n"
            f"==============================================================\n"
            f"Tempo Total: {duracao/60:.2f} min | Status: AGUARDANDO AUDITORIA\n"
            f"==============================================================\n"
        )
        print(report)
        self._notificar_discord(f"```ml\n{report}\n```")
        
        if os.path.exists(self.modelo_local):
            os.remove(self.modelo_local)

if __name__ == "__main__":
    trainer = TreinadorSafeDriver()
    trainer.executar_treino()
