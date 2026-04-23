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
        
        # Inicialização do log de auditoria
        self.auditoria = {
            "projeto": "SafeDriver",
            "fase": "Treinamento de Modelo Preditivo",
            "data_processamento": str(datetime.now()),
            "parametros_modelo": {
                "iterations": 1500,
                "learning_rate": 0.01,
                "depth": 8,
                "loss_function": "Expectile:alpha=0.85"
            },
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except: pass

    def executar_treino(self):
        inicio_processo = time.time()
        print("[INFO] Iniciando carregamento da ABT Ouro...", flush=True)
        
        # =================================================================
        # 1. CARREGAMENTO DOS DADOS E ENGENHARIA DE FEATURES
        # =================================================================
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.caminho_abt)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        linhas_originais = df.height
        
        print("[INFO] Processando Feature Cross (Sazonalidade x Perfil)...", flush=True)
        df = df.with_columns(
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        )

        cols_features = [c for c in df.columns if any(c.startswith(pre) for pre in ["FEAT_", "INFRA_", "CENSO_", "MICRO_", "SAZON_", "FS_"])]
        cols_features.append("H3_INDEX")
        target = "LABEL_PESO_RISCO"

        # =================================================================
        # 2. BALANCEAMENTO DA BASE (UNDERSAMPLING FÍSICO)
        # =================================================================
        print("[INFO] Aplicando undersampling baseado na distribuição do target...", flush=True)
        
        df_graves = df.filter(pl.col(target) >= 4.0)
        qtd_graves = df_graves.height
        
        df_leves = df.filter(pl.col(target) < 4.0)
        n_amostra = min(qtd_graves * 3, df_leves.height)
        df_leves_amostra = df_leves.sample(n=n_amostra, seed=42)
        
        df = pl.concat([df_graves, df_leves_amostra]).sample(fraction=1.0, seed=42)
        linhas_balanceadas = df.height 
        
        print(f"[INFO] Volumetria ajustada: {linhas_balanceadas} registros mantidos.")

        # =================================================================
        # 3. SPLIT TEMPORAL
        # =================================================================
        print("[INFO] Executando split temporal (85/15)...")
        df = df.sort("DATAOCORRENCIA")
        total_rows = df.height
        split_idx = int(total_rows * 0.85)
        
        train_df = df.slice(0, split_idx)
        test_df = df.slice(split_idx, total_rows - split_idx)

        # Liberação explícita de memória
        del df  

        # =================================================================
        # 4. PREPARAÇÃO DO DATASET (Conversão para Pandas)
        # =================================================================
        pdf_train = train_df.select(cols_features + [target]).to_pandas()
        pdf_test = test_df.select(cols_features + [target]).to_pandas()
        
        # Adicionada FEAT_IS_FIM_DE_SEMANA como variável categórica explícita
        cat_features_declaradas = [
            "H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", 
            "FEAT_PERFIL_VITIMA", "FEAT_CONTEXTO_CRITICO", "FEAT_TIPO_FERIADO", 
            "FEAT_IS_FIM_DE_SEMANA"
        ]
        cat_features = [c for c in cat_features_declaradas if c in pdf_train.columns]
        
        for col in cat_features:
            pdf_train[col] = pdf_train[col].fillna("DESCONHECIDO").astype(str)
            pdf_test[col] = pdf_test[col].fillna("DESCONHECIDO").astype(str)

        # =================================================================
        # 5. TREINAMENTO DO MODELO (CATBOOST)
        # =================================================================
        print("[INFO] Iniciando treinamento do modelo CatBoost (Expectile Regression)...")
        train_pool = Pool(pdf_train[cols_features], pdf_train[target], cat_features=cat_features)
        test_pool = Pool(pdf_test[cols_features], pdf_test[target], cat_features=cat_features)

        modelo = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.01,
            depth=8,
            l2_leaf_reg=5,
            loss_function='Expectile:alpha=0.85', # Penalização assimétrica para minimizar falsos negativos
            eval_metric='R2',         
            od_type='Iter',
            od_wait=200,
            use_best_model=True,
            thread_count=-1,
            random_seed=42,
            verbose=100
        )

        modelo.fit(train_pool, eval_set=test_pool)
        
        # =================================================================
        # 6. EXPLICABILIDADE DO MODELO (SHAP)
        # =================================================================
        print("[INFO] Calculando valores SHAP...")
        explainer = shap.TreeExplainer(modelo)
        sample_test = pdf_test[cols_features].sample(min(5000, len(pdf_test)))
        shap_values = explainer.shap_values(sample_test)
        
        shap_importance = pd.DataFrame({
            'feature': cols_features,
            'impacto_medio': np.abs(shap_values).mean(0)
        }).sort_values(by='impacto_medio', ascending=False)

        # =================================================================
        # 7. AVALIAÇÃO DE PERFORMANCE E SALVAMENTO
        # =================================================================
        y_pred = modelo.predict(pdf_test[cols_features])
        mae = mean_absolute_error(pdf_test[target], y_pred)
        r2 = r2_score(pdf_test[target], y_pred)
        duracao = time.time() - inicio_processo

        print("[INFO] Exportando artefatos para o Data Lake...")
        modelo.save_model(self.modelo_local)
        with open(self.modelo_local, "rb") as f:
            self.s3.put_object(Bucket=self.bucket, Key=f"modelos/{self.modelo_local}", Body=f.read())
        
        shap_json = json.dumps(shap_importance.to_dict(orient='records'), indent=4)
        self.s3.put_object(Bucket=self.bucket, Key="modelos/SHAP_IMPORTANCE.json", Body=shap_json.encode())

        self.auditoria["metricas"] = {
            "linhas_originais_abt": linhas_originais,
            "linhas_pos_undersampling": linhas_balanceadas,
            "r2_score_validacao": round(r2, 4),
            "mae_validacao": round(mae, 4),
            "tempo_execucao_segundos": round(duracao, 2),
            "top_features": shap_importance['feature'].head(10).tolist()
        }
        
        buf_json_auditoria = io.BytesIO(json.dumps(self.auditoria, indent=4).encode())
        self.s3.put_object(Bucket=self.bucket, Key="modelos/AUDITORIA_TREINO_CATBOOST.json", Body=buf_json_auditoria.getvalue())

        # =================================================================
        # 8. GERAÇÃO DE LOG DE AUDITORIA
        # =================================================================
        top_15 = shap_importance.head(15)
        resumo_shap_detalhado = "\n".join([f"   [{str(i+1).zfill(2)}] {r['feature'].ljust(30)} : {r['impacto_medio']:.4f}" for i, r in top_15.iterrows()])
        
        # Agregadores de SHAP atualizados para incluir as novas features
        fs_impact = shap_importance[shap_importance['feature'].str.startswith('FS_')]['impacto_medio'].sum()
        infra_impact = shap_importance[shap_importance['feature'].str.startswith('INFRA_')]['impacto_medio'].sum()
        contexto_impact = shap_importance[shap_importance['feature'].isin([
            'SAZON_PERIODO', 'FEAT_PERFIL_VITIMA', 'FEAT_CONTEXTO_CRITICO', 
            'FEAT_IS_FERIADO', 'FEAT_TIPO_FERIADO', 'FEAT_IS_PONTO_FACULTATIVO',
            'FEAT_IS_FIM_DE_SEMANA', 'FEAT_DIA_SEMANA', 'FEAT_MES'
        ])]['impacto_medio'].sum()

        report = (
            f"==============================================================\n"
            f" RELATÓRIO DE TREINAMENTO - SAFEDRIVER \n"
            f"==============================================================\n"
            f"1. VOLUMETRIA E BALANCEAMENTO\n"
            f"   • Base Original           : {linhas_originais} registros\n"
            f"   • Base Treinamento        : {linhas_balanceadas} registros\n"
            f"   • Retenção de Dados       : {(linhas_balanceadas / linhas_originais * 100):.2f}%\n\n"
            f"2. HIPERPARÂMETROS DO MODELO\n"
            f"   • Algoritmo               : CatBoostRegressor\n"
            f"   • Função de Perda         : Expectile (alpha=0.85)\n"
            f"   • Iterações Utilizadas    : {modelo.tree_count_}\n"
            f"   • Profundidade da Árvore  : 8\n\n"
            f"3. PERFORMANCE EM VALIDAÇÃO\n"
            f"   • R² Score                : {r2:.4f}\n"
            f"   • Erro Médio Absoluto     : {mae:.4f}\n\n"
            f"4. ANÁLISE DE IMPORTÂNCIA DE FEATURES (SHAP)\n"
            f"   --- Top 15 Variáveis Mais Relevantes ---\n"
            f"{resumo_shap_detalhado}\n\n"
            f"   --- Agregação por Grupo de Variáveis ---\n"
            f"   • Feature Store Histórica : {fs_impact:.4f}\n"
            f"   • Contexto (Temporal/Alvo): {contexto_impact:.4f}\n"
            f"   • Infraestrutura Física   : {infra_impact:.4f}\n"
            f"==============================================================\n"
            f"Duração Total: {duracao/60:.2f} min | Status: Processamento Concluído\n"
            f"==============================================================\n"
        )
        print(report)
        self._notificar_discord(f"```text\n{report}\n```")
        
        if os.path.exists(self.modelo_local):
            os.remove(self.modelo_local)

if __name__ == "__main__":
    trainer = TreinadorSafeDriver()
    trainer.executar_treino()
