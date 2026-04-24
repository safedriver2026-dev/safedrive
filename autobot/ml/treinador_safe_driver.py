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
    """
    Modulo responsavel pelo treinamento do modelo preditivo de risco (CatBoost).
    Executa o particionamento temporal, balanceamento de classes (undersampling),
    modelagem de severidade via Expectile Loss e analise de explicabilidade (SHAP).
    """
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
        
        self.auditoria = {
            "projeto": "SafeDriver",
            "fase": "Treinamento de Modelo Preditivo (High-Res)",
            "data_processamento": str(datetime.now()),
            "parametros_modelo": {
                "iterations": 2500,
                "learning_rate": 0.005,
                "depth": 8,
                "loss_function": "Expectile:alpha=0.85"
            },
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        """Transmite os logs de execucao para a plataforma de telemetria."""
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except: pass

    def executar_treino(self):
        inicio_processo = time.time()
        print("Iniciando carregamento da Analytical Base Table (Camada Ouro)...", flush=True)
        
        # 1. CARREGAMENTO DOS DADOS E MAPEAMENTO DE FEATURES
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.caminho_abt)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        linhas_originais = df.height
        
        # O mapeamento varre a ABT e seleciona apenas os atributos de modelagem
        cols_features = [c for c in df.columns if any(c.startswith(pre) for pre in ["FEAT_", "MACRO_", "CENSO_", "MICRO_", "SAZON_", "FS_"])]
        cols_features.append("H3_INDEX")
        target = "LABEL_PESO_RISCO"

        # 2. PARTICIONAMENTO TEMPORAL BLINDADO (Data Leakage Prevention)
        print("Executando particionamento temporal (Split 85/15)...", flush=True)
        df = df.sort(["DATAOCORRENCIA", "H3_INDEX"])
        
        total_rows = df.height
        split_idx = int(total_rows * 0.85)
        
        train_df = df.slice(0, split_idx)
        test_df = df.slice(split_idx, total_rows - split_idx)

        del df  # Liberacao de memoria volatil

        # 3. BALANCEAMENTO DA BASE DE TREINO (Undersampling Direcionado)
        print("Aplicando balanceamento de classes na base de treinamento...", flush=True)
        
        df_treino_graves = train_df.filter(pl.col(target) >= 4.0)
        qtd_graves = df_treino_graves.height
        
        df_treino_leves = train_df.filter(pl.col(target) < 4.0)
        n_amostra = min(qtd_graves * 3, df_treino_leves.height)
        
        df_treino_leves_amostra = df_treino_leves.sample(n=n_amostra, seed=42)
        train_df = pl.concat([df_treino_graves, df_treino_leves_amostra]).sample(fraction=1.0, seed=42)
        
        linhas_balanceadas_treino = train_df.height
        linhas_teste_real = test_df.height
        
        print(f"Base de treinamento equilibrada: {linhas_balanceadas_treino} registros.")
        print(f"Base de validacao (Mundo Real): {linhas_teste_real} registros.")

        # 4. PREPARAÇÃO DO DATASET PARA ALGORITMO CATBOOST
        pdf_train = train_df.select(cols_features + [target]).to_pandas()
        pdf_test = test_df.select(cols_features + [target]).to_pandas()
        
        # Atualizacao do vetor categorico para refletir a nova estrutura da Camada Ouro
        cat_features_declaradas = [
            "H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", 
            "FEAT_PERFIL_VITIMA", "FEAT_CONTEXTO_CRITICO", "FEAT_TIPO_DIA"
        ]
        cat_features = [c for c in cat_features_declaradas if c in pdf_train.columns]
        
        for col in cat_features:
            pdf_train[col] = pdf_train[col].fillna("DESCONHECIDO").astype(str)
            pdf_test[col] = pdf_test[col].fillna("DESCONHECIDO").astype(str)

        # 5. TREINAMENTO DE MODELO DE ALTA RESOLUÇÃO
        print("Iniciando otimizacao do modelo CatBoostRegressor...", flush=True)
        train_pool = Pool(pdf_train[cols_features], pdf_train[target], cat_features=cat_features)
        test_pool = Pool(pdf_test[cols_features], pdf_test[target], cat_features=cat_features)

        modelo = CatBoostRegressor(
            iterations=2500,
            learning_rate=0.005,
            depth=8,
            l2_leaf_reg=3,
            loss_function='Expectile:alpha=0.85', 
            eval_metric='MAE',        
            od_type='Iter',
            od_wait=300,
            use_best_model=True,
            thread_count=-1,
            random_seed=42,
            verbose=200
        )

        modelo.fit(train_pool, eval_set=test_pool)
        
        # 6. EXPLICABILIDADE DE INTELIGÊNCIA ARTIFICIAL (Valores SHAP)
        print("Processando analise de importancia global (Metodo SHAP)...", flush=True)
        explainer = shap.TreeExplainer(modelo)
        sample_test = pdf_test[cols_features].sample(min(5000, len(pdf_test)))
        shap_values = explainer.shap_values(sample_test)
        
        shap_importance = pd.DataFrame({
            'feature': cols_features,
            'impacto_medio': np.abs(shap_values).mean(0)
        }).sort_values(by='impacto_medio', ascending=False)

        # 7. AVALIAÇÃO DE DESEMPENHO E EXPORTAÇÃO
        y_pred = modelo.predict(pdf_test[cols_features])
        mae = mean_absolute_error(pdf_test[target], y_pred)
        r2 = r2_score(pdf_test[target], y_pred)
        duracao = time.time() - inicio_processo

        print("Sincronizando artefatos preditivos com o Data Lake...", flush=True)
        modelo.save_model(self.modelo_local)
        with open(self.modelo_local, "rb") as f:
            self.s3.put_object(Bucket=self.bucket, Key=f"modelos/{self.modelo_local}", Body=f.read())
        
        shap_json = json.dumps(shap_importance.to_dict(orient='records'), indent=4)
        self.s3.put_object(Bucket=self.bucket, Key="modelos/SHAP_IMPORTANCE.json", Body=shap_json.encode())

        self.auditoria["metricas"] = {
            "linhas_originais_abt": linhas_originais,
            "linhas_pos_undersampling": linhas_balanceadas_treino,
            "r2_score_validacao": round(r2, 4),
            "mae_validacao": round(mae, 4),
            "tempo_execucao_segundos": round(duracao, 2),
            "top_features": shap_importance['feature'].head(10).tolist()
        }
        
        buf_json_auditoria = io.BytesIO(json.dumps(self.auditoria, indent=4).encode())
        self.s3.put_object(Bucket=self.bucket, Key="modelos/AUDITORIA_TREINO_CATBOOST.json", Body=buf_json_auditoria.getvalue())

        # 8. CONSOLIDAÇÃO DO RELATÓRIO DE AUDITORIA
        top_15 = shap_importance.head(15)
        resumo_shap_detalhado = "\n".join([f"   [{str(i+1).zfill(2)}] {r['feature'].ljust(30)} : {r['impacto_medio']:.4f}" for i, r in top_15.iterrows()])
        
        fs_impact = shap_importance[shap_importance['feature'].str.startswith('FS_')]['impacto_medio'].sum()
        infra_impact = shap_importance[shap_importance['feature'].str.startswith('MACRO_')]['impacto_medio'].sum()
        
        # Atualizado para contemplar a unificacao da FEAT_TIPO_DIA
        contexto_impact = shap_importance[shap_importance['feature'].isin([
            'SAZON_PERIODO', 'FEAT_PERFIL_VITIMA', 'FEAT_CONTEXTO_CRITICO', 
            'FEAT_TIPO_DIA', 'FEAT_DIA_SEMANA', 'FEAT_MES'
        ])]['impacto_medio'].sum()

        report = (
            f"Relatorio de Avaliacao do Modelo Preditivo\n"
            f"==============================================================\n"
            f"1. ESTATISTICA DA BASE DE DADOS\n"
            f"   - Ocorrencias Brutas          : {linhas_originais} registros\n"
            f"   - Treinamento (Balanceado)    : {linhas_balanceadas_treino} registros\n"
            f"   - Validacao (Mundo Real)      : {linhas_teste_real} registros\n\n"
            f"2. CONFIGURACAO DO ALGORITMO (CatBoostRegressor)\n"
            f"   - Funcao de Perda             : Expectile (alpha=0.85)\n"
            f"   - Profundidade (Depth)        : 8\n"
            f"   - Iteracoes Convergidas       : {modelo.tree_count_}\n\n"
            f"3. PERFORMANCE EM DADOS NAO VISTOS\n"
            f"   - R-Squared Score             : {r2:.4f}\n"
            f"   - Mean Absolute Error (MAE)   : {mae:.4f}\n\n"
            f"4. ANALISE GLOBAL DE IMPORTANCIA (Metodo SHAP)\n"
            f"   --- Fatores de Maior Contribuicao ---\n"
            f"{resumo_shap_detalhado}\n\n"
            f"   --- Impacto Agregado por Dimensao ---\n"
            f"   - Dinamica Temporal / Alvo    : {contexto_impact:.4f}\n"
            f"   - Historico Criminal Previo   : {fs_impact:.4f}\n"
            f"   - Distribuicao de Atividades  : {infra_impact:.4f}\n"
            f"==============================================================\n"
            f"Tempo de Processamento: {duracao/60:.2f} min\n"
            f"==============================================================\n"
        )
        print(report)
        self._notificar_discord(f"```text\n{report}\n```")
        
        if os.path.exists(self.modelo_local):
            os.remove(self.modelo_local)

if __name__ == "__main__":
    trainer = TreinadorSafeDriver()
    trainer.executar_treino()
