import os
import boto3
import polars as pl
import io
import pandas as pd
import shap
import time
import requests
import numpy as np
import gc
from datetime import datetime, date
from catboost import CatBoostRegressor
from botocore.config import Config
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class GeradorDossieSafeDriver:
    """
    Motor de Inferência Preditiva SafeDriver.
    Realiza a projeção de malha espacial e enforce de tipagem para o modelo CatBoost.
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
        self.modelo_local = "modelo_safedriver_catboost.cbm"

    def _notificar_webhook(self, msg):
        """Dispara notificação de status de execução."""
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except Exception: pass

    def gerar_dados(self):
        inicio_global = time.time()
        print("[SISTEMA] Iniciando inferência preditiva (Tweedie Regression).")

        # 1. Download do artefato do modelo
        if not os.path.exists(self.modelo_local):
            self.s3.download_file(self.bucket, f"modelos/{self.modelo_local}", self.modelo_local)
        modelo = CatBoostRegressor().load_model(self.modelo_local)

        # 2. Carga dos dados históricos
        obj = self.s3.get_object(Bucket=self.bucket, Key="datalake/ouro/safedriver_abt_treino.parquet")
        df_ouro = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # Filtro temporal e definição de sinalizador de origem
        df_ouro = df_ouro.filter(pl.col("ANO_OCORRENCIA") >= 2025).with_columns([
            pl.col("ANO_OCORRENCIA").cast(pl.Int32).alias("ANO_JOIN"),
            pl.lit(False).alias("IS_MALHA")
        ])

        # 3. Geração da malha sintética futura
        features_estruturais = [c for c in df_ouro.columns if any(c.startswith(pre) for pre in ["INFRA_", "FS_", "CENSO_", "MICRO_"])]
        features_geo = ["H3_INDEX", "LATITUDE", "LONGITUDE", "CIDADE", "BAIRRO", "LOGRADOURO"]
        
        # Extração de localizações únicas
        df_dna_hex = df_ouro.select(features_geo + features_estruturais).unique(subset=["H3_INDEX"])

        # Definição dos cenários de predição (CORRIGIDO)
        df_cenarios = pl.DataFrame({
            "SAZON_PERIODO": ["MANHA", "TARDE", "NOITE", "MADRUGADA"],
            "FEAT_TIPO_DIA": ["DIA_UTIL", "DIA_UTIL", "FIM_DE_SEMANA", "FIM_DE_SEMANA"], 
            "FEAT_PERFIL_VITIMA": ["PEDESTRE", "MOTORISTA", "MOTORISTA", "PEDESTRE"]
        }).with_columns([
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO"),
            pl.when(pl.col("FEAT_TIPO_DIA") == "FIM_DE_SEMANA").then(pl.lit("SIM")).otherwise(pl.lit("NAO")).alias("FEAT_IS_FIM_DE_SEMANA")
        ])

        # Geração do vetor temporal
        datas_malha = [date(2025, m, 15) for m in range(1, 13)] + [date(2026, m, 15) for m in range(1, 9)]
        df_tempo = pl.DataFrame({"DATA_REF": datas_malha})

        # Produto cartesiano espacial-temporal
        df_malha = df_dna_hex.join(df_cenarios, how="cross").join(df_tempo, how="cross")
        df_malha = df_malha.with_columns([
            pl.col("DATA_REF").cast(pl.Date).alias("DATAOCORRENCIA"),
            pl.col("DATA_REF").dt.year().cast(pl.Int32).alias("ANO_JOIN"),
            pl.col("DATA_REF").dt.month().alias("FEAT_MES"),
            pl.col("DATA_REF").dt.weekday().alias("FEAT_DIA_SEMANA"),
            ((pl.col("DATA_REF").dt.year() * 12) + pl.col("DATA_REF").dt.month()).alias("MES_ABSOLUTO"),
            pl.lit(True).alias("IS_MALHA"),
            pl.lit("PREVISÃO_IA").alias("RUBRICA"),
            pl.lit(0.0).alias("LABEL_PESO_RISCO")
        ]).drop("DATA_REF")

        # 4. Concatenação vertical
        cols_modelo = modelo.feature_names_
        cols_comuns = list(set(df_ouro.columns).intersection(set(df_malha.columns)))
        df_master = pl.concat([df_ouro.select(cols_comuns), df_malha.select(cols_comuns)], how="diagonal")

        del df_ouro, df_malha, df_dna_hex
        gc.collect()

        # 5. Tratamento rigoroso de tipagem categórica
        cat_features_modelo = [cols_modelo[i] for i in modelo.get_cat_feature_indices()]
        print(f"[PROCESSAMENTO] Realizando casting restrito em {len(cat_features_modelo)} colunas categóricas.")

        pdf_master = df_master.to_pandas()
        for col in cols_modelo:
            if col in cat_features_modelo:
                pdf_master[col] = pdf_master[col].fillna('DESCONHECIDO').astype(str).str.replace(r'\.0$', '', regex=True).replace(['nan', 'NaN', 'None', '<NA>', ''], 'DESCONHECIDO').astype(object)
            else:
                pdf_master[col] = pdf_master[col].fillna(0.0).astype(float)

        # 6. Inferência preditiva em lotes
        print("[SISTEMA] Executando inferência massiva no vetor de dados.")
        batch_size = 250000
        preds = []
        for i in range(0, len(pdf_master), batch_size):
            batch = pdf_master.iloc[i : i + batch_size][cols_modelo]
            preds.extend(modelo.predict(batch))

        # 7. Engenharia da variável de risco
        volume_predito = np.maximum(np.array(preds), 0.0)
        risco_log = np.log1p(volume_predito)
        p99 = np.percentile(risco_log, 99.9) or 1.0
        risco_final = np.clip(0.5 + (risco_log / p99) * 9.5, 0.5, 10.0)

        pdf_master["VOLUME_TWEEDIE"] = volume_predito
        pdf_master["RISCO_IA"] = np.round(risco_final, 2)

        # Classificação por k-Means multivariado
        X_cluster = MinMaxScaler().fit_transform(pdf_master[["RISCO_IA", "FS_VOL_CRIMES_ANO_ANT"]].fillna(0))
        km = KMeans(n_clusters=4, random_state=42, n_init="auto")
        clusters = km.fit_predict(X_cluster)
        map_rank = {v: i for i, v in enumerate(np.argsort(km.cluster_centers_[:, 0]))}
        pdf_master["CLUSTER_RANK"] = np.vectorize(map_rank.get)(clusters)

        # Derivação de indicadores
        pdf_master["KPI_RISCO_EVOLUCAO"] = np.where(pdf_master["CLUSTER_RANK"] >= 1, pdf_master["RISCO_IA"], np.nan)
        pdf_master["KPI_VOLUME_TOTAL"] = np.where(pdf_master["IS_MALHA"] == False, 1.0, 
                                                 np.where(pdf_master["ANO_JOIN"] == 2026, pdf_master["VOLUME_TWEEDIE"], 0.0))
        
        status_map = {3: "ALERTA CRITICO", 2: "RISCO ALTO", 1: "ATENCAO MEDIA", 0: "AREA MONITORADA"}
        pdf_master["STATUS_OPERACIONAL"] = pdf_master["CLUSTER_RANK"].map(status_map)

        # 8. Extração de explicabilidade global (SHAP) agregada por Cidade, Bairro e Logradouro
        print("[PROCESSAMENTO] Calculando tensores SHAP preditivos para a malha total.")
        explainer = shap.TreeExplainer(modelo)
        
        # Inovação: Calcular SHAP sobre um lote representativo de TODA a malha preditiva
        # Isso garante que mesmo locais sem histórico (Volume Histórico = 0) recebam a carga 
        # explicativa de "por que a IA acha que ali vai ter crime" (ex: Contexto do Entorno, Infraestrutura).
        tamanho_amostra_shap = min(200000, len(pdf_master))
        pdf_amostra_shap = pdf_master.sample(tamanho_amostra_shap, random_state=42)
        
        valores_shap = explainer.shap_values(pdf_amostra_shap[cols_modelo])
        df_shap = pd.DataFrame(np.abs(valores_shap), columns=[f"DNA_{c}" for c in cols_modelo])
        
        # Injeção das chaves geográficas e re-agregação
        df_shap['CIDADE'] = pdf_amostra_shap['CIDADE'].values
        df_shap['BAIRRO'] = pdf_amostra_shap['BAIRRO'].values
        df_shap['LOGRADOURO'] = pdf_amostra_shap['LOGRADOURO'].values
        
        df_dna_geografico = df_shap.groupby(['CIDADE', 'BAIRRO', 'LOGRADOURO']).mean().reset_index()

        # 9. Integração Cloud Storage
        print("[SISTEMA] Exportando matrizes de dados (Eventos e Dimensão SHAP Expandida).")
        
        buf_eventos = io.BytesIO()
        pdf_master.to_parquet(buf_eventos, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dossie_eventos.parquet", Body=buf_eventos.getvalue())

        buf_dna = io.BytesIO()
        df_dna_geografico.to_parquet(buf_dna, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dim_dna_cidade.parquet", Body=buf_dna.getvalue())

        tempo_execucao = time.time() - inicio_global
        self._notificar_webhook(f"[INFO] Pipeline Preditivo finalizado em {tempo_execucao:.2f} segundos. Cobertura SHAP estendida para malha preditiva.")

if __name__ == "__main__":
    GeradorDossieSafeDriver().gerar_dados()
