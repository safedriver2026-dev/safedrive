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
import gc
from datetime import datetime, date
from catboost import CatBoostRegressor
from botocore.config import Config
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import warnings

# Supressão de avisos para limpeza de log em produção
warnings.filterwarnings("ignore", category=FutureWarning)

class GeradorDossieSafeDriver:
    """
    ENGINE DE INTELIGÊNCIA GEOGRÁFICA PREDITIVA - PROJETO SAFEDRIVER
    ---------------------------------------------------------------
    Arquitetura: OBT (One Big Table) para BI de Alta Performance.
    Estatística: Tweedie Regression para modelagem de eventos raros (Zero-Inflated).
    Visão Operacional: "O B.O. é a evidência; a Malha é a Suscetibilidade".
    """
    
    def __init__(self):
        # Configurações de Cloud (R2/S3) via Variáveis de Ambiente - Boa prática de Segurança/ADS
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
        self.project_id = "safe-driver-fc3a9" # Identificador único do GCP/Billing
        
        # Auditoria de Processamento - Requisito de Governança
        self.auditoria = {
            "projeto": "SafeDriver",
            "ambiente": "Production",
            "data_inicio": str(datetime.now()),
            "versao_motor": "3.0.0-Tweedie-Equalized"
        }

    def _notificar_discord(self, msg):
        """Monitoramento em tempo real do pipeline."""
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except Exception: pass

    def gerar_dados(self):
        inicio_global = time.time()
        print(f"🛡️ [SAFEDRIVER] Iniciando Engine Preditiva - Project: {self.project_id}")

        # 1. CARGA DE MODELO E ARTEFATOS
        if not os.path.exists(self.modelo_local):
            print(f"📥 Cloud Sync: Baixando {self.modelo_local}...")
            self.s3.download_file(self.bucket, f"modelos/{self.modelo_local}", self.modelo_local)
        
        modelo = CatBoostRegressor().load_model(self.modelo_local)

        # 2. CARGA DA BASE OURO (Fatos Reais)
        # Uso de Polars para performance em grandes volumes (ADS Requirement)
        print("📥 I/O: Lendo DataLake Ouro...")
        obj = self.s3.get_object(Bucket=self.bucket, Key="datalake/ouro/safedriver_abt_treino.parquet")
        df_ouro = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # Normalização Temporal para o Biênio 2025-2026
        df_ouro = df_ouro.with_columns([
            pl.col("ANO_JOIN").cast(pl.Int32),
            pl.lit(False).alias("IS_MALHA") # Diferenciação entre Fato e Projeção
        ]).filter(pl.col("ANO_JOIN") >= 2025)

        # 3. CONSTRUÇÃO DA MALHA DE SUSCETIBILIDADE (Superfície Contínua)
        print("🔮 Arquitetura: Expandindo Malha Geo-Temporal...")
        
        # Preservação de Features Críticas para Storytelling
        features_geo = ["H3_INDEX", "LATITUDE", "LONGITUDE", "CIDADE", "BAIRRO", "LOGRADOURO"]
        features_fs = [c for c in df_ouro.columns if c.startswith("FS_") or c.startswith("MACRO_")]
        
        df_geo_base = df_ouro.select(features_geo + features_fs).unique(subset=["H3_INDEX"])

        # Definição de Cenários Operacionais (Manhã, Tarde, Noite, Madrugada)
        # Essencial para alocação de viaturas por turno (Segurança Pública)
        df_cenarios = pl.DataFrame({
            "SAZON_PERIODO": ["MANHA", "TARDE", "NOITE", "MADRUGADA"],
            "FEAT_TIPO_DIA": ["DIA_UTIL", "DIA_UTIL", "FIM_DE_SEMANA", "FIM_DE_SEMANA"]
        })

        # Janela de Projeção: Biênio Completo até Agosto de 2026
        datas_malha = [date(2025, m, 15) for m in range(1, 13)] + [date(2026, m, 15) for m in range(1, 9)]
        df_tempo = pl.DataFrame({"DATA_REF": datas_malha})

        # Cross Join para criar a Matriz de Risco Total
        df_malha = df_geo_base.join(df_cenarios, how="cross").join(df_tempo, how="cross")
        
        df_malha = df_malha.with_columns([
            pl.col("DATA_REF").cast(pl.Date).alias("DATAOCORRENCIA"),
            pl.col("DATA_REF").dt.year().cast(pl.Int32).alias("ANO_JOIN"),
            pl.col("DATA_REF").dt.month().alias("FEAT_MES"),
            pl.col("DATA_REF").dt.weekday().alias("FEAT_DIA_SEMANA"),
            pl.lit(True).alias("IS_MALHA"),
            pl.lit("PREVISÃO_IA").alias("RUBRICA")
        ]).drop("DATA_REF")

        # 4. CONSOLIDAÇÃO E LIMPEZA DE MEMÓRIA (ADS Best Practices)
        print("⚡ Data Wrangling: Unificando Grão de Dados...")
        cols_final = list(set(df_ouro.columns).intersection(set(df_malha.columns)))
        df_master = pl.concat([df_ouro.select(cols_final), df_malha.select(cols_final)], how="vertical")

        del df_ouro, df_malha, df_geo_base # Garbage Collection antecipada
        gc.collect()

        # 5. INFERÊNCIA TWEEDIE (O "Coração" Estatístico)
        print("🧠 ML: Executando Inferência Massiva (Tweedie Optimized)...")
        
        # Casting para strings para compatibilidade com CatBoost
        cat_features = [c for c in modelo.feature_names_ if df_master[c].dtype == pl.Utf8 or c == "H3_INDEX"]
        df_master = df_master.with_columns([pl.col(c).fill_null("DESCONHECIDO").cast(pl.Utf8) for c in cat_features])

        # Predição em batches para não estourar a RAM do container
        batch_size = 250000
        preds = []
        for i in range(0, df_master.height, batch_size):
            batch = df_master.slice(i, batch_size).select(modelo.feature_names_).to_pandas()
            preds.extend(modelo.predict(batch))

        # 6. ENGENHARIA DE KPIS PARA STORYTELLING (Visão Gestor)
        print("🏗️ BI: Calculando métricas fidedignas...")
        preds_raw = np.array(preds)
        volume_predito = np.maximum(preds_raw, 0.0)
        
        # Calibração Logarítmica de Risco (Escala 0.5 a 10)
        # Mais intuitivo para o policial: "Risco 10" é intervenção imediata.
        risco_log = np.log1p(volume_predito)
        p99 = np.percentile(risco_log, 99.9) or 1.0
        risco_final = np.clip(0.5 + (risco_log / p99) * 9.5, 0.5, 10.0)

        df_master = df_master.with_columns([
            pl.Series("VOLUME_TWEEDIE", volume_predito),
            pl.Series("RISCO_IA", risco_final).round(2)
        ])

        # 7. TAXONOMIA OPERACIONAL (K-Means Clustering)
        # Transforma matemática pura em "Categorias de Policiamento"
        print("🤖 Clusterização: Definindo Níveis de Alerta...")
        X_cluster = MinMaxScaler().fit_transform(df_master.select([
            pl.col("RISCO_IA"), pl.col("FS_VOL_CRIMES_ANO_ANT").fill_null(0.0)
        ]).to_numpy())

        km = KMeans(n_clusters=4, random_state=42, n_init="auto")
        clusters = km.fit_predict(X_cluster)
        
        # Ordenação: Cluster 3 sempre será o mais perigoso (Facilita o BI)
        rank = np.argsort(km.cluster_centers_[:, 0])
        map_rank = {v: i for i, v in enumerate(rank)}
        
        df_master = df_master.with_columns(
            pl.Series("CLUSTER_RANK", np.vectorize(map_rank.get)(clusters))
        )

        # 8. KPIS DE EVOLUÇÃO (Azeite do TCC)
        df_master = df_master.with_columns([
            # Risco fidedigno: Só mostra evolução em áreas que o K-Means validou como relevantes
            pl.when(pl.col("CLUSTER_RANK") >= 1)
            .then(pl.col("RISCO_IA"))
            .otherwise(pl.lit(None))
            .alias("KPI_RISCO_EVOLUCAO"),

            # Volume: B.O. (2025) vs Expectativa IA (2026)
            pl.when(pl.col("IS_MALHA") == False)
            .then(pl.lit(1.0)) # 1 BO Real
            .otherwise(
                pl.when(pl.col("ANO_JOIN") == 2026).then(pl.col("VOLUME_TWEEDIE")).otherwise(pl.lit(0.0))
            ).alias("KPI_VOLUME_TOTAL"),

            pl.when(pl.col("CLUSTER_RANK") == 3).then(pl.lit("🔴 ALERTA CRÍTICO"))
            .when(pl.col("CLUSTER_RANK") == 2).then(pl.lit("🟠 RISCO ALTO"))
            .when(pl.col("CLUSTER_RANK") == 1).then(pl.lit("🟡 ATENÇÃO MÉDIA"))
            .otherwise(pl.lit("🟢 ÁREA MONITORADA")).alias("STATUS_OPERACIONAL")
        ])

        # 9. OUTPUT E GOVERNANÇA
        print("📦 Cloud I/O: Sincronizando Datalake...")
        buf = io.BytesIO()
        df_master.write_parquet(buf, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dossie_eventos.parquet", Body=buf.getvalue())

        tempo_total = time.time() - inicio_global
        self._notificar_discord(f"🌐 **PROJETO SAFEDRIVER**\nDeploy concluído com sucesso.\nTempo: {tempo_total:.2f}s\nGrão: Equalizado (2025-2026)\nFoco: Storytelling de Evolução.")

if __name__ == "__main__":
    GeradorDossieSafeDriver().gerar_dados()
