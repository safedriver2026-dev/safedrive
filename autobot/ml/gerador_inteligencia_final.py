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
    Estatística: Tweedie Regression para modelagem de eventos raros.
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
        self.project_id = "safe-driver-fc3a9"
        
        self.auditoria = {
            "projeto": "SafeDriver",
            "ambiente": "Production",
            "data_inicio": str(datetime.now()),
            "versao_motor": "3.2.0-Fix-Features"
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except Exception: pass

    def gerar_dados(self):
        inicio_global = time.time()
        print(f"🛡️ [SAFEDRIVER] Iniciando Engine Preditiva - Project: {self.project_id}")

        # 1. CARGA DE MODELO
        if not os.path.exists(self.modelo_local):
            print(f"📥 Cloud Sync: Baixando {self.modelo_local}...")
            self.s3.download_file(self.bucket, f"modelos/{self.modelo_local}", self.modelo_local)
        
        modelo = CatBoostRegressor().load_model(self.modelo_local)

        # 2. CARGA DA BASE OURO + GERAÇÃO DE FEATURES FALTANTES
        print("📥 I/O: Lendo DataLake Ouro...")
        obj = self.s3.get_object(Bucket=self.bucket, Key="datalake/ouro/safedriver_abt_treino.parquet")
        df_ouro = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # Correção de ANO_JOIN e geração de colunas temporais exigidas pelo modelo
        df_ouro = df_ouro.with_columns([
            pl.col("DATAOCORRENCIA").dt.year().cast(pl.Int32).alias("ANO_JOIN"),
            pl.col("DATAOCORRENCIA").dt.weekday().alias("FEAT_DIA_SEMANA"),
            pl.col("DATAOCORRENCIA").dt.month().alias("FEAT_MES")
        ])

        # Criação da FEAT_IS_FIM_DE_SEMANA (Sábado=6, Domingo=7 no Polars)
        df_ouro = df_ouro.with_columns([
            pl.when(pl.col("FEAT_DIA_SEMANA").is_in([6, 7]))
            .then(pl.lit("SIM"))
            .otherwise(pl.lit("NAO"))
            .alias("FEAT_IS_FIM_DE_SEMANA"),
            pl.lit(False).alias("IS_MALHA"),
            pl.lit("MOTORISTA").alias("FEAT_PERFIL_VITIMA") # Fallback para o modelo
        ]).with_columns([
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        ])

        df_ouro = df_ouro.filter(pl.col("ANO_JOIN") >= 2025)

        # 3. CONSTRUÇÃO DA MALHA DE SUSCETIBILIDADE (2026)
        print("🔮 Arquitetura: Expandindo Malha Geo-Temporal...")
        features_geo = ["H3_INDEX", "LATITUDE", "LONGITUDE", "CIDADE", "BAIRRO", "LOGRADOURO"]
        features_fs = [c for c in df_ouro.columns if c.startswith("FS_") or c.startswith("MACRO_")]
        
        df_geo_base = df_ouro.select(features_geo + features_fs).unique(subset=["H3_INDEX"])

        df_cenarios = pl.DataFrame({
            "SAZON_PERIODO": ["MANHA", "TARDE", "NOITE", "MADRUGADA"],
            "FEAT_TIPO_DIA": ["DIA_UTIL", "DIA_UTIL", "FIM_DE_SEMANA", "FIM_DE_SEMANA"]
        })

        datas_malha = [date(2025, m, 15) for m in range(1, 13)] + [date(2026, m, 15) for m in range(1, 9)]
        df_tempo = pl.DataFrame({"DATA_REF": datas_malha})

        df_malha = df_geo_base.join(df_cenarios, how="cross").join(df_tempo, how="cross")
        
        df_malha = df_malha.with_columns([
            pl.col("DATA_REF").cast(pl.Date).alias("DATAOCORRENCIA"),
            pl.col("DATA_REF").dt.year().cast(pl.Int32).alias("ANO_JOIN"),
            pl.col("DATA_REF").dt.month().alias("FEAT_MES"),
            pl.col("DATA_REF").dt.weekday().alias("FEAT_DIA_SEMANA"),
            pl.lit(True).alias("IS_MALHA"),
            pl.lit("PREVISÃO_IA").alias("RUBRICA"),
            pl.lit("MOTORISTA").alias("FEAT_PERFIL_VITIMA")
        ]).with_columns([
            pl.when(pl.col("FEAT_DIA_SEMANA").is_in([6, 7]))
            .then(pl.lit("SIM"))
            .otherwise(pl.lit("NAO"))
            .alias("FEAT_IS_FIM_DE_SEMANA"),
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        ]).drop("DATA_REF")

        # 4. CONSOLIDAÇÃO
        print("⚡ Data Wrangling: Unificando Grão de Dados...")
        cols_modelo = modelo.feature_names_
        # Garantir que todas as colunas do modelo existem na master
        cols_final = list(set(df_ouro.columns).intersection(set(df_malha.columns)))
        df_master = pl.concat([df_ouro.select(cols_final), df_malha.select(cols_final)], how="vertical")

        del df_ouro, df_malha, df_geo_base
        gc.collect()

        # 5. INFERÊNCIA MASSIVA
        print("🧠 ML: Executando Inferência Massiva...")
        # Casting para strings (requisito CatBoost para features categóricas)
        df_master = df_master.with_columns([
            pl.col(c).fill_null("DESCONHECIDO").cast(pl.Utf8) 
            for c in cols_modelo if c in df_master.columns and df_master[c].dtype != pl.Float64
        ])

        batch_size = 250000
        preds = []
        for i in range(0, df_master.height, batch_size):
            # O select agora vai bater com os feature_names do modelo
            batch = df_master.slice(i, batch_size).select(cols_modelo).to_pandas()
            preds.extend(modelo.predict(batch))

        # 6. ENGENHARIA DE KPIS
        print("🏗️ BI: Calculando métricas de risco...")
        preds_raw = np.array(preds)
        volume_predito = np.maximum(preds_raw, 0.0)
        
        # Calibração Logarítmica de Risco: $R = 0.5 + (\frac{\ln(1+V)}{p_{99.9}} \times 9.5)$
        risco_log = np.log1p(volume_predito)
        p99 = np.percentile(risco_log, 99.9) or 1.0
        risco_final = np.clip(0.5 + (risco_log / p99) * 9.5, 0.5, 10.0)

        df_master = df_master.with_columns([
            pl.Series("VOLUME_TWEEDIE", volume_predito),
            pl.Series("RISCO_IA", risco_final).round(2)
        ])

        # 7. CLUSTERIZAÇÃO K-MEANS
        print("🤖 Clusterização: Definindo Níveis de Alerta...")
        X_cluster = MinMaxScaler().fit_transform(df_master.select([
            pl.col("RISCO_IA"), pl.col("FS_VOL_CRIMES_ANO_ANT").fill_null(0.0)
        ]).to_numpy())

        km = KMeans(n_clusters=4, random_state=42, n_init="auto")
        clusters = km.fit_predict(X_cluster)
        rank = np.argsort(km.cluster_centers_[:, 0])
        map_rank = {v: i for i, v in enumerate(rank)}
        
        df_master = df_master.with_columns(
            pl.Series("CLUSTER_RANK", np.vectorize(map_rank.get)(clusters))
        )

        # 8. KPIS DEStorytelling (EQUALIZADOS)
        df_master = df_master.with_columns([
            pl.when(pl.col("CLUSTER_RANK") >= 1).then(pl.col("RISCO_IA")).otherwise(pl.lit(None)).alias("KPI_RISCO_EVOLUCAO"),
            pl.when(pl.col("IS_MALHA") == False).then(pl.lit(1.0)).otherwise(
                pl.when(pl.col("ANO_JOIN") == 2026).then(pl.col("VOLUME_TWEEDIE")).otherwise(pl.lit(0.0))
            ).alias("KPI_VOLUME_TOTAL"),
            pl.when(pl.col("CLUSTER_RANK") == 3).then(pl.lit("🔴 ALERTA CRÍTICO"))
            .when(pl.col("CLUSTER_RANK") == 2).then(pl.lit("🟠 RISCO ALTO"))
            .when(pl.col("CLUSTER_RANK") == 1).then(pl.lit("🟡 ATENÇÃO MÉDIA"))
            .otherwise(pl.lit("🟢 ÁREA MONITORADA")).alias("STATUS_OPERACIONAL")
        ])

        # 10. OUTPUT FINAL
        print("📦 Cloud I/O: Sincronizando Datalake...")
        buf_master = io.BytesIO()
        df_master.write_parquet(buf_master, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dossie_eventos.parquet", Body=buf_master.getvalue())

        tempo_total = time.time() - inicio_global
        print(f"✅ Sucesso: Pipeline SafeDriver concluído em {tempo_total:.2f}s")
        self._notificar_discord(f"🚀 **SAFEDRIVER: DEPLOY FINALIZADO**\n\nFeatures: Corrigidas\nDNA Criminal: Gerado\nStatus: Pronto.")

if __name__ == "__main__":
    GeradorDossieSafeDriver().gerar_dados()
