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

# Silenciando o ruído visual
warnings.filterwarnings("ignore", category=FutureWarning)

class GeradorDossieSafeDriver:
    """
    ENGINE PREDITIVA (Totalmente Alinhada com a ABT Ouro).
    Respeita a linhagem de dados estruturais (Infra, Social e População).
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

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except Exception: pass

    def gerar_dados(self):
        inicio_global = time.time()
        print(f"🛡️ [SAFEDRIVER] Iniciando Engine Preditiva. Sincronizando com a Ouro...")

        # 1. DOWNLOAD E CARGA DO MODELO
        if not os.path.exists(self.modelo_local):
            print(f"📥 Baixando {self.modelo_local}...")
            self.s3.download_file(self.bucket, f"modelos/{self.modelo_local}", self.modelo_local)
        modelo = CatBoostRegressor().load_model(self.modelo_local)

        # 2. CARGA DA ABT OURO
        print("📥 I/O: Lendo DataLake Ouro...")
        obj = self.s3.get_object(Bucket=self.bucket, Key="datalake/ouro/safedriver_abt_treino.parquet")
        df_ouro = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # Correção de nomenclatura de tempo (compatibilidade com ArquitetoOuro)
        if "ANO_OCORRENCIA" in df_ouro.columns and "ANO_JOIN" not in df_ouro.columns:
            df_ouro = df_ouro.with_columns(pl.col("ANO_OCORRENCIA").alias("ANO_JOIN"))
        
        # Filtro de biênio
        df_ouro = df_ouro.filter(pl.col("ANO_JOIN") >= 2025)
        df_ouro = df_ouro.with_columns(pl.lit(False).alias("IS_MALHA"))

        # 3. CONSTRUÇÃO DA MALHA FUTURA (DNA ESPACIAL COMPLETO)
        print("🔮 Arquitetura: Clonando DNA Estrutural para Malha 2026...")
        
        # CORREÇÃO CRÍTICA: Captura TODAS as features necessárias para o CatBoost
        # Incluindo as colunas MICRO_ (População) e CENSO_
        features_estruturais = [
            c for c in df_ouro.columns 
            if c.startswith("INFRA_") 
            or c.startswith("FS_") 
            or c.startswith("MICRO_") 
            or c.startswith("CENSO_")
            or c in ["H3_INDEX", "LATITUDE", "LONGITUDE", "CIDADE", "BAIRRO", "LOGRADOURO"]
        ]
        
        # Foto única dos hexágonos (Dicionário de Features Estruturais)
        df_dna_hex = df_ouro.select(features_estruturais).unique(subset=["H3_INDEX"])

        # Cenários Operacionais
        df_cenarios = pl.DataFrame({
            "SAZON_PERIODO": ["MANHA", "TARDE", "NOITE", "MADRUGADA"],
            "FEAT_TIPO_DIA": ["DIA_UTIL", "DIA_UTIL", "FIM_DE_SEMANA", "FIM_DE_SEMANA"], 
            "FEAT_PERFIL_VITIMA": ["PEDESTRE", "MOTORISTA", "MOTORISTA", "PEDESTRE"]
        }).with_columns([
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO"),
            pl.when(pl.col("FEAT_TIPO_DIA") == "FIM_DE_SEMANA").then(pl.lit("SIM")).otherwise(pl.lit("NAO")).alias("FEAT_IS_FIM_DE_SEMANA")
        ])

        # Calendário Preditivo (Jan/2025 a Ago/2026)
        datas_malha = [date(2025, m, 15) for m in range(1, 13)] + [date(2026, m, 15) for m in range(1, 9)]
        df_tempo = pl.DataFrame({"DATA_REF": datas_malha})

        # Expansão da Malha
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

        # 4. UNIFICAÇÃO
        print("⚡ Data Wrangling: Unificando Universo...")
        cols_comuns = list(set(df_ouro.columns).intersection(set(df_malha.columns)))
        df_master = pl.concat([df_ouro.select(cols_comuns), df_malha.select(cols_comuns)], how="vertical")

        del df_ouro, df_malha, df_dna_hex
        gc.collect()

        # 5. INFERÊNCIA MASSIVA TWEEDIE
        print("🧠 ML: Executando Inferência Massiva...")
        
        # Casting de segurança para o CatBoost
        cat_features = [c for c in modelo.feature_names_ if c in df_master.columns]
        df_master = df_master.with_columns([pl.col(c).fill_null("DESCONHECIDO").cast(pl.Utf8) for c in cat_features])

        batch_size = 250000
        preds = []
        for i in range(0, df_master.height, batch_size):
            # O select agora garante a presença das colunas MICRO_POPULACAO e outras estruturais
            batch = df_master.slice(i, batch_size).select(modelo.feature_names_).to_pandas()
            preds.extend(modelo.predict(batch))

        volume_predito = np.maximum(np.array(preds), 0.0)
        
        # 6. ENGENHARIA DE RISCO (LOGARÍTMICA)
        risco_log = np.log1p(volume_predito)
        p99 = np.percentile(risco_log, 99.9) or 1.0
        risco_final = np.clip(0.5 + (risco_log / p99) * 9.5, 0.5, 10.0)

        df_master = df_master.with_columns([
            pl.Series("VOLUME_TWEEDIE", volume_predito),
            pl.Series("RISCO_IA", risco_final).round(2)
        ])

        # 7. CLUSTERIZAÇÃO OPERACIONAL
        print("🤖 Clusterização K-Means (Risco vs Volume)...")
        # Preenche nulos estruturais para o clusterizador não quebrar
        X_cluster = MinMaxScaler().fit_transform(df_master.select([
            pl.col("RISCO_IA"), 
            pl.col("FS_VOL_CRIMES_ANO_ANT").fill_null(0.0)
        ]).to_numpy())

        km = KMeans(n_clusters=4, random_state=42, n_init="auto")
        clusters = km.fit_predict(X_cluster)
        rank = np.argsort(km.cluster_centers_[:, 0])
        map_rank = {v: i for i, v in enumerate(rank)}
        df_master = df_master.with_columns(pl.Series("CLUSTER_RANK", np.vectorize(map_rank.get)(clusters)))

        # 8. KPIS DE STORYTELLING
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

        # 9. EXPORTAÇÃO
        print("📦 Cloud I/O: Sincronizando Datalake...")
        buf = io.BytesIO()
        df_master.write_parquet(buf, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dossie_eventos.parquet", Body=buf.getvalue())

        print(f"✅ Pipeline concluído em {time.time() - inicio_global:.2f}s")
        self._notificar_discord("🚀 **MOTOR SAFEDRIVER**\nMalha Inteligente gerada. DNA Populacional e Estrutural preservado.")

if __name__ == "__main__":
    GeradorDossieSafeDriver().gerar_dados()
