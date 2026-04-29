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
    ENGINE PREDITIVA (Totalmente Alinhada com a ABT Ouro).
    Arquitetura de respeito à engenharia de features feita na camada anterior.
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
        print(f"🛡️ [SAFEDRIVER] Iniciando Engine Preditiva. Lendo a verdade da Ouro...")

        # 1. DOWNLOAD DO MODELO
        if not os.path.exists(self.modelo_local):
            self.s3.download_file(self.bucket, f"modelos/{self.modelo_local}", self.modelo_local)
        modelo = CatBoostRegressor().load_model(self.modelo_local)

        # 2. CARGA DA ABT OURO 
        # (Aqui nós pegamos toda a inteligência que o seu ArquitetoOuro construiu)
        obj = self.s3.get_object(Bucket=self.bucket, Key="datalake/ouro/safedriver_abt_treino.parquet")
        df_ouro = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # Filtro de início da série histórica (2025)
        df_ouro = df_ouro.filter(pl.col("ANO_OCORRENCIA") >= 2025)
        
        # Padroniza a flag de controle
        df_ouro = df_ouro.with_columns([
            pl.col("ANO_OCORRENCIA").cast(pl.Int32).alias("ANO_JOIN"),
            pl.lit(False).alias("IS_MALHA")
        ])

        # 3. CONSTRUÇÃO DA MALHA FUTURA (Apoiada no DNA da Ouro)
        # O segredo é: o hexágono carrega TODAS as features estruturais do IBGE e Histórico
        features_estruturais = [
            c for c in df_ouro.columns 
            if c.startswith("INFRA_") or c.startswith("FS_") or c in [
                "H3_INDEX", "LATITUDE", "LONGITUDE", "CIDADE", "BAIRRO", "LOGRADOURO"
            ]
        ]
        
        print("🔮 Arquitetura: Expandindo Malha usando o DNA Espacial da Ouro...")
        # Pegamos uma "foto" única de cada hexágono, com as features mais recentes (último ano)
        df_dna_hex = df_ouro.sort("DATAOCORRENCIA", descending=True).select(features_estruturais).unique(subset=["H3_INDEX"], keep="first")

        # Recria as opções de cenário que o Policial pode encontrar
        df_cenarios = pl.DataFrame({
            "SAZON_PERIODO": ["MANHA", "TARDE", "NOITE", "MADRUGADA"],
            "FEAT_TIPO_DIA": ["DIA_UTIL", "DIA_UTIL", "FIM_DE_SEMANA", "FIM_DE_SEMANA"], 
            "FEAT_PERFIL_VITIMA": ["PEDESTRE", "MOTORISTA", "MOTORISTA", "PEDESTRE"]
        }).with_columns([
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO"),
            pl.when(pl.col("FEAT_TIPO_DIA") == "FIM_DE_SEMANA").then(pl.lit("SIM")).otherwise(pl.lit("NAO")).alias("FEAT_IS_FIM_DE_SEMANA")
        ])

        # O Calendário Preditivo Equalizado
        datas_malha = [date(2025, m, 15) for m in range(1, 13)] + [date(2026, m, 15) for m in range(1, 9)]
        df_tempo = pl.DataFrame({"DATA_REF": datas_malha})

        # O Big Bang da Malha: Hexágono * Cenários * Meses
        df_malha = df_dna_hex.join(df_cenarios, how="cross").join(df_tempo, how="cross")
        
        # Injeta as features temporais baseadas no mês projetado
        df_malha = df_malha.with_columns([
            pl.col("DATA_REF").cast(pl.Date).alias("DATAOCORRENCIA"),
            pl.col("DATA_REF").dt.year().cast(pl.Int32).alias("ANO_JOIN"),
            pl.col("DATA_REF").dt.year().cast(pl.Int32).alias("ANO_OCORRENCIA"),
            pl.col("DATA_REF").dt.month().alias("FEAT_MES"),
            pl.col("DATA_REF").dt.weekday().alias("FEAT_DIA_SEMANA"),
            ((pl.col("DATA_REF").dt.year() * 12) + pl.col("DATA_REF").dt.month()).alias("MES_ABSOLUTO"),
            pl.lit(True).alias("IS_MALHA"),
            pl.lit("PREVISÃO_IA").alias("RUBRICA"),
            pl.lit(0.0).alias("LABEL_PESO_RISCO")
        ]).drop("DATA_REF")

        # 4. UNIFICAÇÃO DO UNIVERSO (Treino + Malha)
        print("⚡ Data Wrangling: Unificando Universo...")
        # Usa APENAS as colunas que importam para o modelo e para o Looker
        cols_vitais = list(set(df_ouro.columns).intersection(set(df_malha.columns)))
        df_master = pl.concat([df_ouro.select(cols_vitais), df_malha.select(cols_vitais)], how="vertical")

        del df_ouro, df_malha, df_dna_hex
        gc.collect()

        # 5. PREDIÇÃO MASSIVA TWEEDIE
        print("🧠 ML: Executando Inferência Massiva...")
        # Casting cego: Transforma todas as features categóricas que o modelo quer em String.
        # Isso evita o chato do CatBoost reclamar que uma coluna está como Float.
        cat_features = [c for c in modelo.feature_names_ if c in df_master.columns]
        df_master = df_master.with_columns([pl.col(c).fill_null("DESCONHECIDO").cast(pl.Utf8) for c in cat_features])

        batch_size = 250000
        preds = []
        for i in range(0, df_master.height, batch_size):
            # Envia exatamente as colunas na ordem que o modelo foi treinado
            batch = df_master.slice(i, batch_size).select(modelo.feature_names_).to_pandas()
            preds.extend(modelo.predict(batch))

        volume_predito = np.maximum(np.array(preds), 0.0)
        
        # 6. ENGENHARIA DE RISCO (LOG) E K-MEANS
        risco_log = np.log1p(volume_predito)
        p99 = np.percentile(risco_log, 99.9) or 1.0
        risco_final = np.clip(0.5 + (risco_log / p99) * 9.5, 0.5, 10.0)

        df_master = df_master.with_columns([
            pl.Series("VOLUME_TWEEDIE", volume_predito),
            pl.Series("RISCO_IA", risco_final).round(2)
        ])

        print("🤖 Clusterização K-Means (Risco vs Volume IBGE)...")
        X_cluster = MinMaxScaler().fit_transform(df_master.select([
            pl.col("RISCO_IA"), pl.col("FS_VOL_CRIMES_ANO_ANT").fill_null(0.0)
        ]).to_numpy())

        km = KMeans(n_clusters=4, random_state=42, n_init="auto")
        clusters = km.fit_predict(X_cluster)
        
        map_rank = {v: i for i, v in enumerate(np.argsort(km.cluster_centers_[:, 0]))}
        df_master = df_master.with_columns(pl.Series("CLUSTER_RANK", np.vectorize(map_rank.get)(clusters)))

        # 7. KPIS DE STORYTELLING PARA O LOOKER STUDIO
        print("🏗️ BI: Montando as Tabelas Finais...")
        df_master = df_master.with_columns([
            # Linha Contínua: Só os clusters perigosos 
            pl.when(pl.col("CLUSTER_RANK") >= 1).then(pl.col("RISCO_IA")).otherwise(pl.lit(None)).alias("KPI_RISCO_EVOLUCAO"),
            
            # Barras: O Fato (2025) e a Suscetibilidade (2026)
            pl.when(pl.col("IS_MALHA") == False).then(pl.lit(1.0)).otherwise(
                pl.when(pl.col("ANO_JOIN") == 2026).then(pl.col("VOLUME_TWEEDIE")).otherwise(pl.lit(0.0))
            ).alias("KPI_VOLUME_TOTAL"),
            
            # Taxonomia Policial
            pl.when(pl.col("CLUSTER_RANK") == 3).then(pl.lit("🔴 ALERTA CRÍTICO"))
            .when(pl.col("CLUSTER_RANK") == 2).then(pl.lit("🟠 RISCO ALTO"))
            .when(pl.col("CLUSTER_RANK") == 1).then(pl.lit("🟡 ATENÇÃO MÉDIA"))
            .otherwise(pl.lit("🟢 ÁREA MONITORADA")).alias("STATUS_OPERACIONAL")
        ])

        # 8. SHAP (DNA CRIMINAL) E EXPORT
        print("🧬 Genética do Crime: Gerando SHAP...")
        df_sample = df_master.filter(pl.col("CLUSTER_RANK") >= 2).sample(n=min(5000, df_master.height))
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(df_sample.select(modelo.feature_names_).to_pandas())
        
        df_shap = pd.DataFrame(shap_values, columns=[f"DNA_{c}" for c in modelo.feature_names_])
        df_shap[['BAIRRO', 'CIDADE']] = df_sample.select(['BAIRRO', 'CIDADE']).to_pandas()
        df_dna_final = df_shap.groupby(['CIDADE', 'BAIRRO']).mean().reset_index()

        print("📦 Cloud I/O: Exportando para o Lake...")
        buf_master = io.BytesIO()
        df_master.write_parquet(buf_master, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dossie_eventos.parquet", Body=buf_master.getvalue())
        
        buf_dna = io.BytesIO()
        pl.from_pandas(df_dna_final).write_parquet(buf_dna, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dim_dna_shap.parquet", Body=buf_dna.getvalue())

        print(f"✅ Pipeline Preditivo concluído em {time.time() - inicio_global:.2f}s")
        self._notificar_discord("🚀 **MOTOR SAFEDRIVER**\nMalha Inteligente gerada com sucesso e baseada 100% no DNA da Camada Ouro.")

if __name__ == "__main__":
    GeradorDossieSafeDriver().gerar_dados()
