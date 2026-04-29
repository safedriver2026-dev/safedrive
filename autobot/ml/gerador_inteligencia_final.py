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
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class GeradorDossieSafeDriver:
    """
    Motor de Inteligência Preditiva (Versão Otimizada BQ Free).
    Gera Dossiê Focado no Biênio (2025 e 2026).
    Usa Cenários Macro e recuo de 4 meses para poupar limites do BigQuery.
    Proteção nativa contra vazamento de RAM (Exit Code 137).
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
        
        self.auditoria = {
            "projeto": "SafeDriver",
            "fase": "Dossiê de Inteligência Geográfica (Otimizado/Anti-OOM)",
            "data_processamento": str(datetime.now()),
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except: pass

    def gerar_dados(self):
        inicio_processo = time.time()
        print("🧠 [DOSSIÊ] Iniciando motor de inteligência...", flush=True)
        
        # 1. DOWNLOAD E CARGA DO MODELO
        if not os.path.exists(self.modelo_local):
            print(f"📥 Baixando {self.modelo_local} do bucket...", flush=True)
            self.s3.download_file(self.bucket, f"modelos/{self.modelo_local}", self.modelo_local)
        
        modelo = CatBoostRegressor().load_model(self.modelo_local)
        
        # 2. CARGA DA ABT OURO E CORTE TEMPORAL (Foco 2025)
        print("📥 Lendo a base Ouro para inferência massiva...", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key="datalake/ouro/safedriver_abt_treino.parquet")
        df_ouro_raw = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # O Foco Laser: Corta tudo antes de 2025
        if "ANO_JOIN" in df_ouro_raw.columns:
            df_ouro = df_ouro_raw.filter(pl.col("ANO_JOIN") >= 2025)
        else:
            df_ouro = df_ouro_raw
            
        total_historico = df_ouro.height

        # 3. GERAÇÃO DA MALHA FUTURA (Foco em Eficiência Financeira / BQ Free)
        print("🔮 Projetando cenários macro-futuros (Otimização para BQ Free)...", flush=True)
        
        colunas_preservadas = [c for c in df_ouro.columns if c in [
            "H3_INDEX", "LATITUDE", "LONGITUDE", "CIDADE", "BAIRRO", "LOGRADOURO", "RUA", "RUBRICA", "ANO_JOIN",
            "MICRO_POPULACAO_FACES", "CENSO_MEDIA_V0001", "CENSO_MEDIA_V0002"
        ] or c.startswith("MACRO_") or c.startswith("FS_")]
        
        df_dna_geografico = df_ouro.select(colunas_preservadas).unique(subset=["H3_INDEX"])

        # OTIMIZAÇÃO DE CUSTO 1: Cenários Macro (Reduz de 16 para 4 cruzamentos por hexágono)
        df_cenarios = pl.DataFrame({
            "SAZON_PERIODO": ["MANHA", "TARDE", "NOITE", "MADRUGADA"],
            "FEAT_TIPO_DIA": ["DIA_UTIL", "DIA_UTIL", "FIM_DE_SEMANA", "FIM_DE_SEMANA"], # Agrupamento representativo
            "FEAT_PERFIL_VITIMA": ["PEDESTRE", "MOTORISTA", "MOTORISTA", "PEDESTRE"] # Agrupamento representativo
        }).with_columns([
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO"),
            pl.when(pl.col("FEAT_TIPO_DIA") == "FIM_DE_SEMANA").then(pl.lit("SIM")).otherwise(pl.lit("NAO")).alias("FEAT_IS_FIM_DE_SEMANA")
        ])

        # OTIMIZAÇÃO DE CUSTO 2: Projeta os próximos 4 meses da linha de tendência (Em vez de 12)
        meses_alvo = [1, 2, 3, 4] 
        df_meses_futuro = pl.DataFrame({
            "DATA_REF_MES": [date(2026, mes, 15) for mes in meses_alvo]
        })

        # Funde o Mapa (H3) com os Cenários (Macro) e os Meses (Curtos)
        df_futuro = df_dna_geografico.join(df_cenarios, how="cross").join(df_meses_futuro, how="cross")
        
        df_futuro = df_futuro.with_columns([
            pl.col("DATA_REF_MES").cast(pl.Date).alias("DATAOCORRENCIA"),
            pl.lit(0.0).alias("LABEL_PESO_RISCO"),
            pl.lit("PREVISÃO_MACRO").alias("RUBRICA"), # Marca que é uma previsão otimizada
            pl.lit(2026).cast(pl.Int32).alias("ANO_JOIN"),
            pl.col("DATA_REF_MES").dt.month().alias("FEAT_MES"),
            pl.col("DATA_REF_MES").dt.weekday().alias("FEAT_DIA_SEMANA")
        ]).drop("DATA_REF_MES")

        # 4. TRATAMENTO ANTI-OOM (A Mágica da Memória)
        print("⚡ Tratando colunas e unificando bases...", flush=True)
        cols_comuns = list(set(df_ouro.columns).intersection(set(df_futuro.columns)))
        df_completo_pl = pl.concat([df_ouro.select(cols_comuns), df_futuro.select(cols_comuns)], how="vertical")

        # Liberação Imediata de Memória RAM
        del df_ouro
        del df_ouro_raw
        del df_futuro
        del df_dna_geografico
        gc.collect()

        cat_features_declaradas = [
            "H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", 
            "FEAT_PERFIL_VITIMA", "FEAT_CONTEXTO_CRITICO", "FEAT_TIPO_FERIADO", 
            "FEAT_IS_FIM_DE_SEMANA", "FEAT_TIPO_DIA"
        ]
        cat_features = [c for c in cat_features_declaradas if c in df_completo_pl.columns]

        print("🧹 Otimizando Strings na memória (Via Polars)...", flush=True)
        exprs = []
        for col in cat_features:
            expr = pl.col(col).cast(pl.Utf8).fill_null("DESCONHECIDO")
            expr = expr.str.replace(r"\.0$", "")
            expr = pl.when(expr.is_in(["nan", "NaN", "None", "<NA>", ""])).then(pl.lit("DESCONHECIDO")).otherwise(expr)
            exprs.append(expr.alias(col))

        df_completo_pl = df_completo_pl.with_columns(exprs)

        # 5. PREDIÇÃO EM LOTES (Chunking)
        print("🧠 Rodando predição em lotes de segurança...", flush=True)
        batch_size = 200000
        preds_list = []
        
        total_linhas = df_completo_pl.height
        
        for i in range(0, total_linhas, batch_size):
            # Extrai apenas o pedaço de dados necessário
            df_batch = df_completo_pl.slice(i, batch_size).select(modelo.feature_names_).to_pandas()
            
            # CatBoost precisa ver isso explicitamente como string no Pandas
            for col in cat_features:
                if col in df_batch.columns:
                    df_batch[col] = df_batch[col].astype(str)
            
            preds_batch = modelo.predict(df_batch)
            preds_list.extend(preds_batch)

        preds_raw = np.array(preds_list)

        print("⚖️ Calculando Risco de Exposição (Calibração 0.5 - 10.0)...", flush=True)
        volume_historico = df_completo_pl["FS_VOL_CRIMES_ANO_ANT"].fill_null(0.0).cast(pl.Float64).to_numpy()
        fator_frequencia = np.log1p(volume_historico) + 1.0
        massa_criminal = preds_raw * fator_frequencia
        
        p_min, p_max = massa_criminal.min(), massa_criminal.max()
        piso, teto = 0.5, 10.0
        preds_calibrados = piso + (massa_criminal - p_min) * (teto - piso) / (p_max - p_min)
        preds_clipped = np.clip(preds_calibrados, piso, teto) 
        
        df_dossie = df_completo_pl.with_columns(
            pl.Series("RISCO_PREDITO_IA", preds_clipped).round(2)
        )

        # 6. DNA DE RISCO (SHAP)
        print("🧬 Analisando DNA criminal (SHAP)...", flush=True)
        df_shap_sample = df_dossie.sample(n=min(35000, df_dossie.height), seed=42)
        X_shap = df_shap_sample.select(modelo.feature_names_).to_pandas()
        
        for col in cat_features:
            if col in X_shap.columns:
                X_shap[col] = X_shap[col].astype(str)
        
        explainer = shap.TreeExplainer(modelo)
        shap_vals = explainer.shap_values(X_shap)
        
        df_chaves_geo = df_shap_sample.select(["CIDADE", "BAIRRO"]).to_pandas()
        df_shap_geo = pd.concat([
            df_chaves_geo,
            pd.DataFrame(shap_vals, columns=[f"SHAP_{f}" for f in modelo.feature_names_])
        ], axis=1).groupby(["CIDADE", "BAIRRO"], dropna=False).mean().reset_index()

        # 7. SINCRONIZAÇÃO R2
        print("📦 Sincronizando com o Datalake (Ouro)...", flush=True)
        for key, data in [("looker_dossie_eventos.parquet", df_dossie), 
                         ("looker_dim_shap.parquet", pl.from_pandas(df_shap_geo))]:
            buf = io.BytesIO()
            data.write_parquet(buf, compression="zstd")
            self.s3.put_object(Bucket=self.bucket, Key=f"datalake/ouro/{key}", Body=buf.getvalue())

        duracao = time.time() - inicio_processo
        
        self.auditoria["metricas"] = {
            "historico": total_historico,
            "futuro": total_linhas - total_historico,
            "min": round(float(np.min(preds_clipped)), 4),
            "media": round(float(np.mean(preds_clipped)), 4),
            "max": round(float(np.max(preds_clipped)), 4),
            "tempo_s": round(duracao, 2)
        }

        report = (
            f"==============================================================\n"
            f" 🛡️ RELATÓRIO DE INTELIGÊNCIA MENSALIZADA - SAFEDRIVER \n"
            f"==============================================================\n"
            f"1. VOLUMETRIA (Foco: 2025 a 2026 - BQ FREE)\n"
            f"   • Histórico (2025)       : {total_historico:,} eventos\n"
            f"   • Malha Futura (Macro - 4 Meses) : {total_linhas - total_historico:,} projeções\n\n"
            f"2. RISCO (ESCALA 0.5 A 10)\n"
            f"   • Média de Risco Estado  : {self.auditoria['metricas']['media']:.4f}\n"
            f"   • Risco Máximo (Hotspots): {self.auditoria['metricas']['max']:.4f}\n"
            f"==============================================================\n"
        )
        print(report)
        self._notificar_discord(f"""```text\n{report}\n```""")

if __name__ == "__main__":
    GeradorDossieSafeDriver().gerar_dados()
