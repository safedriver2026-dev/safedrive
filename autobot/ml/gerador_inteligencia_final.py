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
from datetime import datetime, date
from catboost import CatBoostRegressor
from botocore.config import Config
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

class GeradorDossieSafeDriver:
    """
    Motor de Inteligência Preditiva (Versão Estável R2).
    Gera Dossiê do Passado + Projeções para 2026 com Min/Max e SHAP Values.
    Blindagem de Titânio contra tipagem Pandas (NaN).
    """
    def __init__(self):
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # --- Conexão Estável R2 (Testada e Aprovada) ---
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
        
        # Inicialização da Auditoria
        self.auditoria = {
            "projeto": "SafeDriver",
            "fase": "Dossiê de Inteligência Geográfica (Histórico + Futuro)",
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
        
        # =================================================================
        # 1. DOWNLOAD E CARGA DO MODELO
        # =================================================================
        if not os.path.exists(self.modelo_local):
            print(f"📥 Baixando {self.modelo_local} do bucket...", flush=True)
            self.s3.download_file(self.bucket, f"modelos/{self.modelo_local}", self.modelo_local)
        
        modelo = CatBoostRegressor().load_model(self.modelo_local)
        
        # =================================================================
        # 2. CARGA DA ABT OURO
        # =================================================================
        print("📥 Lendo a base Ouro para inferência massiva...", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key="datalake/ouro/safedriver_abt_treino.parquet")
        df_ouro = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        total_historico = df_ouro.height
        
        print("⚙️ Recriando Feature Cross (Sazonalidade x Perfil)...", flush=True)
        df_ouro = df_ouro.with_columns(
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        )

        # =================================================================
        # 3. GERAÇÃO DA MALHA FUTURA (2026)
        # =================================================================
        print("🔮 Projetando cenários futuros para 2026...", flush=True)
        
        colunas_dna = [c for c in df_ouro.columns if c in [
            "H3_INDEX", "LATITUDE", "LONGITUDE", "CIDADE", "BAIRRO",
            "MICRO_POPULACAO_FACES", "CENSO_MEDIA_V0001", "CENSO_MEDIA_V0002"
        ] or c.startswith("MACRO_") or c.startswith("FS_")]
        
        df_dna_bairros = df_ouro.select(colunas_dna).unique(subset=["H3_INDEX"])

        # 16 Combinações: Turno x Dia x Vítima
        df_cenarios = pl.DataFrame({
            "SAZON_PERIODO": ["MANHA", "TARDE", "NOITE", "MADRUGADA"] * 4,
            "FEAT_TIPO_DIA": ["DIA_UTIL"] * 8 + ["FIM_DE_SEMANA"] * 8,
            "FEAT_PERFIL_VITIMA": (["MOTORISTA"] * 4 + ["PEDESTRE"] * 4) * 2
        }).with_columns(
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        )

        df_futuro = df_dna_bairros.join(df_cenarios, how="cross")
        data_ref = date(2026, 6, 15)
        df_futuro = df_futuro.with_columns([
            pl.lit(data_ref).cast(pl.Date).alias("DATAOCORRENCIA"),
            pl.lit(0.0).alias("LABEL_PESO_RISCO"),
            pl.lit(2026).cast(pl.Int32).alias("ANO_JOIN"),
            pl.lit(data_ref.month).alias("FEAT_MES"),
            pl.lit(data_ref.weekday()).alias("FEAT_DIA_SEMANA")
        ])

        # =================================================================
        # 4. PREDIÇÃO E A MARRETA DE TITÂNIO
        # =================================================================
        print("⚡ Rodando predição (Histórico + Projeções 2026)...", flush=True)
        df_hist_pd = df_ouro.to_pandas()
        df_fut_pd = df_futuro.to_pandas()
        
        cols_comuns = list(set(df_hist_pd.columns).intersection(set(df_fut_pd.columns)))
        df_completo_pd = pd.concat([df_hist_pd[cols_comuns], df_fut_pd[cols_comuns]], ignore_index=True)
        
        X_all = df_completo_pd[modelo.feature_names_].copy()
        
        cat_features_declaradas = [
            "H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", 
            "FEAT_PERFIL_VITIMA", "FEAT_CONTEXTO_CRITICO", "FEAT_TIPO_FERIADO", 
            "FEAT_IS_FIM_DE_SEMANA", "FEAT_TIPO_DIA"
        ]
        cat_features = [c for c in cat_features_declaradas if c in X_all.columns]
        
        # A Marreta de Titânio: Mata o NaN real e o NaN em texto
        for col in cat_features:
            X_all[col] = X_all[col].fillna('DESCONHECIDO') # Pega np.nan injetado pelo pd.concat
            X_all[col] = X_all[col].astype(str)            # Força string nativa
            X_all[col] = X_all[col].str.replace(r'\.0$', '', regex=True) # Tira casa decimal fantasma
            X_all[col] = X_all[col].replace(['nan', 'NaN', 'None', '<NA>', ''], 'DESCONHECIDO')
            X_all[col] = X_all[col].astype(object)         # Tranca o tipo para o CatBoost
            
        preds_raw = modelo.predict(X_all)
        preds_clipped = np.clip(preds_raw, 0, 10) # Trava de segurança (0 a 10)
        
        df_dossie = pl.from_pandas(df_completo_pd).with_columns(
            pl.Series("RISCO_PREDITO_IA", preds_clipped).round(2)
        )

        # =================================================================
        # 5. DNA DE RISCO (SHAP) COM BLINDAGEM REPLICADA
        # =================================================================
        print("🧬 Analisando DNA criminal (SHAP) geográfico...", flush=True)
        df_shap_sample = df_dossie.sample(n=min(35000, df_dossie.height), seed=42)
        X_shap = df_shap_sample.select(modelo.feature_names_).to_pandas()
        
        # Reaplica a marreta na amostra do SHAP para evitar que quebre aqui
        for col in cat_features:
            X_shap[col] = X_shap[col].fillna('DESCONHECIDO')
            X_shap[col] = X_shap[col].astype(str)
            X_shap[col] = X_shap[col].str.replace(r'\.0$', '', regex=True)
            X_shap[col] = X_shap[col].replace(['nan', 'NaN', 'None', '<NA>', ''], 'DESCONHECIDO')
            X_shap[col] = X_shap[col].astype(object)
        
        explainer = shap.TreeExplainer(modelo)
        shap_vals = explainer.shap_values(X_shap)
        
        df_shap_geo = pd.concat([
            df_shap_sample.select(["CIDADE", "BAIRRO"]).to_pandas(),
            pd.DataFrame(shap_vals, columns=[f"SHAP_{f}" for f in modelo.feature_names_])
        ], axis=1).groupby(["CIDADE", "BAIRRO"]).mean().reset_index()

        # =================================================================
        # 6. SALVAMENTO E RELATÓRIO
        # =================================================================
        print("📦 Sincronizando resultados com o R2...", flush=True)
        buf_eventos = io.BytesIO()
        df_dossie.write_parquet(buf_eventos, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dossie_eventos.parquet", Body=buf_eventos.getvalue())
        
        buf_shap = io.BytesIO()
        pl.from_pandas(df_shap_geo).write_parquet(buf_shap, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dim_shap.parquet", Body=buf_shap.getvalue())

        duracao = time.time() - inicio_processo
        
        min_risco = float(np.min(preds_clipped))
        media_risco = float(np.mean(preds_clipped))
        max_risco = float(np.max(preds_clipped))
        
        self.auditoria["metricas"] = {
            "historico_processado": total_historico,
            "futuro_gerado_2026": df_futuro.height,
            "total_processado_geral": df_dossie.height,
            "min_risco_detectado": round(min_risco, 4),
            "media_risco_predito": round(media_risco, 4),
            "max_risco_detectado": round(max_risco, 4),
            "bairros_analisados": len(df_shap_geo),
            "tempo_execucao_s": round(duracao, 2)
        }

        buf_log = io.BytesIO(json.dumps(self.auditoria, indent=4).encode())
        self.s3.put_object(Bucket=self.bucket, Key="modelos/AUDITORIA_DOSSIE_INTELIGENCIA.json", Body=buf_log.getvalue())

        report = (
            f"==============================================================\n"
            f" 🛡️ RELATÓRIO DE INTELIGÊNCIA - SAFEDRIVER \n"
            f"==============================================================\n"
            f"1. VOLUMETRIA DO DOSSIÊ\n"
            f"   • Histórico Processado    : {total_historico:,}\n"
            f"   • Projeções Futuras (26)  : {df_futuro.height:,}\n"
            f"   • Total Consolidado       : {df_dossie.height:,}\n"
            f"   • Bairros Mapeados (DNA)  : {len(df_shap_geo)}\n\n"
            f"2. PERFORMANCE DA IA (ESCALA DE RISCO 0 A 10)\n"
            f"   • Risco Mínimo Detectado  : {min_risco:.4f}\n"
            f"   • Risco Médio Predito     : {media_risco:.4f}\n"
            f"   • Risco Máximo Detectado  : {max_risco:.4f}\n\n"
            f"3. STATUS DO DEPLOY\n"
            f"   • Dossiê Eventos (R2)     : Sincronizado\n"
            f"   • Dimensão SHAP (R2)      : Sincronizado\n"
            f"   • Log de Auditoria        : Gerado\n"
            f"==============================================================\n"
            f"Duração: {duracao/60:.2f} min | Status: Pronto para BigQuery\n"
            f"==============================================================\n"
        )
        print(report)
        self._notificar_discord(f"```text\n{report}\n```")

if __name__ == "__main__":
    GeradorDossieSafeDriver().gerar_dados()
