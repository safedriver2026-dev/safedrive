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
    Motor de Inteligência e Inferência.
    Conexão blindada com o R2 + Malha Futura (2026) + Extração SHAP + Min/Max Score.
    """
    def __init__(self):
        self.projeto = os.getenv("NOME_PROJETO", "safedriver").strip().lower()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # --- Limpeza do Endpoint R2 ---
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
        self.caminho_abt = f"datalake/ouro/{self.projeto}_abt_treino.parquet"
        
        # Inicialização da Auditoria
        self.auditoria = {
            "projeto": self.projeto.upper(),
            "fase": "Dossiê de Inteligência Geográfica Preditiva",
            "data_processamento": str(datetime.now()),
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except: pass

    def gerar_dados(self):
        inicio_processo = time.time()
        print("🧠 [DOSSIÊ] Iniciando motor de inteligência preditiva...", flush=True)
        
        # =================================================================
        # 1. DOWNLOAD DO MODELO E CARGA DA BASE
        # =================================================================
        if not os.path.exists(self.modelo_local):
            print(f"📥 Baixando {self.modelo_local} do bucket...", flush=True)
            self.s3.download_file(self.bucket, f"modelos/{self.modelo_local}", self.modelo_local)
        
        modelo = CatBoostRegressor().load_model(self.modelo_local)
        
        print("📥 Lendo a base Ouro para inferência massiva...", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.caminho_abt)
        df_ouro = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # Recriação da Feature Necessária
        df_ouro = df_ouro.with_columns(
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        )

        # =================================================================
        # 2. GERAÇÃO DA MALHA FUTURA (CENÁRIOS PARA 2026)
        # =================================================================
        print("🔮 Gerando Matriz Sintética de cenários futuros (2026)...", flush=True)
        
        # Puxa o DNA estático dos bairros
        colunas_dna = [c for c in df_ouro.columns if c in [
            "H3_INDEX", "LATITUDE", "LONGITUDE", "CIDADE", "BAIRRO",
            "MICRO_POPULACAO_FACES", "CENSO_MEDIA_V0001", "CENSO_MEDIA_V0002"
        ] or c.startswith("MACRO_") or c.startswith("FS_")]
        
        df_dna_bairros = df_ouro.select(colunas_dna).unique(subset=["H3_INDEX"])

        # 16 Cenários possíveis: 4 Turnos x 2 Tipos de Dia x 2 Perfis
        df_cenarios = pl.DataFrame({
            "SAZON_PERIODO": ["MANHA", "TARDE", "NOITE", "MADRUGADA"] * 4,
            "FEAT_TIPO_DIA": ["DIA_UTIL"] * 8 + ["FIM_DE_SEMANA"] * 8,
            "FEAT_PERFIL_VITIMA": (["MOTORISTA"] * 4 + ["PEDESTRE"] * 4) * 2
        }).with_columns(
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        )

        df_futuro = df_dna_bairros.join(df_cenarios, how="cross")
        
        # Simulação para o Looker Studio ter dados de 2026
        data_ref = date(2026, 6, 15)
        df_futuro = df_futuro.with_columns([
            pl.lit(data_ref).cast(pl.Date).alias("DATAOCORRENCIA"),
            pl.lit(0.0).alias("LABEL_PESO_RISCO"),
            pl.lit(2026).cast(pl.Int32).alias("ANO_JOIN"),
            pl.lit(data_ref.month).alias("FEAT_MES"),
            pl.lit(data_ref.weekday()).alias("FEAT_DIA_SEMANA")
        ])

        # =================================================================
        # 3. PREDIÇÃO MASSIVA (PASSADO + FUTURO)
        # =================================================================
        print("⚡ Rodando predição em todos os registros (Reais + Sintéticos)...", flush=True)
        df_hist_pd = df_ouro.to_pandas()
        df_fut_pd = df_futuro.to_pandas()
        
        cols_comuns = list(set(df_hist_pd.columns).intersection(set(df_fut_pd.columns)))
        df_completo_pd = pd.concat([df_hist_pd[cols_comuns], df_fut_pd[cols_comuns]], ignore_index=True)
        
        X_all = df_completo_pd[modelo.feature_names_].copy()
        
        for col in X_all.select_dtypes(['object', 'category']).columns: 
            X_all[col] = X_all[col].fillna("DESCONHECIDO").astype(str)
            
        preds_raw = modelo.predict(X_all)
        # Trava matemática: O risco não pode ser negativo nem maior que 10
        preds_clipped = np.clip(preds_raw, 0, 10)
        
        df_dossie = pl.from_pandas(df_completo_pd).with_columns(
            pl.Series("RISCO_PREDITO_IA", preds_clipped).round(2)
        )

        # =================================================================
        # 4. DNA DE RISCO (SHAP POR BAIRRO)
        # =================================================================
        print("🧬 Analisando DNA criminal (SHAP) geográfico...", flush=True)
        df_shap_sample = df_dossie.sample(n=min(35000, df_dossie.height), seed=42)
        X_shap = df_shap_sample.select(modelo.feature_names_).to_pandas()
        
        for col in X_shap.select_dtypes(['object', 'category']).columns: 
            X_shap[col] = X_shap[col].astype(str)
        
        explainer = shap.TreeExplainer(modelo)
        shap_vals = explainer.shap_values(X_shap)
        
        df_shap_geo = pd.concat([
            df_shap_sample.select(["CIDADE", "BAIRRO"]).to_pandas(),
            pd.DataFrame(shap_vals, columns=[f"SHAP_{f}" for f in modelo.feature_names_])
        ], axis=1).groupby(["CIDADE", "BAIRRO"]).mean().reset_index()

        # =================================================================
        # 5. SALVAMENTO NO R2
        # =================================================================
        print("📦 Sincronizando resultados com o R2...", flush=True)
        
        buf_eventos = io.BytesIO()
        df_dossie.write_parquet(buf_eventos, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dossie_eventos.parquet", Body=buf_eventos.getvalue())
        
        buf_shap = io.BytesIO()
        pl.from_pandas(df_shap_geo).write_parquet(buf_shap, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key="datalake/ouro/looker_dim_shap.parquet", Body=buf_shap.getvalue())

        # =================================================================
        # 6. FINALIZAÇÃO E RELATÓRIO (COM MIN/MAX)
        # =================================================================
        duracao = time.time() - inicio_processo
        
        # Extração de Métricas
        min_risco = float(np.min(preds_clipped))
        media_risco = float(np.mean(preds_clipped))
        max_risco = float(np.max(preds_clipped))
        
        self.auditoria["metricas"] = {
            "historico_processado": df_ouro.height,
            "cenarios_futuros_gerados": df_futuro.height,
            "total_processado": df_dossie.height,
            "min_risco_detectado": round(min_risco, 4),
            "media_risco_predito": round(media_risco, 4),
            "max_risco_detectado": round(max_risco, 4),
            "bairros_analisados": len(df_shap_geo),
            "tempo_execucao_s": round(duracao, 2)
        }

        # Salva o log de auditoria no R2
        buf_log = io.BytesIO(json.dumps(self.auditoria, indent=4).encode())
        self.s3.put_object(Bucket=self.bucket, Key="modelos/AUDITORIA_DOSSIE_INTELIGENCIA.json", Body=buf_log.getvalue())

        report = (
            f"==============================================================\n"
            f" 🛡️ RELATÓRIO DE INTELIGÊNCIA - {self.projeto.upper()} \n"
            f"==============================================================\n"
            f"1. VOLUMETRIA DO DOSSIÊ\n"
            f"   • Registros Históricos    : {df_ouro.height:,}\n"
            f"   • Projeções Futuras (26)  : {df_futuro.height:,}\n"
            f"   • Total Processado        : {df_dossie.height:,}\n"
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
        
        if os.path.exists(self.modelo_local): os.remove(self.modelo_local)

if __name__ == "__main__":
    GeradorDossieSafeDriver().gerar_dados()
