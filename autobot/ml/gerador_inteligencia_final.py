import os
import io
import json
import boto3
import polars as pl
import pandas as pd
import shap
import time
import requests
import numpy as np
import warnings
from datetime import datetime, date
from catboost import CatBoostRegressor
from botocore.config import Config

warnings.filterwarnings("ignore", category=FutureWarning)

class GeradorDossieSafeDriver:
    """
    Componente 'Cérebro' do SafeDriver.
    Realiza a inferência massiva, gera a malha futura 2026 e extrai explicabilidade (SHAP).
    """
    def __init__(self):
        # 1. IDENTIDADE E SINCRONIA (Chave para o Data Lake)
        self.projeto = os.getenv("NOME_PROJETO", "safedriver").strip().lower()
        
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint, 
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4', retries={'max_attempts': 5})
        )
        
        self.webhook_url = os.getenv("DISCORD_SUCESSO")
        self.modelo_local = "modelo_safedriver_catboost.cbm"
        self.auditoria = {
            "projeto": "SafeDriver",
            "fase": "Dossiê de Inteligência Geográfica (Histórico + Futuro)",
            "timestamp": str(datetime.now()),
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except: pass

    def _sincronizar_r2_otimizado(self, buffer, s3_key):
        print(f"   -> Sincronizando artefato: {s3_key}...", flush=True)
        buffer.seek(0)
        self.s3.upload_fileobj(buffer, self.bucket, s3_key)

    def gerar_dados(self):
        inicio_processo = time.time()
        print(f"🚀 Iniciando Motor de Inteligência para o projeto: {self.projeto}", flush=True)
        
        # 1. DOWNLOAD E CARREGAMENTO DO MODELO (Tweedie Optimized)
        if not os.path.exists(self.modelo_local):
            print("📥 Baixando artefato de modelo do R2...", flush=True)
            self.s3.download_file(self.bucket, f"modelos/{self.modelo_local}", self.modelo_local)
        
        modelo = CatBoostRegressor().load_model(self.modelo_local)
        features_modelo = modelo.feature_names_
        cat_features = [
            "H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", 
            "FEAT_PERFIL_VITIMA", "FEAT_CONTEXTO_CRITICO", "FEAT_TIPO_DIA"
        ]

        # 2. CARREGAMENTO DA ABT OURO (HISTÓRICO)
        key_ouro = f"datalake/ouro/{self.projeto}_abt_treino.parquet"
        print(f"📖 Acessando Camada Ouro: {key_ouro}", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key_ouro)
        df_historico = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        # 3. GERAÇÃO DA MALHA FUTURA (A MÁQUINA DO TEMPO 2026)
        print("🔮 Gerando Matriz Sintética de cenários para 2026...", flush=True)
        
        # Extrai o 'DNA' estrutural dos bairros
        df_dna_bairros = df_historico.select([
            "H3_INDEX", "LATITUDE", "LONGITUDE", "CIDADE", "BAIRRO", 
            "MICRO_POPULACAO_FACES", "CENSO_MEDIA_V0001", "CENSO_MEDIA_V0002",
            "MACRO_FINANCEIRO", "MACRO_LAZER_NOTURNO", "MACRO_VAREJO",
            "FS_VOL_CRIMES_ANO_ANT", "FS_RISCO_MEDIO_ANO_ANT"
        ]).unique(subset=["H3_INDEX"])

        # Define 16 cenários por hexágono (4 Turnos x 2 Tipos Dia x 2 Perfis)
        df_cenarios = pl.DataFrame({
            "SAZON_PERIODO": ["MANHA", "TARDE", "NOITE", "MADRUGADA"] * 4,
            "FEAT_TIPO_DIA": ["DIA_UTIL"] * 8 + ["FIM_DE_SEMANA"] * 8,
            "FEAT_PERFIL_VITIMA": (["MOTORISTA"] * 4 + ["PEDESTRE"] * 4) * 2
        }).with_columns(
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        )

        df_futuro = df_dna_bairros.join(df_cenarios, how="cross")
        
        # Parâmetros Temporais de 2026
        data_referencia = date(2026, 6, 15)
        df_futuro = df_futuro.with_columns([
            pl.lit(data_referencia).cast(pl.Date).alias("DATAOCORRENCIA"),
            pl.lit(0.0).alias("LABEL_PESO_RISCO"),
            pl.lit(2026).alias("ANO_JOIN"),
            pl.lit(data_referencia.month).alias("FEAT_MES"),
            pl.lit(data_referencia.weekday()).alias("FEAT_DIA_SEMANA")
        ])

        # 4. CONSOLIDAÇÃO E INFERÊNCIA
        print("🔗 Unindo Passado e Futuro para processamento massivo...", flush=True)
        df_hist_pd = df_historico.to_pandas()
        df_fut_pd = df_futuro.to_pandas()
        
        cols_comuns = list(set(df_hist_pd.columns).intersection(set(df_fut_pd.columns)))
        df_completo_pd = pd.concat([df_hist_pd[cols_comuns], df_fut_pd[cols_comuns]], ignore_index=True)
        
        # Saneamento Categórico (Evita descompasso de tipagem)
        X_all = df_completo_pd[features_modelo].copy()
        for col in cat_features:
            X_all[col] = X_all[col].fillna("DESCONHECIDO").astype(str).str.replace(r'\.0$', '', regex=True)

        print("🤖 Executando inferência CatBoost Tweedie...", flush=True)
        preds = modelo.predict(X_all)
        
        df_dossie = pl.from_pandas(df_completo_pd).with_columns(
            pl.Series("RISCO_PREDITO_IA", np.clip(preds, 0, 10)).round(2)
        )

        # 5. EXPLICABILIDADE SHAP (O 'PORQUÊ' DO RISCO)
        print("🧬 Extraindo DNA Criminal via SHAP Values...", flush=True)
        explainer = shap.TreeExplainer(modelo)
        # Amostra estatística para não estourar memória, mantendo a relevância
        shap_sample = X_all.sample(n=min(35000, len(X_all)), random_state=42)
        shap_values = explainer.shap_values(shap_sample)
        
        df_shap_geo = pd.concat([
            df_completo_pd.loc[shap_sample.index, ["CIDADE", "BAIRRO"]].reset_index(drop=True),
            pd.DataFrame(shap_values, columns=[f"SHAP_{f}" for f in features_modelo])
        ], axis=1).groupby(["CIDADE", "BAIRRO"]).mean().reset_index()

        # 6. SINCRONIZAÇÃO FINAL COM O DATA LAKE
        print("📤 Enviando Dossiê e Dimensão SHAP para o R2...", flush=True)
        
        for data, s3_key in [(df_dossie, "datalake/ouro/looker_dossie_eventos.parquet"), 
                             (pl.from_pandas(df_shap_geo), "datalake/ouro/looker_dim_shap.parquet")]:
            buf = io.BytesIO()
            data.write_parquet(buf, compression="zstd")
            self._sincronizar_r2_otimizado(buf, s3_key)

        # TELEMETRIA
        duracao = round(time.time() - inicio_processo, 2)
        self.auditoria["metricas"] = {
            "registros_totais": df_dossie.height,
            "bairros_mapeados": len(df_shap_geo),
            "tempo_execucao_s": duracao,
            "pico_risco_detectado": float(df_dossie["RISCO_PREDITO_IA"].max())
        }
        
        buf_log = io.BytesIO(json.dumps(self.auditoria, indent=4).encode())
        self.s3.put_object(Bucket=self.bucket, Key="modelos/AUDITORIA_INTELIGENCIA_FINAL.json", Body=buf_log.getvalue())
        
        self._notificar_discord(f"🧬 **[Inteligência]** Dossiê gerado: {df_dossie.height} linhas em {duracao}s. Risco Máximo: {self.auditoria['metricas']['pico_risco_detectado']}")
        print(f"✨ Inteligência Final concluída com sucesso!")

if __name__ == "__main__":
    GeradorDossieSafeDriver().gerar_dados()
