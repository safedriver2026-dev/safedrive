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
import warnings
from datetime import datetime, date
from catboost import CatBoostRegressor
from botocore.config import Config

warnings.filterwarnings("ignore", category=FutureWarning)

class GeradorDossieSafeDriver:
    """
    Componente responsavel pela execucao da camada de Inteligencia Preditiva.
    Realiza a inferencia do historico E GERA A MALHA FUTURA (2026).
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
            config=Config(signature_version='s3v4', retries={'max_attempts': 5})
        )
        
        self.webhook_url = os.getenv("DISCORD_SUCESSO")
        self.modelo_local = "modelo_safedriver_catboost.cbm"
        self.auditoria = {
            "projeto": "SafeDriver",
            "fase": "Dossie de Inteligencia Geografica (Histórico + Futuro)",
            "data_processamento": str(datetime.now()),
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except: pass

    def _sincronizar_r2_otimizado(self, buffer, s3_key):
        print(f"   -> Iniciando upload de alta performance: {s3_key}...", flush=True)
        buffer.seek(0)
        self.s3.upload_fileobj(buffer, self.bucket, s3_key)
        print(f"   -> Sincronizacao concluida.", flush=True)

    def gerar_dados(self):
        inicio_processo = time.time()
        print("Inicializando Motor de Inferencia Preditiva (Passado e Futuro)...", flush=True)
        
        # 1. CARREGAMENTO DO MODELO
        if not os.path.exists(self.modelo_local):
            print(f"Baixando artefato de modelo do Data Lake...", flush=True)
            self.s3.download_file(self.bucket, f"modelos/{self.modelo_local}", self.modelo_local)
        modelo = CatBoostRegressor().load_model(self.modelo_local)
        
        cat_features_originais = [
            "H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", 
            "FEAT_PERFIL_VITIMA", "FEAT_CONTEXTO_CRITICO", "FEAT_TIPO_DIA"
        ]

        # 2. CARREGAMENTO DA ABT HISTÓRICA
        print("Acessando Camada Ouro (Histórico)...", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key="datalake/ouro/safedriver_abt_treino.parquet")
        df_historico = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # =====================================================================
        # NOVA ROTINA: GERAÇÃO DA MALHA FUTURA (2026)
        # =====================================================================
        print("Gerando Matriz Sintética para Previsões Futuras (2026)...", flush=True)
        
        df_dna_bairros = df_historico.select([
            "H3_INDEX", "LATITUDE", "LONGITUDE", "CIDADE", "BAIRRO_right", 
            "MICRO_POPULACAO_FACES", "CENSO_MEDIA_V0001", "CENSO_MEDIA_V0002",
            "MACRO_FINANCEIRO", "MACRO_LAZER_NOTURNO", "MACRO_VAREJO",
            "FS_VOL_CRIMES_ANO_ANT", "FS_RISCO_MEDIO_ANO_ANT"
        ]).unique(subset=["H3_INDEX"], maintain_order=True)

        turnos = ["MANHA", "TARDE", "NOITE", "MADRUGADA"]
        tipos_dia = ["DIA_UTIL", "FIM_DE_SEMANA"]
        perfis = ["MOTORISTA", "PEDESTRE"]

        df_cenarios = pl.DataFrame({
            "SAZON_PERIODO": turnos * len(tipos_dia) * len(perfis),
            "FEAT_TIPO_DIA": [t for t in tipos_dia for _ in range(len(turnos) * len(perfis))],
            "FEAT_PERFIL_VITIMA": [p for _ in range(len(tipos_dia)) for _ in range(len(turnos)) for p in perfis]
        }).with_columns(
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        ).unique()

        df_futuro = df_dna_bairros.join(df_cenarios, how="cross")
        
        data_futura = date(2026, 6, 15)
        
        # Correção: Adicionamos a extração de Dia da Semana e Mês para o modelo não falhar!
        df_futuro = df_futuro.with_columns([
            pl.lit(data_futura).alias("DATAOCORRENCIA"),
            pl.lit(0.0).alias("LABEL_PESO_RISCO"),
            pl.col("BAIRRO_right").alias("BAIRRO"),
            pl.lit(2026).cast(pl.Int32).alias("ANO_JOIN")
        ]).with_columns([
            pl.col("DATAOCORRENCIA").dt.weekday().cast(pl.Float64).alias("FEAT_DIA_SEMANA"),
            pl.col("DATAOCORRENCIA").dt.month().cast(pl.Float64).alias("FEAT_MES")
        ])
        
        print("Fundindo Histórico (Passado) com Matriz Sintética (Futuro)...", flush=True)
        cols_comuns = list(set(df_historico.columns).intersection(set(df_futuro.columns)))
        df_completo = pl.concat([df_historico.select(cols_comuns), df_futuro.select(cols_comuns)], how="vertical")
        
        # =====================================================================
        # FIM DA NOVA ROTINA
        # =====================================================================

        # 3. CONVERSAO ESTRITA E INFERENCIA
        print("Ajustando tipagem vetorial e executando inferência massiva...", flush=True)
        X_all = df_completo.select(modelo.feature_names_).to_pandas()
        for col in X_all.columns:
            if col in cat_features_originais:
                X_all[col] = X_all[col].fillna("DESCONHECIDO").astype(str).str.replace(r'\.0$', '', regex=True)
        
        preds = modelo.predict(X_all)
        df_dossie = df_completo.with_columns(
            pl.Series("RISCO_PREDITO_IA", preds).round(2)
        )

        # 4. EXPLICABILIDADE GEOSPACIAL (DNA SHAP)
        print("Extraindo DNA criminal via SHAP...", flush=True)
        df_shap_sample = df_completo.sample(n=min(50000, df_completo.height), seed=42)
        X_shap = df_shap_sample.select(modelo.feature_names_).to_pandas()
        for col in X_shap.columns:
            if col in cat_features_originais:
                X_shap[col] = X_shap[col].fillna("DESCONHECIDO").astype(str).str.replace(r'\.0$', '', regex=True)
        
        shap_vals = shap.TreeExplainer(modelo).shap_values(X_shap)
        df_shap_geo = pd.concat([
            df_shap_sample.select(["CIDADE", "BAIRRO"]).to_pandas(),
            pd.DataFrame(shap_vals, columns=[f"SHAP_{f}" for f in modelo.feature_names_])
        ], axis=1).groupby(["CIDADE", "BAIRRO"]).mean().reset_index()

        # 5. SINCRONIZACAO OTIMIZADA COM O DATA LAKEHOUSE
        print("Sincronizando Dossiê (Passado + Futuro) com o Cloudflare R2...", flush=True)
        
        buf_eventos = io.BytesIO()
        df_dossie.write_parquet(buf_eventos, compression="zstd")
        self._sincronizar_r2_otimizado(buf_eventos, "datalake/ouro/looker_dossie_eventos.parquet")
        buf_eventos.close()

        buf_shap = io.BytesIO()
        pl.from_pandas(df_shap_geo).write_parquet(buf_shap, compression="zstd")
        self._sincronizar_r2_otimizado(buf_shap, "datalake/ouro/looker_dim_shap.parquet")
        buf_shap.close()

        # 6. TELEMETRIA
        duracao = time.time() - inicio_processo
        
        self.auditoria["metricas"] = {
            "total_processado": df_completo.height,
            "bairros_analisados": len(df_shap_geo),
            "tempo_execucao_s": round(duracao, 2)
        }

        buf_log = io.BytesIO(json.dumps(self.auditoria, indent=4).encode())
        self.s3.put_object(Bucket=self.bucket, Key="modelos/AUDITORIA_DOSSIE_INTELIGENCIA.json", Body=buf_log.getvalue())

        print(f"🏆 Sucesso! Dossiê com {df_completo.height} registros (incluindo 2026) gerado em {duracao:.2f}s.")

if __name__ == "__main__":
    GeradorDossieSafeDriver().gerar_dados()
