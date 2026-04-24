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
from datetime import datetime
from catboost import CatBoostRegressor
from botocore.config import Config

# Desativa alertas de deprecacao para otimizar a legibilidade dos logs de execucao
warnings.filterwarnings("ignore", category=FutureWarning)

class GeradorDossieSafeDriver:
    """
    Componente responsavel pela execucao da camada de Inteligencia Preditiva.
    Realiza a inferencia em larga escala (4.5M+ registros) e extrai o DNA criminal
    atraves de valores SHAP agregados por localidade. Implementa streaming de dados
    para garantir estabilidade em ambientes de memoria limitada.
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
            "fase": "Dossie de Inteligencia Geografica",
            "data_processamento": str(datetime.now()),
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        """Transmissao de metadados operacionais para telemetria."""
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=15)
            except: pass

    def _sincronizar_r2_otimizado(self, buffer, s3_key):
        """
        Executa o upload via streaming (Multipart) para evitar picos de consumo de RAM.
        Esta abordagem elimina a necessidade do metodo .getvalue(), enviando o buffer
        diretamente do ponteiro de memoria para o repositorio em nuvem.
        """
        print(f"   -> Iniciando upload de alta performance: {s3_key}...", flush=True)
        buffer.seek(0)
        self.s3.upload_fileobj(buffer, self.bucket, s3_key)
        print(f"   -> Sincronizacao concluida.", flush=True)

    def gerar_dados(self):
        inicio_processo = time.time()
        print("Inicializando Motor de Inferencia Preditiva...", flush=True)
        
        # 1. CARREGAMENTO DO MODELO PREDITIVO
        if not os.path.exists(self.modelo_local):
            print(f"Baixando artefato de modelo do Data Lake...", flush=True)
            self.s3.download_file(self.bucket, f"modelos/{self.modelo_local}", self.modelo_local)
        modelo = CatBoostRegressor().load_model(self.modelo_local)
        
        # Mapeamento de variaveis categoricas estritas do projeto
        cat_features_originais = [
            "H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", 
            "FEAT_PERFIL_VITIMA", "FEAT_CONTEXTO_CRITICO", "FEAT_TIPO_DIA"
        ]

        # 2. CARREGAMENTO DA ANALYTICAL BASE TABLE (OURO)
        print("Acessando Camada Ouro (Refined Data)...", flush=True)
        obj = self.s3.get_object(Bucket=self.bucket, Key="datalake/ouro/safedriver_abt_treino.parquet")
        df_ouro = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        total_linhas = df_ouro.height
        
        # 3. CONVERSAO ESTRITA DE TIPOS (PREVENCAO DE ERRO DE FLOAT NO CATBOOST)
        print("Ajustando tipagem vetorial para compatibilidade C++...", flush=True)
        X_all = df_ouro.select(modelo.feature_names_).to_pandas()
        for col in X_all.columns:
            if col in cat_features_originais:
                # Normaliza nulos e remove sufixos decimais (ex: '3.0' vira '3')
                X_all[col] = X_all[col].fillna("DESCONHECIDO").astype(str).str.replace(r'\.0$', '', regex=True)
            
        # 4. EXECUCAO DA INFERENCIA MASSIVA
        print("Processando predições de vulnerabilidade urbana...", flush=True)
        preds = modelo.predict(X_all)
        df_dossie = df_ouro.with_columns(
            pl.Series("RISCO_PREDITO_IA", preds).round(2)
        )

        # 5. EXPLICABILIDADE GEOSPACIAL (DNA SHAP)
        print("Extraindo DNA criminal via SHapley Additive exPlanations...", flush=True)
        # Amostragem estatistica de 50k registros para viabilidade computacional
        df_shap_sample = df_ouro.sample(n=min(50000, df_ouro.height), seed=42)
        X_shap = df_shap_sample.select(modelo.feature_names_).to_pandas()
        for col in X_shap.columns:
            if col in cat_features_originais:
                X_shap[col] = X_shap[col].fillna("DESCONHECIDO").astype(str).str.replace(r'\.0$', '', regex=True)
        
        shap_vals = shap.TreeExplainer(modelo).shap_values(X_shap)
        df_shap_geo = pd.concat([
            df_shap_sample.select(["CIDADE", "BAIRRO"]).to_pandas(),
            pd.DataFrame(shap_vals, columns=[f"SHAP_{f}" for f in modelo.feature_names_])
        ], axis=1).groupby(["CIDADE", "BAIRRO"]).mean().reset_index()

        # 6. SINCRONIZACAO OTIMIZADA COM O DATA LAKEHOUSE
        print("Sincronizando artefatos refinados com o Cloudflare R2...", flush=True)
        
        # Upload da Tabela Fato (Eventos com Risco IA)
        buf_eventos = io.BytesIO()
        df_dossie.write_parquet(buf_eventos, compression="zstd")
        self._sincronizar_r2_otimizado(buf_eventos, "datalake/ouro/looker_dossie_eventos.parquet")
        buf_eventos.close()

        # Upload da Tabela Dimensao (DNA Espacial SHAP)
        buf_shap = io.BytesIO()
        pl.from_pandas(df_shap_geo).write_parquet(buf_shap, compression="zstd")
        self._sincronizar_r2_otimizado(buf_shap, "datalake/ouro/looker_dim_shap.parquet")
        buf_shap.close()

        # 7. TELEMETRIA E AUDITORIA FINAL
        duracao = time.time() - inicio_processo
        media_risco = float(np.mean(preds))
        max_risco = float(np.max(preds))
        
        self.auditoria["metricas"] = {
            "total_processado": total_linhas,
            "media_risco_predito": round(media_risco, 4),
            "max_risco_detectado": round(max_risco, 4),
            "bairros_analisados": len(df_shap_geo),
            "tempo_execucao_s": round(duracao, 2)
        }

        buf_log = io.BytesIO(json.dumps(self.auditoria, indent=4).encode())
        self.s3.put_object(Bucket=self.bucket, Key="modelos/AUDITORIA_DOSSIE_INTELIGENCIA.json", Body=buf_log.getvalue())

        report = (
            f"Relatorio Operacional - Dossie de Inteligencia\n"
            f"==============================================================\n"
            f"   - Registros Processados       : {total_linhas}\n"
            f"   - Localidades (Bairros)       : {len(df_shap_geo)}\n"
            f"   - Risco Medio da Malha        : {media_risco:.4f}\n"
            f"   - Tempo de Processamento      : {duracao:.2f}s\n"
            f"==============================================================\n"
        )
        print(report)
        self._notificar_discord(f"```text\n{report}\n```")

if __name__ == "__main__":
    GeradorDossieSafeDriver().gerar_dados()
