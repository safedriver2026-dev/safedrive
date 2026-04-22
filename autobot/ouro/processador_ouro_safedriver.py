import os
import boto3
import polars as pl
import io
import requests
import time
import json
from botocore.config import Config
from datetime import datetime

class ArquitetoSafeDriverOuro:
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
        self.prata_crimes = "datalake/prata/crimes_trusted"
        self.prata_malha = "datalake/prata/malha_trusted"
        self.ouro_dir = "datalake/ouro"
        
        self.auditoria = {
            "camada": "OURO",
            "data_processamento": str(datetime.now()),
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=10)
            except: pass

    def _ler_parquet_r2(self, key):
        """Versão com log de erro para não esconder falhas."""
        try:
            print(f"   -> Procurando: {key}", flush=True)
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
            print(f"      [OK] {df.height} linhas carregadas.")
            return df
        except Exception as e:
            print(f"      [ERRO FATAL] Falha ao carregar {key}. Detalhe: {e}")
            return None

    def construir_abt_final(self):
        inicio_timer = time.time()
        print("🚀 [OURO] Consolidando ABT Final para IA e Dashboard...", flush=True)
        
        # 1. CARREGAMENTO DOS COMPONENTES DA MALHA (Infra + Social)
        print("📥 Carregando Infraestrutura e Dados Sociais do Censo...", flush=True)
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_social = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_SOCIAL_H3.parquet")

        # Guard Clauses: Se faltar a base, não tenta fazer o Join
        if df_infra is None:
            raise FileNotFoundError(f"O ficheiro de Infraestrutura não foi encontrado no R2 ({self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet). Verifique se a Prata da Malha rodou com sucesso.")
        
        if df_social is None:
            # Caso não tenha o social, criamos um vazio apenas com a chave para o Join não quebrar.
            # Isto salva a execução caso o CSV do IBGE não tenha sido processado por falta de ficheiro.
            print("⚠️ Ficheiro Social ausente. Prosseguindo sem variáveis demográficas (Apenas Infra e Crimes).")
            df_social = pl.DataFrame({"H3_INDEX": df_infra["H3_INDEX"].unique()})

        # Unimos as bases de apoio num único dicionário de inteligência por hexágono
        df_universo_h3 = df_infra.join(df_social, on="H3_INDEX", how="full", coalesce=True).fill_null(0)

        # 2. CARREGAMENTO DOS CRIMES (Com os resgatados inclusos!)
        print("📥 Agregando crimes processados...", flush=True)
        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
        
        if not crime_files:
            raise FileNotFoundError("Nenhum ficheiro de crimes foi encontrado na Prata. A base analítica precisa de crimes para treinar o modelo.")

        df_crimes = pl.concat([self._ler_parquet_r2(f) for f in crime_files if self._ler_parquet_r2(f) is not None], how="diagonal")

        # 3. FEATURE ENGINEERING (Cérebro da IA)
        print("📅 Gerando Variáveis Temporais e de Risco...", flush=True)
        df_gold = df_crimes.with_columns([
            pl.col("DATAOCORRENCIA").dt.weekday().alias("SAZON_DIA_SEMANA"),
            pl.col("DATAOCORRENCIA").dt.month().alias("SAZON_MES"),
            # Target de Risco SafeDriver
            pl.when(pl.col("RUBRICA").str.contains("LATROCINIO|HOMICIDIO")).then(pl.lit(10))
            .when(pl.col("RUBRICA").str.contains("ROUBO")).then(pl.lit(5))
            .otherwise(pl.lit(1)).alias("SCORE_GRAVIDADE")
        ])

        # 4. O GRANDE JOIN FINAL (Crimes + Malha)
        print("🏙️ Fazendo o enriquecimento espacial completo...", flush=True)
        # Cada crime agora sabe exatamente quantas pessoas moram ali e quantos bares existem por perto
        df_final = df_gold.join(df_universo_h3, on="H3_INDEX", how="left")

        # 5. UPLOAD E AUDITORIA
        print("📦 Exportando ABT para o R2...", flush=True)
        buf = io.BytesIO()
        df_final
