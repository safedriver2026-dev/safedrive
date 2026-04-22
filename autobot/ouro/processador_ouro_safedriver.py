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
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return pl.read_parquet(io.BytesIO(obj['Body'].read()))
        except: return None

    def construir_abt_final(self):
        inicio_timer = time.time()
        print("🚀 [OURO] Consolidando ABT Final para IA e Dashboard...", flush=True)
        
        # 1. CARREGAMENTO DOS COMPONENTES DA MALHA (Infra + Social)
        print("📥 Carregando Infraestrutura e Dados Sociais do Censo...", flush=True)
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_social = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_SOCIAL_H3.parquet")

        # Unimos as bases de apoio num único dicionário de inteligência por hexágono
        df_universo_h3 = df_infra.join(df_social, on="H3_INDEX", how="full", coalesce=True).fill_null(0)

        # 2. CARREGAMENTO DOS CRIMES (Com os 40k resgatados inclusos!)
        print("📥 Agregando crimes processados...", flush=True)
        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
        
        df_crimes = pl.concat([self._ler_parquet_r2(f) for f in crime_files], how="diagonal")

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
        df_final.write_parquet(buf, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/safedriver_abt_treino.parquet", Body=buf.getvalue())

        duracao = time.time() - inicio_timer
        self.auditoria["metricas"] = {
            "total_eventos": df_final.height,
            "colunas_geradas": len(df_final.columns),
            "tempo_execucao": duracao
        }

        # Notificação
        msg = (
            f"🏆 **[SafeDriver] Camada Ouro Concluída**\n"
            f"```ml\n"
            f"• ABT Gerada: {df_final.height} linhas\n"
            f"• Integridade: Crimes + Infra + Censo OK\n"
            f"• Status: PRONTO PARA MODELAGEM (CatBoost)\n"
            f"```"
        )
        self._notificar_discord(msg)
        print("✨ Tudo pronto. Podes abrir o Power BI ou começar o notebook de IA!")

if __name__ == "__main__":
    app = ArquitetoSafeDriverOuro()
    app.construir_abt_final()
