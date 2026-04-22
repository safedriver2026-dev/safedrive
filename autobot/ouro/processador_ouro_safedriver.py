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
            "projeto": "SafeDriver - Camada Ouro",
            "data_processamento": str(datetime.now()),
            "metricas": {}
        }

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=10)
            except: pass

    def _ler_parquet_r2(self, key):
        try:
            print(f"   -> Procurando: {key}", flush=True)
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
            print(f"      [OK] {df.height} linhas carregadas.")
            return df
        except Exception as e:
            print(f"      [AVISO] Falha ao carregar {key}. Detalhe: {e}")
            return None

    def construir_abt_final(self):
        inicio_timer = time.time()
        print("🚀 [OURO] Iniciando Consolidação da ABT Final...", flush=True)
        
        # 1. CARREGAMENTO DOS COMPONENTES DA MALHA
        print("📥 Carregando Infraestrutura e Dados Sociais...", flush=True)
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_social = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_SOCIAL_H3.parquet")

        if df_infra is None:
            raise FileNotFoundError("Base de Infraestrutura não encontrada. Abortando.")
        
        if df_social is None:
            print("⚠️ Ficheiro Social ausente. Gerando esqueleto H3 para manter o Join.")
            df_social = pl.DataFrame({"H3_INDEX": df_infra["H3_INDEX"].unique()})

        # Unimos as bases de apoio num único mapa de inteligência por hexágono
        df_universo_h3 = df_infra.join(df_social, on="H3_INDEX", how="full", coalesce=True).fill_null(0)

        # 2. CARREGAMENTO DOS CRIMES
        print("📥 Consolidando crimes da Prata...", flush=True)
        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
        
        if not crime_files:
            raise FileNotFoundError("Nenhum crime encontrado na Prata.")

        # Carrega e empilha todos os anos
        lista_crimes = []
        for f in crime_files:
            df_ano = self._ler_parquet_r2(f)
            if df_ano is not None:
                lista_crimes.append(df_ano)
        
        df_crimes = pl.concat(lista_crimes, how="diagonal")

        # 3. FEATURE ENGINEERING
        print("📅 Criando Features de Tempo e Risco...", flush=True)
        df_gold = df_crimes.with_columns([
            pl.col("DATAOCORRENCIA").dt.weekday().alias("FEAT_DIA_SEMANA"),
            pl.col("DATAOCORRENCIA").dt.month().alias("FEAT_MES"),
            pl.when(pl.col("RUBRICA").str.contains("LATROCINIO|HOMICIDIO")).then(pl.lit(10))
            .when(pl.col("RUBRICA").str.contains("ROUBO")).then(pl.lit(5))
            .otherwise(pl.lit(1)).alias("LABEL_PESO_RISCO")
        ])

        # 4. JOIN FINAL: EVENTO + CONTEXTO URBANO
        print("🏙️ Realizando enriquecimento espacial (H3)...", flush=True)
        df_final = df_gold.join(df_universo_h3, on="H3_INDEX", how="left")
        
        # Garante que colunas de infraestrutura não tenham nulos (essencial para CatBoost/XGBoost)
        cols_infra = [c for c in df_final.columns if "INFRA_" in c or "CENSO_" in c or "MICRO_" in c]
        df_final = df_final.with_columns([pl.col(c).fill_null(0) for c in cols_infra])

        # 5. SALVAMENTO NO R2 (O QUE ESTAVA FALTANDO)
        print("📦 Gravando ABT Final no Data Lake...", flush=True)
        
        # Salvando o Parquet de Treino
        buf_parquet = io.BytesIO()
        df_final.write_parquet(buf_parquet, compression="zstd")
        self.s3.put_object(
            Bucket=self.bucket, 
            Key=f"{self.ouro_dir}/safedriver_abt_treino.parquet", 
            Body=buf_parquet.getvalue()
        )

        # 6. FINALIZAÇÃO E AUDITORIA
        duracao = round(time.time() - inicio_timer, 2)
        self.auditoria["metricas"] = {
            "linhas_processadas": df_final.height,
            "colunas_totais": len(df_final.columns),
            "tempo_execucao_segundos": duracao,
            "memoria_estimada_mb": round(df_final.estimated_size() / (1024 * 1024), 2)
        }

        # Salva o JSON de auditoria da Ouro
        buf_json = io.BytesIO(json.dumps(self.auditoria, indent=4).encode())
        self.s3.put_object(
            Bucket=self.bucket, 
            Key=f"{self.ouro_dir}/auditoria/AUDITORIA_OURO_FINAL.json", 
            Body=buf_json.getvalue()
        )

        # Notificação Final
        msg = (
            f"🏆 **[SafeDriver] ABT Gold Gerada com Sucesso!**\n"
            f"```ml\n"
            f"• Registros para IA: {df_final.height}\n"
            f"• Features Criadas : {len(df_final.columns)}\n"
            f"• Tamanho em RAM   : {self.auditoria['metricas']['memoria_estimada_mb']} MB\n"
            f"• Tempo de Spark   : {duracao}s\n"
            f"-----------------------------------\n"
            f"Status: PRONTO PARA MODELAGEM\n"
            f"```"
        )
        self._notificar_discord(msg)
        print(f"✨ Processamento concluído! ABT salva em: {self.ouro_dir}/safedriver_abt_treino.parquet")

if __name__ == "__main__":
    app = ArquitetoSafeDriverOuro()
    app.construir_abt_final()
