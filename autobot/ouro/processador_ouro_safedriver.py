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
        self.prata_ref = "datalake/prata/referencias"
        self.ouro_dir = "datalake/ouro"
        
        # Objeto de Auditoria
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
        except Exception as e:
            print(f"⚠️ Erro ao ler {key}: {e}")
            return None

    def _ler_json_r2(self, key):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return pl.read_json(io.BytesIO(obj['Body'].read()))
        except: return None

    def construir_tabela_analitica(self):
        inicio_timer = time.time()
        print("🚀 Iniciando Consolidação da Camada Ouro (ABT)...", flush=True)
        
        # 1. CARREGAMENTO
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_micro_pop = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_MICRO_POPULACAO.parquet")
        df_feriados = self._ler_json_r2(f"{self.prata_ref}/feriados_sp_2022_2026.json")

        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
        
        if not crime_files: return

        df_crimes = pl.concat([self._ler_parquet_r2(f) for f in crime_files], how="diagonal")
        total_entrada = df_crimes.height

        # 2. FEATURE ENGINEERING TEMPORAL
        if df_feriados is not None:
            df_feriados = df_feriados.select([
                pl.col("data").alias("DATA_FERIADO"),
                pl.lit(1).alias("SAZON_IS_FERIADO")
            ])

        df_ouro = df_crimes.with_columns([
            pl.col("DATAOCORRENCIA").str.to_date(format="%Y-%m-%d", strict=False).alias("_dt_temp")
        ]).with_columns([
            pl.col("_dt_temp").dt.weekday().alias("SAZON_DIA_SEMANA"),
            pl.col("_dt_temp").dt.month().alias("SAZON_MES"),
            pl.col("_dt_temp").dt.year().alias("SAZON_ANO"),
            pl.when(pl.col("_dt_temp").dt.weekday() >= 6).then(pl.lit(1)).otherwise(pl.lit(0)).alias("SAZON_FIM_SEMANA")
        ])

        if df_feriados is not None:
            df_ouro = df_ouro.join(df_feriados, left_on="DATAOCORRENCIA", right_on="DATA_FERIADO", how="left")
            df_ouro = df_ouro.with_columns(pl.col("SAZON_IS_FERIADO").fill_null(0))

        # 3. CATEGORIZAÇÃO E SCORE
        df_ouro = df_ouro.with_columns([
            pl.when(pl.col("RUBRICA").str.contains("VEICULO|CARGA|AUTO|MOTO|CAMINHAO")).then(pl.lit("VEICULO"))
            .when(pl.col("RUBRICA").str.contains("CELULAR|TRANSEUNTE|PEDESTRE")).then(pl.lit("TRANSEUNTE"))
            .otherwise(pl.lit("OUTROS")).alias("META_PERFIL_ALVO"),

            pl.when(pl.col("RUBRICA").str.contains("LATROCINIO|HOMICIDIO")).then(pl.lit("CRIME_FATAL"))
            .when(pl.col("RUBRICA").str.contains("ROUBO")).then(pl.lit("ROUBO"))
            .when(pl.col("RUBRICA").str.contains("FURTO")).then(pl.lit("FURTO"))
            .otherwise(pl.lit("OUTROS")).alias("META_CATEGORIA"),
            
            pl.when(pl.col("RUBRICA").str.contains("LATROCINIO|HOMICIDIO")).then(pl.lit(10.0))
            .when(pl.col("RUBRICA").str.contains("ROUBO")).then(pl.lit(5.0))
            .otherwise(pl.lit(1.0)).alias("META_SCORE_RISCO")
        ])

        # 4. ENRIQUECIMENTO URBANO
        df_ouro = df_ouro.join(df_infra, on="H3_INDEX", how="left")
        df_ouro = df_ouro.join(df_micro_pop, on="H3_INDEX", how="left")

        # 5. AUDITORIA DE SAÚDE (Coleta de dados antes de preencher nulos)
        infra_cols = [c for c in df_ouro.columns if "INFRA_DIV_" in c]
        rows_with_infra = df_ouro.filter(pl.any_horizontal(pl.col(infra_cols).is_not_null())).height if infra_cols else 0
        
        dist_crimes = df_ouro.group_by("META_CATEGORIA").len().to_dicts()
        
        self.auditoria["metricas"] = {
            "total_registros_abt": df_ouro.height,
            "enriquecimento_infra_percentual": round((rows_with_infra / total_entrada) * 100, 2),
            "periodo_coberto": {
                "inicio": str(df_ouro["_dt_temp"].min()),
                "fim": str(df_ouro["_dt_temp"].max())
            },
            "distribuicao_classes": {d["META_CATEGORIA"]: d["len"] for d in dist_crimes},
            "schema_check": "OK" if "H3_INDEX" in df_ouro.columns else "FAIL"
        }

        # 6. TRATAMENTO FINAL E UPLOAD
        cat_cols = ["SAZON_PERIODO", "META_PERFIL_ALVO", "META_CATEGORIA", "MUNICIPIO", "BAIRRO"]
        df_ouro = df_ouro.with_columns([pl.col(c).cast(pl.Utf8).fill_null("NAO_INFORMADO") for c in cat_cols])
        df_ouro = df_ouro.with_columns(pl.all().fill_null(0)).drop("_dt_temp")

        # Salvar Auditoria no R2
        buf_audit = io.BytesIO(json.dumps(self.auditoria, indent=4).encode())
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/auditoria/AUDITORIA_QUALIDADE_OURO.json", Body=buf_audit.getvalue())

        # Salvar ABT
        buf_abt = io.BytesIO()
        df_ouro.write_parquet(buf_abt, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/safedriver_abt_eventos.parquet", Body=buf_abt.getvalue())

        # Relatório Discord
        duracao = time.time() - inicio_timer
        report = (
            f"🏆 **[SafeDriver] Ouro: Atestado de Saúde Gerado**\n"
            f"```ml\n"
            f"• ABT Final: {df_ouro.height} registros\n"
            f"• Enriquecimento Urbano: {self.auditoria['metricas']['enriquecimento_infra_percentual']}%\n"
            f"• Classes: {self.auditoria['metricas']['distribuicao_classes']}\n"
            f"• Score de Risco: Aplicado\n"
            f"-----------------------------------\n"
            f"Status: SAUDÁVEL | Tempo: {duracao:.2f}s\n"
            f"```"
        )
        self._notificar_discord(report)
        print("✨ Camada Ouro e Auditoria finalizadas!")

if __name__ == "__main__":
    app = ArquitetoSafeDriverOuro()
    app.construir_tabela_analitica()
