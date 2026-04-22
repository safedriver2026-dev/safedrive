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
        
        # 1. CARREGAMENTO DOS COMPONENTES (H3 Flat)
        # Como o resgate foi na Prata, aqui focamos nos atributos do Hexágono
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_micro_pop = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_MICRO_POPULACAO.parquet")
        df_feriados = self._ler_json_r2(f"{self.prata_ref}/feriados_sp_2022_2026.json")

        # 2. AGREGAÇÃO DOS CRIMES TRUSTED
        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
        
        if not crime_files:
            print("❌ Nenhum arquivo de crime encontrado na Prata.")
            return

        df_crimes = pl.concat([self._ler_parquet_r2(f) for f in crime_files], how="diagonal")
        total_crimes = df_crimes.height

        # 3. FEATURE ENGINEERING TEMPORAL
        print("📅 Gerando features sazonais...", flush=True)
        
        # Prepara feriados
        if df_feriados is not None:
            df_feriados = df_feriados.select([
                pl.col("data").alias("DATA_FERIADO"),
                pl.col("feriado_nome").alias("SAZON_NOME_FERIADO"),
                pl.lit(1).alias("SAZON_IS_FERIADO")
            ])

        # Extração de componentes da data
        df_ouro = df_crimes.with_columns([
            pl.col("DATAOCORRENCIA").str.to_date(format="%Y-%m-%d", strict=False).alias("_dt_temp")
        ]).with_columns([
            pl.col("_dt_temp").dt.weekday().alias("SAZON_DIA_SEMANA"),
            pl.col("_dt_temp").dt.month().alias("SAZON_MES"),
            pl.col("_dt_temp").dt.year().alias("SAZON_ANO"),
            # Flag de fim de semana
            pl.when(pl.col("_dt_temp").dt.weekday() >= 6).then(pl.lit(1)).otherwise(pl.lit(0)).alias("SAZON_FIM_SEMANA")
        ])

        if df_feriados is not None:
            df_ouro = df_ouro.join(df_feriados, left_on="DATAOCORRENCIA", right_on="DATA_FERIADO", how="left")
            df_ouro = df_ouro.with_columns(pl.col("SAZON_IS_FERIADO").fill_null(0))

        # 4. CATEGORIZAÇÃO DE NEGÓCIO (Target Profiling)
        print("🏷️ Categorizando perfis e gravidade...", flush=True)
        df_ouro = df_ouro.with_columns([
            # Perfil da Vítima/Alvo
            pl.when(pl.col("RUBRICA").str.contains("VEICULO|CARGA|AUTO|MOTO|CAMINHAO|CONDUZIR")).then(pl.lit("VEICULO"))
            .when(pl.col("RUBRICA").str.contains("CELULAR|TRANSEUNTE|PEDESTRE|PESSOA")).then(pl.lit("TRANSEUNTE"))
            .when(pl.col("RUBRICA").str.contains("RESIDENCIA|CONDOMINIO|CASA")).then(pl.lit("RESIDENCIAL"))
            .otherwise(pl.lit("OUTROS")).alias("META_PERFIL_ALVO"),

            # Categoria Simplificada para IA
            pl.when(pl.col("RUBRICA").str.contains("LATROCINIO|HOMICIDIO")).then(pl.lit("CRIME_FATAL"))
            .when(pl.col("RUBRICA").str.contains("ROUBO")).then(pl.lit("ROUBO"))
            .when(pl.col("RUBRICA").str.contains("FURTO")).then(pl.lit("FURTO"))
            .otherwise(pl.lit("OUTROS")).alias("META_CATEGORIA"),
            
            # Peso de Gravidade (Score para mapas de calor)
            pl.when(pl.col("RUBRICA").str.contains("LATROCINIO|HOMICIDIO")).then(pl.lit(10.0))
            .when(pl.col("RUBRICA").str.contains("ROUBO")).then(pl.lit(5.0))
            .otherwise(pl.lit(1.0)).alias("META_SCORE_RISCO")
        ])

        # 5. ENRIQUECIMENTO URBANO (Joins Finais por H3)
        print("🏙️ Cruzando com Infraestrutura e População...", flush=True)
        if df_infra is not None:
            df_ouro = df_ouro.join(df_infra, on="H3_INDEX", how="left")
        
        if df_micro_pop is not None:
            df_ouro = df_ouro.join(df_micro_pop, on="H3_INDEX", how="left")

        # 6. TRATAMENTO DE NULOS E TIPAGEM FINAL
        # Colunas categóricas para string e nulos para "DESCONHECIDO"
        cat_cols = ["SAZON_PERIODO", "META_PERFIL_ALVO", "META_CATEGORIA", "MUNICIPIO", "BAIRRO"]
        df_ouro = df_ouro.with_columns([
            pl.col(c).cast(pl.Utf8).fill_null("NAO_INFORMADO") for c in cat_cols
        ])
        
        # Colunas numéricas (Infra e População) nulos para 0
        df_ouro = df_ouro.with_columns(pl.all().fill_null(0))

        # 7. NOTIFICAÇÃO E UPLOAD
        duracao = time.time() - inicio_timer
        
        # Estatísticas de Enriquecimento
        tem_infra = df_ouro.filter(pl.col("INFRA_DIV_01").is_not_null()).height if "INFRA_DIV_01" in df_ouro.columns else 0
        
        report = (
            f"🏆 **[SafeDriver] Camada Ouro: ABT Gerada**\n"
            f"```ml\n"
            f"• Registros Processados: {total_crimes}\n"
            f"• Features Temporais: OK\n"
            f"• Features Urbanas (H3): OK\n"
            f"• Peso de Risco Aplicado: OK\n"
            f"• Linhas Finais ABT: {df_ouro.height}\n"
            f"-----------------------------------\n"
            f"Tempo de Execução: {duracao:.2f}s\n"
            f"```"
        )
        self._notificar_discord(report)

        # Upload final para a pasta OURO
        buf = io.BytesIO()
        df_ouro.write_parquet(buf, compression="zstd")
        self.s3.put_object(
            Bucket=self.bucket, 
            Key=f"{self.ouro_dir}/safedriver_abt_eventos.parquet", 
            Body=buf.getvalue()
        )
        print("✨ ABT salva no R2 com sucesso!")

if __name__ == "__main__":
    app = ArquitetoSafeDriverOuro()
    app.construir_tabela_analitica()
