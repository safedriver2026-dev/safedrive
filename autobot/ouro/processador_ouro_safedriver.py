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
        print("🚀 Iniciando Consolidação Gold (ABT - Analytical Base Table)...", flush=True)
        
        # 1. CARREGAMENTO DOS COMPONENTES
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_micro_pop = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_MICRO_POPULACAO.parquet")
        df_feriados = self._ler_json_r2(f"{self.prata_ref}/feriados_sp_2022_2026.json")

        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
        
        if not crime_files:
            print("❌ Nenhum crime encontrado na Prata.")
            return

        # Concatenação diagonal para lidar com possíveis variações de schema entre anos
        df_crimes = pl.concat([self._ler_parquet_r2(f) for f in crime_files], how="diagonal")
        total_entrada = df_crimes.height

        # 2. FEATURE ENGINEERING TEMPORAL (Baseada no Date real da Prata)
        print("📅 Enriquecendo Dimensão Temporal...", flush=True)
        if df_feriados is not None:
            df_feriados = df_feriados.select([
                pl.col("data").alias("DATA_FERIADO"),
                pl.lit(1).alias("SAZON_IS_FERIADO")
            ])

        df_gold = df_crimes.with_columns([
            pl.col("DATAOCORRENCIA").dt.weekday().alias("SAZON_DIA_SEMANA"),
            pl.col("DATAOCORRENCIA").dt.month().alias("SAZON_MES"),
            pl.col("DATAOCORRENCIA").dt.year().alias("SAZON_ANO"),
            pl.when(pl.col("DATAOCORRENCIA").dt.weekday() >= 6).then(pl.lit(1)).otherwise(pl.lit(0)).alias("SAZON_FIM_SEMANA")
        ])

        if df_feriados is not None:
            df_gold = df_gold.join(df_feriados, left_on="DATAOCORRENCIA", right_on="DATA_FERIADO", how="left")
            df_gold = df_gold.with_columns(pl.col("SAZON_IS_FERIADO").fill_null(0))

        # 3. TRATAMENTO DO "INCERTO" E CATEGORIZAÇÃO
        # O CatBoost usará "INCERTO" como uma categoria de risco própria (comum em furtos)
        print("🏷️ Gerando Scores de Gravidade e Perfis...", flush=True)
        df_gold = df_gold.with_columns([
            # Perfil do Alvo
            pl.when(pl.col("RUBRICA").str.contains("VEICULO|CARGA|AUTO|MOTO")).then(pl.lit("VEICULO"))
            .when(pl.col("RUBRICA").str.contains("CELULAR|TRANSEUNTE|PEDESTRE")).then(pl.lit("TRANSEUNTE"))
            .otherwise(pl.lit("OUTROS")).alias("META_PERFIL_ALVO"),

            # Categorização para Treino
            pl.when(pl.col("RUBRICA").str.contains("LATROCINIO|HOMICIDIO")).then(pl.lit("FATAL"))
            .when(pl.col("RUBRICA").str.contains("ROUBO")).then(pl.lit("ROUBO"))
            .when(pl.col("RUBRICA").str.contains("FURTO")).then(pl.lit("FURTO"))
            .otherwise(pl.lit("OUTROS")).alias("META_CATEGORIA"),
            
            # Peso de Risco (Score SafeDriver)
            pl.when(pl.col("RUBRICA").str.contains("LATROCINIO|HOMICIDIO")).then(pl.lit(10.0))
            .when(pl.col("RUBRICA").str.contains("ROUBO")).then(pl.lit(5.0))
            .when(pl.col("RUBRICA").str.contains("FURTO")).then(pl.lit(2.0))
            .otherwise(pl.lit(1.0)).alias("META_SCORE_RISCO")
        ])

        # 4. ENRIQUECIMENTO URBANO (Joins por Hexágono H3)
        print("🏙️ Integrando Infraestrutura e Micro-População...", flush=True)
        df_gold = df_gold.join(df_infra, on="H3_INDEX", how="left")
        df_gold = df_gold.join(df_micro_pop, on="H3_INDEX", how="left")

        # 5. ATESTADO DE SAÚDE DA GOLD (Auditoria de Densidade)
        infra_cols = [c for c in df_gold.columns if "INFRA_DIV_" in c]
        rows_with_infra = df_gold.filter(pl.any_horizontal(pl.col(infra_cols).is_not_null())).height if infra_cols else 0
        incertos_count = df_gold.filter(pl.col("SAZON_PERIODO") == "INCERTO").height
        
        dist_classes = df_gold.group_by("META_CATEGORIA").len().to_dicts()
        
        self.auditoria["metricas"] = {
            "total_registros_abt": df_gold.height,
            "aproveitamento_infra_urbana": f"{round((rows_with_infra / total_entrada) * 100, 2)}%",
            "volume_crimes_incertos": f"{round((incertos_count / total_entrada) * 100, 2)}%",
            "distribuicao": {d["META_CATEGORIA"]: d["len"] for d in dist_classes},
            "periodo_dados": {
                "inicio": str(df_gold["DATAOCORRENCIA"].min()),
                "fim": str(df_gold["DATAOCORRENCIA"].max())
            }
        }

        # 6. FINALIZAÇÃO E TIPAGEM PARA PRODUÇÃO
        # Preenchemos nulos de infra com 0 e categorias com strings seguras
        df_gold = df_gold.with_columns([
            pl.col("SAZON_PERIODO").fill_null("INCERTO"),
            pl.col("MUNICIPIO").fill_null("NAO_INFORMADO"),
            pl.col("BAIRRO").fill_null("NAO_INFORMADO")
        ])
        df_gold = df_gold.with_columns(pl.all().fill_null(0))

        # 7. EXPORTAÇÃO E FEEDBACK
        # Salva ABT Final
        buf_abt = io.BytesIO()
        df_gold.write_parquet(buf_abt, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/safedriver_abt_eventos.parquet", Body=buf_abt.getvalue())

        # Salva Auditoria JSON
        buf_audit = io.BytesIO(json.dumps(self.auditoria, indent=4).encode())
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/auditoria/AUDITORIA_QUALIDADE_OURO.json", Body=buf_audit.getvalue())

        duracao = time.time() - inicio_timer
        report = (
            f"🏆 **[SafeDriver] Camada Gold: ABT Consolidada**\n"
            f"```ml\n"
            f"• Volume Final: {df_gold.height} registros\n"
            f"• Eficiência Infra: {self.auditoria['metricas']['aproveitamento_infra_urbana']}\n"
            f"• Crimes 'INCERTOS': {self.auditoria['metricas']['volume_crimes_incertos']} (Recuperados)\n"
            f"• Score de Gravidade: Aplicado\n"
            f"-----------------------------------\n"
            f"Status: PRONTO PARA TREINO | {duracao:.2f}s\n"
            f"```"
        )
        self._notificar_discord(report)
        print("✨ Camada Gold finalizada com sucesso!")

if __name__ == "__main__":
    app = ArquitetoSafeDriverOuro()
    app.construir_tabela_analitica()
