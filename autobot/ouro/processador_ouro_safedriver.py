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
            "fase": "ABT Master + Holiday Integration + Feature Store",
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
        print("🚀 [OURO] Iniciando Consolidação...", flush=True)
        
        # 1. Carregamento da Malha (Infra + Social)
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_social = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_SOCIAL_H3.parquet")
        
        if df_infra is None: raise FileNotFoundError("Infraestrutura ausente.")
        df_universo_h3 = df_infra.join(df_social, on="H3_INDEX", how="full", coalesce=True).fill_null(0)

        # 2. Carregamento dos Crimes
        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
        
        lista_crimes = [df for f in crime_files if (df := self._ler_parquet_r2(f)) is not None]
        df_crimes = pl.concat(lista_crimes, how="diagonal").filter(pl.col("H3_INDEX").is_not_null())

        # 3. Integração de Feriados (Processando o seu JSON)
        print("🗓️  Acoplando Calendário de Feriados...", flush=True)
        try:
            obj_feriados = self.s3.get_object(Bucket=self.bucket, Key="datalake/prata/referencias/feriados_sp_2022_2026.json")
            json_feriados = json.loads(obj_feriados['Body'].read().decode('utf-8'))
            
            # Converte lista de objetos JSON para DataFrame
            df_feriados = pl.from_dicts(json_feriados)
            df_feriados = df_feriados.select([
                pl.col("data").cast(pl.Date).alias("DATA_CHAVE"),
                pl.lit(1).alias("FEAT_IS_FERIADO"),
                pl.col("feriado_tipo").alias("FEAT_TIPO_FERIADO"),
                pl.col("is_ponto_facultativo").cast(pl.Int8).alias("FEAT_IS_PONTO_FACULTATIVO")
            ])
            
            df_crimes = df_crimes.with_columns(pl.col("DATAOCORRENCIA").cast(pl.Date, strict=False))
            df_crimes = df_crimes.join(df_feriados, left_on="DATAOCORRENCIA", right_on="DATA_CHAVE", how="left")
            
            # Preenchimento para dias úteis normais
            df_crimes = df_crimes.with_columns([
                pl.col("FEAT_IS_FERIADO").fill_null(0),
                pl.col("FEAT_TIPO_FERIADO").fill_null("DIA_UTIL"),
                pl.col("FEAT_IS_PONTO_FACULTATIVO").fill_null(0)
            ])
        except Exception as e:
            print(f"⚠️ Erro nos feriados: {e}. Prosseguindo com flag zero.")
            df_crimes = df_crimes.with_columns([
                pl.lit(0).alias("FEAT_IS_FERIADO"),
                pl.lit("DIA_UTIL").alias("FEAT_TIPO_FERIADO"),
                pl.lit(0).alias("FEAT_IS_PONTO_FACULTATIVO")
            ])

        # 4. Feature Engineering: Dosimetria Penal e Perfil
        print("⚖️  Aplicando Dosimetria Penal...", flush=True)
        df_gold = df_crimes.with_columns([
            pl.col("RUBRICA").fill_null("").str.to_uppercase().alias("RUBRICA_UPPER"),
            pl.col("DATAOCORRENCIA").dt.year().alias("ANO_OCORRENCIA"),
            pl.col("DATAOCORRENCIA").dt.weekday().alias("FEAT_DIA_SEMANA"),
            pl.col("DATAOCORRENCIA").dt.month().alias("FEAT_MES")
        ]).with_columns([
            # Perfil Vítima
            pl.when(pl.col("RUBRICA_UPPER").str.contains(r"VEICULO|CARGA")).then(pl.lit("MOTORISTA"))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"TRANSEUNTE|CELULAR")).then(pl.lit("PEDESTRE"))
            .otherwise(pl.lit("GERAL")).alias("FEAT_PERFIL_VITIMA"),

            # Pesos Penais
            pl.when(pl.col("RUBRICA_UPPER").str.contains(r"ART\s*121|LATROC")).then(pl.lit(10.0))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"ART\s*158|SEQUESTRO")).then(pl.lit(8.0))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"ART\s*33|TRAFICO")).then(pl.lit(7.0))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"ART\s*157|ROUBO")).then(pl.lit(5.0))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"ART\s*129|LESAO")).then(pl.lit(4.0))
            .when(pl.col("RUBRICA_UPPER").str.contains(r"ART\s*155|FURTO")).then(pl.lit(2.0))
            .otherwise(pl.lit(1.0)).alias("LABEL_PESO_RISCO")
        ])

        # 5. Feature Store (Lag 1 Ano) e Materialização
        print("🏪 Materializando Feature Store...", flush=True)
        df_fs_hex = df_gold.group_by(["H3_INDEX", "ANO_OCORRENCIA"]).agg([
            pl.len().alias("FS_VOL_CRIMES_ANO_ANT"),
            pl.col("LABEL_PESO_RISCO").sum().alias("FS_RISCO_TOTAL_ANO_ANT"),
            pl.col("LABEL_PESO_RISCO").mean().alias("FS_RISCO_MEDIO_ANO_ANT")
        ]).with_columns((pl.col("ANO_OCORRENCIA") + 1).alias("ANO_JOIN"))

        # Salva tabelas da FS separadas
        for name, df_fs in [("macro", df_fs_hex)]:
            buf = io.BytesIO(); df_fs.drop("ANO_JOIN").write_parquet(buf, compression="zstd")
            self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/feature_store/fs_{name}.parquet", Body=buf.getvalue())

        # 6. Join Final ABT
        df_final = df_gold.join(df_universo_h3, on="H3_INDEX", how="left") \
                          .join(df_fs_hex.drop("ANO_OCORRENCIA"), left_on=["H3_INDEX", "ANO_OCORRENCIA"], right_on=["H3_INDEX", "ANO_JOIN"], how="left")
        
        # Limpeza e Salvamento
        cols_fill = [c for c in df_final.columns if any(x in c for x in ["INFRA_", "CENSO_", "FS_"])]
        df_final = df_final.with_columns([pl.col(c).fill_null(0) for c in cols_fill]).drop(["RUBRICA_UPPER", "ANO_OCORRENCIA"])

        buf_abt = io.BytesIO(); df_final.write_parquet(buf_abt, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/safedriver_abt_treino.parquet", Body=buf_abt.getvalue())

        # Auditoria e Log
        duracao = round(time.time() - inicio_timer, 2)
        report = f"✅ Ouro Finalizada | Linhas: {df_final.height} | Hit Rate FS: {df_final.filter(pl.col('FS_VOL_CRIMES_ANO_ANT') > 0).height / df_final.height:.1%}"
        print(report)
        
        # Correção da última linha: enviando o relatório e o tempo para o Discord.
        self._notificar_discord(f"{report} | Tempo: {duracao}s")
