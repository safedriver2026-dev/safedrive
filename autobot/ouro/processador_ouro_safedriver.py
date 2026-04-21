import os
import boto3
import polars as pl
import io
import requests
import time
from botocore.config import Config

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
            requests.post(self.webhook_url, json={"content": msg})

    def _ler_parquet_r2(self, key):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return pl.read_parquet(io.BytesIO(obj['Body'].read()))
        except: return None

    def _ler_json_r2(self, key):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return pl.read_json(io.BytesIO(obj['Body'].read()))
        except: return None

    def construir_tabela_analitica(self):
        inicio_timer = time.time()
        print("Carregando bases da camada prata...", flush=True)
        
        df_vias = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_GEOGRAFICA_VIAS.parquet")
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_social = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_SOCIAL.parquet")
        df_micro_pop = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_MICRO_POPULACAO.parquet")
        df_feriados = self._ler_json_r2(f"{self.prata_ref}/feriados_sp_2022_2026.json")

        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
            
        df_crimes = pl.concat([self._ler_parquet_r2(f) for f in crime_files], how="diagonal")
        
        # MÉTRICA 1: Volume Inicial
        total_inicial = df_crimes.height
        bo_com_gps_inicial = df_crimes.filter(pl.col("H3_INDEX").is_not_null()).height

        print(f"Executando Geocoding de Resgate em {total_inicial} registros...", flush=True)
        h3_mun_map = df_crimes.filter(pl.col("H3_INDEX").is_not_null()) \
                              .group_by("H3_INDEX") \
                              .agg(pl.col("MUNICIPIO").mode().first().alias("MUN_H3"))
        
        df_vias_enriquecida = df_vias.join(h3_mun_map, on="H3_INDEX", how="inner")
        ref_vias = df_vias_enriquecida.group_by(["RUA", "MUN_H3"]).agg(pl.col("H3_INDEX").first())
        
        df_crimes = df_crimes.join(
            ref_vias, left_on=["LOGRADOURO", "MUNICIPIO"], right_on=["RUA", "MUN_H3"], how="left"
        ).with_columns(
            pl.col("H3_INDEX").fill_null(pl.col("H3_INDEX_right"))
        ).drop("H3_INDEX_right")

        # Filtro de Localização (Cérebro da ABT)
        df_ouro = df_crimes.filter(pl.col("H3_INDEX").is_not_null())
        
        # MÉTRICA 2: Eficiência da Recuperação
        total_final = df_ouro.height
        bo_recuperados = total_final - bo_com_gps_inicial
        taxa_recuperacao = (bo_recuperados / (total_inicial - bo_com_gps_inicial)) * 100 if (total_inicial - bo_com_gps_inicial) > 0 else 0

        # --- PROCESSAMENTO TEMPORAL E ESPACIAL (Igual à versão equalizada) ---
        df_feriados = df_feriados.select([
            pl.col("data").alias("DATA_FERIADO"),
            pl.col("feriado_nome").alias("SAZON_NOME_FERIADO"),
            pl.col("feriado_tipo").alias("SAZON_TIPO_FERIADO"),
            pl.col("is_ponto_facultativo").alias("SAZON_PONTO_FACULTATIVO")
        ])

        df_ouro = df_ouro.with_columns([
            pl.col("DATAOCORRENCIA").str.to_date(format="%Y-%m-%d", strict=False).dt.weekday().alias("SAZON_DIA_SEMANA"),
            pl.col("DATAOCORRENCIA").str.to_date(format="%Y-%m-%d", strict=False).dt.month().alias("SAZON_MES"),
            pl.col("HORAOCORRENCIA").str.to_time(format="%H:%M:%S", strict=False).dt.hour().alias("SAZON_HORA")
        ]).join(df_feriados, left_on="DATAOCORRENCIA", right_on="DATA_FERIADO", how="left")

        # Hierarquia de Período
        df_ouro = df_ouro.with_columns(
            pl.when(pl.col("SAZON_HORA").is_between(0, 5)).then(pl.lit("MADRUGADA"))
            .when(pl.col("SAZON_HORA").is_between(6, 11)).then(pl.lit("MANHA"))
            .when(pl.col("SAZON_HORA").is_between(12, 17)).then(pl.lit("TARDE"))
            .when(pl.col("SAZON_HORA").is_between(18, 23)).then(pl.lit("NOITE"))
            .otherwise(pl.col("SAZON_PERIODO"))
            .alias("SAZON_PERIODO")
        )

        moda_global = df_ouro.filter(pl.col("SAZON_PERIODO").is_not_null())["SAZON_PERIODO"].mode().first()
        df_ouro = df_ouro.with_columns(pl.col("SAZON_PERIODO").fill_null(pl.lit(moda_global)))

        # Categorização e Enriquecimento
        df_ouro = df_ouro.with_columns([
            pl.when(pl.col("RUBRICA").str.contains("VEICULO|CARGA|AUTO|MOTO|CAMINHAO|CONDUZIR")).then(pl.lit("MOTORISTA"))
            .when(pl.col("RUBRICA").str.contains("CELULAR|TRANSEUNTE|PEDESTRE|PESSOA|ESTABELECIMENTO")).then(pl.lit("PEDESTRE"))
            .otherwise(pl.lit("OUTROS")).alias("META_PERFIL_VITIMA"),

            pl.when(pl.col("RUBRICA").str.contains("LATROCINIO|HOMICIDIO")).then(pl.lit("CRIME_LETO"))
            .when(pl.col("RUBRICA").str.contains("ROUBO")).then(pl.lit("ROUBO"))
            .when(pl.col("RUBRICA").str.contains("FURTO")).then(pl.lit("FURTO"))
            .otherwise(pl.lit("OUTROS")).alias("META_CATEGORIA_CRIME"),
            
            pl.when(pl.col("RUBRICA").str.contains("LATROCINIO|HOMICIDIO")).then(pl.lit(10.0))
            .when(pl.col("RUBRICA").str.contains("ROUBO")).then(pl.lit(5.0))
            .when(pl.col("RUBRICA").str.contains("FURTO")).then(pl.lit(1.0))
            .otherwise(pl.lit(0.5)).alias("META_PESO_GRAVIDADE")
        ])

        # Joins Finais (Infra e Social)
        df_ouro = df_ouro.join(df_infra, on="H3_INDEX", how="left")
        df_ouro = df_ouro.join(df_micro_pop, on="H3_INDEX", how="left")
        
        # Blindagem de Nulos
        cols_cat = ["H3_INDEX", "SAZON_DIA_SEMANA", "SAZON_MES", "SAZON_HORA", "SAZON_PERIODO", "META_CATEGORIA_CRIME", "META_PERFIL_VITIMA"]
        df_ouro = df_ouro.with_columns([pl.col(c).cast(pl.Utf8).fill_null("NAO_INFORMADO") for c in cols_cat])
        df_ouro = df_ouro.with_columns(pl.all().fill_null(0))

        # --- NOTIFICAÇÃO DISCORD ---
        duracao = time.time() - inicio_timer
        report_msg = (
            f"🏆 **[SafeDriver] Camada Ouro Finalizada**\n"
            f"```ml\n"
            f"ESTATÍSTICAS DE RECUPERAÇÃO:\n"
            f"• B.O.s Totais (Bronze/Prata): {total_inicial}\n"
            f"• B.O.s Salvos pelo Resgate: {bo_recuperados}\n"
            f"• Taxa de Sucesso do Geocoding: {taxa_recuperacao:.2f}%\n"
            f"• Base Final (ABT): {total_final} linhas\n"
            f"-------------------------------------\n"
            f"Tempo de Processamento: {duracao:.2f}s\n"
            f"```"
        )
        self._notificar_discord(report_msg)

        # Upload
        buf = io.BytesIO()
        df_ouro.write_parquet(buf, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/safedriver_abt_eventos.parquet", Body=buf.getvalue())

if __name__ == "__main__":
    app = ArquitetoSafeDriverOuro()
    app.construir_tabela_analitica()
