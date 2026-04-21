import os
import boto3
import polars as pl
import io
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

        self.prata_crimes = "datalake/prata/crimes_trusted"
        self.prata_malha = "datalake/prata/malha_trusted"
        self.prata_ref = "datalake/prata/referencias"
        self.ouro_dir = "datalake/ouro"

    def _ler_parquet_r2(self, key):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return pl.read_parquet(io.BytesIO(obj['Body'].read()))
        except:
            return None

    def _ler_json_r2(self, key):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return pl.read_json(io.BytesIO(obj['Body'].read()))
        except:
            return None

    def construir_tabela_analitica(self):
        df_vias = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_GEOGRAFICA_VIAS.parquet")
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_social = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_SOCIAL.parquet")
        df_micro_pop = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_MICRO_POPULACAO.parquet")
        df_feriados = self._ler_json_r2(f"{self.prata_ref}/feriados_sp_2022_2026.json")

        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] 
            for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) 
            if obj['Key'].endswith('.parquet')
        ]
            
        df_crimes = pl.concat([self._ler_parquet_r2(f) for f in crime_files], how="diagonal")

        h3_mun_map = df_crimes.filter(pl.col("H3_INDEX").is_not_null()) \
                              .group_by("H3_INDEX") \
                              .agg(pl.col("MUNICIPIO").mode().first().alias("MUN_H3"))
        
        df_vias_enriquecida = df_vias.join(h3_mun_map, on="H3_INDEX", how="inner")
        ref_vias = df_vias_enriquecida.group_by(["RUA", "MUN_H3"]).agg(pl.col("H3_INDEX").first())
        
        df_crimes = df_crimes.join(
            ref_vias,
            left_on=["LOGRADOURO", "MUNICIPIO"],
            right_on=["RUA", "MUN_H3"],
            how="left"
        ).with_columns(
            pl.col("H3_INDEX").fill_null(pl.col("H3_INDEX_right"))
        ).drop("H3_INDEX_right")

        df_crimes = df_crimes.filter(pl.col("H3_INDEX").is_not_null())

        df_feriados = df_feriados.select([
            pl.col("data").alias("DATA_FERIADO"),
            pl.col("feriado_nome").alias("SAZON_NOME_FERIADO"),
            pl.col("feriado_tipo").alias("SAZON_TIPO_FERIADO"),
            pl.col("is_ponto_facultativo").alias("SAZON_PONTO_FACULTATIVO")
        ])

        df_ouro = df_crimes.with_columns([
            pl.col("DATAOCORRENCIA").str.to_date(format="%Y-%m-%d", strict=False).dt.weekday().alias("SAZON_DIA_SEMANA"),
            pl.col("DATAOCORRENCIA").str.to_date(format="%Y-%m-%d", strict=False).dt.month().alias("SAZON_MES"),
            pl.col("HORAOCORRENCIA").str.to_time(format="%H:%M:%S", strict=False).dt.hour().alias("SAZON_HORA")
        ]).join(
            df_feriados, left_on="DATAOCORRENCIA", right_on="DATA_FERIADO", how="left"
        ).with_columns([
            pl.col("SAZON_NOME_FERIADO").fill_null("DIA NORMAL"),
            pl.col("SAZON_TIPO_FERIADO").fill_null("NENHUM"),
            pl.col("SAZON_PONTO_FACULTATIVO").fill_null(False)
        ])

        df_ouro = df_ouro.with_columns(
            pl.when(pl.col("SAZON_HORA").is_between(0, 5)).then(pl.lit("MADRUGADA"))
            .when(pl.col("SAZON_HORA").is_between(6, 11)).then(pl.lit("MANHA"))
            .when(pl.col("SAZON_HORA").is_between(12, 17)).then(pl.lit("TARDE"))
            .when(pl.col("SAZON_HORA").is_between(18, 23)).then(pl.lit("NOITE"))
            .otherwise(pl.lit(None))
            .alias("SAZON_PERIODO")
        )

        df_modas_h3 = df_ouro.filter(pl.col("SAZON_PERIODO").is_not_null()) \
                             .group_by("H3_INDEX") \
                             .agg(pl.col("SAZON_PERIODO").mode().first().alias("SAZON_PERIODO_IMPUTADO"))

        df_ouro = df_ouro.join(df_modas_h3, on="H3_INDEX", how="left")
        df_ouro = df_ouro.with_columns(
            pl.col("SAZON_PERIODO").fill_null(pl.col("SAZON_PERIODO_IMPUTADO"))
        ).drop("SAZON_PERIODO_IMPUTADO")

        df_ouro = df_ouro.filter(pl.col("SAZON_PERIODO").is_not_null())

        df_ouro = df_ouro.with_columns([
            pl.when(pl.col("RUBRICA").str.contains("VEICULO|CARGA|AUTO|MOTO|CAMINHAO"))
            .then(pl.lit("MOTORISTA"))
            .when(pl.col("RUBRICA").str.contains("CELULAR|TRANSEUNTE|PEDESTRE|PESSOA"))
            .then(pl.lit("PEDESTRE"))
            .otherwise(pl.lit("OUTROS"))
            .alias("META_PERFIL_VITIMA"),

            pl.when(pl.col("RUBRICA").str.contains("LATROCINIO|HOMICIDIO"))
            .then(pl.lit("CRIME_LETO"))
            .when(pl.col("RUBRICA").str.contains("ROUBO"))
            .then(pl.lit("ROUBO"))
            .when(pl.col("RUBRICA").str.contains("FURTO"))
            .then(pl.lit("FURTO"))
            .otherwise(pl.lit("OUTROS"))
            .alias("META_CATEGORIA_CRIME"),
            
            pl.when(pl.col("RUBRICA").str.contains("LATROCINIO|HOMICIDIO"))
            .then(pl.lit(10.0))
            .when(pl.col("RUBRICA").str.contains("ROUBO"))
            .then(pl.lit(5.0))
            .when(pl.col("RUBRICA").str.contains("FURTO"))
            .then(pl.lit(1.0))
            .otherwise(pl.lit(0.5))
            .alias("META_PESO_GRAVIDADE")
        ])

        df_ouro = df_ouro.join(df_infra, on="H3_INDEX", how="left")
        cols_infra = [c for c in df_ouro.columns if c.startswith("INFRA_DIV_")]
        df_ouro = df_ouro.with_columns([pl.col(c).fill_null(0) for c in cols_infra])

        cnaes_lazer = ["56", "90", "93"]
        cols_lazer = [f"INFRA_DIV_{x}" for x in cnaes_lazer if f"INFRA_DIV_{x}" in df_ouro.columns]
        df_ouro = df_ouro.with_columns(pl.sum_horizontal(cols_lazer).alias("META_POLO_LAZER_NOTURNO") if cols_lazer else pl.lit(0).alias("META_POLO_LAZER_NOTURNO"))

        cnaes_deserto = ["41", "42", "43", "46", "49", "52"] + [str(i) for i in range(10, 34)]
        cols_deserto = [f"INFRA_DIV_{x}" for x in cnaes_deserto if f"INFRA_DIV_{x}" in df_ouro.columns]
        df_ouro = df_ouro.with_columns(pl.sum_horizontal(cols_deserto).alias("META_DESERTO_URBANO") if cols_deserto else pl.lit(0).alias("META_DESERTO_URBANO"))

        cnaes_diurno = ["47", "64", "65", "66", "84", "85", "86"]
        cols_diurno = [f"INFRA_DIV_{x}" for x in cnaes_diurno if f"INFRA_DIV_{x}" in df_ouro.columns]
        df_ouro = df_ouro.with_columns(pl.sum_horizontal(cols_diurno).alias("META_FLUXO_DIURNO") if cols_diurno else pl.lit(0).alias("META_FLUXO_DIURNO"))

        df_ouro = df_ouro.join(df_micro_pop, on="H3_INDEX", how="left")
        df_ouro = df_ouro.with_columns(pl.col("MICRO_POPULACAO_H3").fill_null(0))

        df_social_agg = df_social.with_columns([
            pl.col("v0001").cast(pl.Float64, strict=False).fill_null(0),
            pl.col("v0002").cast(pl.Float64, strict=False).fill_null(0)
        ]).group_by(["NM_MUN", "NM_BAIRRO"]).agg([
            pl.sum("v0001").alias("META_POPULACAO_BAIRRO"),
            pl.sum("v0002").alias("META_DOMICILIOS_BAIRRO")
        ])

        df_ouro = df_ouro.join(
            df_social_agg, 
            left_on=["MUNICIPIO", "BAIRRO"], 
            right_on=["NM_MUN", "NM_BAIRRO"], 
            how="left"
        ).with_columns([
            pl.col("META_POPULACAO_BAIRRO").fill_null(0),
            pl.col("META_DOMICILIOS_BAIRRO").fill_null(0)
        ])

        cols_cat = [
            "H3_INDEX", "SAZON_DIA_SEMANA", "SAZON_MES", "SAZON_HORA", 
            "SAZON_PERIODO", "SAZON_NOME_FERIADO", "SAZON_TIPO_FERIADO", 
            "META_CATEGORIA_CRIME", "META_PERFIL_VITIMA"
        ]
        df_ouro = df_ouro.with_columns([
            pl.col(c).cast(pl.Utf8).fill_null("NAO_INFORMADO") for c in cols_cat
        ])
        
        cols_num = [
            "META_PESO_GRAVIDADE", "MICRO_POPULACAO_H3", 
            "META_POPULACAO_BAIRRO", "META_DOMICILIOS_BAIRRO"
        ]
        cols_infra_final = [c for c in df_ouro.columns if c.startswith("INFRA_DIV_") or c.startswith("META_POLO_")]
        
        df_ouro = df_ouro.with_columns([
            pl.col(c).fill_null(0.0) for c in (cols_num + cols_infra_final) if c not in cols_cat
        ])
        
        df_ouro = df_ouro.with_columns(pl.col("SAZON_PONTO_FACULTATIVO").fill_null(False))

        buf = io.BytesIO()
        df_ouro.write_parquet(buf, compression="zstd")
        
        caminho_final = f"{self.ouro_dir}/safedriver_abt_eventos.parquet"
        self.s3.put_object(Bucket=self.bucket, Key=caminho_final, Body=buf.getvalue())

if __name__ == "__main__":
    app = ArquitetoSafeDriverOuro()
    app.construir_tabela_analitica()
