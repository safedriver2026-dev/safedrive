import os, boto3, polars as pl, io, json, time
from botocore.config import Config
from datetime import datetime

class ArquitetoSafeDriverOuro:
    def __init__(self):
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.s3 = boto3.client('s3', endpoint_url=endpoint, aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
                               aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(), config=Config(signature_version='s3v4'))
        self.prata_crimes, self.prata_malha, self.ouro_dir = "datalake/prata/crimes_trusted", "datalake/prata/malha_trusted", "datalake/ouro"

    def _ler_parquet_r2(self, key):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return pl.read_parquet(io.BytesIO(obj['Body'].read()))
        except: return None

    def construir_abt_final(self):
        inicio_timer = time.time()
        print("🚀 [OURO] Iniciando Consolidação com Macro Classes...", flush=True)
        
        # =================================================================
        # 1. MALHA FÍSICA E SOCIAL
        # =================================================================
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_social = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_SOCIAL_H3.parquet")
        df_universo_h3 = df_infra.join(df_social, on="H3_INDEX", how="full", coalesce=True).fill_null(0)

        # =================================================================
        # 2. EVENTOS CRIMINAIS
        # =================================================================
        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/") if obj['Key'].endswith('.parquet')]
        df_crimes = pl.concat([df for f in crime_files if (df := self._ler_parquet_r2(f)) is not None], how="diagonal").filter(pl.col("H3_INDEX").is_not_null())

        # =================================================================
        # 3. CALENDÁRIO INTELIGENTE
        # =================================================================
        try:
            obj_feriados = self.s3.get_object(Bucket=self.bucket, Key="datalake/prata/referencias/feriados_sp_2022_2026.json")
            df_feriados = pl.DataFrame(json.loads(obj_feriados['Body'].read().decode('utf-8')))
            df_feriados = df_feriados.select([pl.col("data").str.to_date("%Y-%m-%d").alias("DATA_CHAVE"), pl.lit(1).alias("FEAT_IS_FERIADO"), pl.col("feriado_tipo").alias("FEAT_TIPO_FERIADO")])
            df_crimes = df_crimes.with_columns(pl.col("DATAOCORRENCIA").cast(pl.Date, strict=False))
            df_crimes = df_crimes.join(df_feriados, left_on="DATAOCORRENCIA", right_on="DATA_CHAVE", how="left")
            df_crimes = df_crimes.with_columns([pl.col("FEAT_IS_FERIADO").fill_null(0), pl.col("FEAT_TIPO_FERIADO").fill_null("DIA_UTIL")])
        except:
            df_crimes = df_crimes.with_columns([pl.lit(0).alias("FEAT_IS_FERIADO"), pl.lit("DIA_UTIL").alias("FEAT_TIPO_FERIADO")])

        # =================================================================
        # 4. DOSIMETRIA PENAL E CONTEXTO
        # =================================================================
        df_gold = df_crimes.with_columns([
            pl.col("RUBRICA").fill_null("").str.to_uppercase().alias("RUBRICA_UPPER"),
            pl.col("DATAOCORRENCIA").dt.year().alias("ANO_OCORRENCIA"),
            pl.col("DATAOCORRENCIA").dt.weekday().alias("FEAT_DIA_SEMANA"),
            pl.col("DATAOCORRENCIA").dt.month().alias("FEAT_MES"),
            pl.col("DATAOCORRENCIA").dt.weekday().is_in([6, 7]).cast(pl.Int8).alias("FEAT_IS_FDS")
        ]).with_columns([
            pl.when(pl.col("RUBRICA_UPPER").str.contains(r"VEICULO|CARGA")).then(pl.lit("MOTORISTA")).when(pl.col("RUBRICA_UPPER").str.contains(r"TRANSEUNTE|CELULAR")).then(pl.lit("PEDESTRE")).otherwise(pl.lit("GERAL")).alias("FEAT_PERFIL_VITIMA"),
            pl.when(pl.col("RUBRICA_UPPER").str.contains(r"ART\s*121|LATROC")).then(pl.lit(10.0)).when(pl.col("RUBRICA_UPPER").str.contains(r"ART\s*158|SEQUESTRO")).then(pl.lit(8.0)).when(pl.col("RUBRICA_UPPER").str.contains(r"ART\s*33|TRAFICO")).then(pl.lit(7.0)).when(pl.col("RUBRICA_UPPER").str.contains(r"ART\s*157|ROUBO")).then(pl.lit(5.0)).when(pl.col("RUBRICA_UPPER").str.contains(r"ART\s*155|FURTO")).then(pl.lit(2.0)).otherwise(pl.lit(1.0)).alias("LABEL_PESO_RISCO")
        ])

        # =================================================================
        # 5. FEATURE STORE HISTÓRICA
        # =================================================================
        print("🏪 Materializando Feature Store...", flush=True)
        df_fs = df_gold.group_by(["H3_INDEX", "ANO_OCORRENCIA"]).agg([pl.len().alias("FS_VOL_CRIMES_ANO_ANT"), pl.col("LABEL_PESO_RISCO").mean().alias("FS_RISCO_MEDIO_ANO_ANT")]).with_columns((pl.col("ANO_OCORRENCIA") + 1).alias("ANO_JOIN"))
        
        buf_fs = io.BytesIO(); df_fs.drop("ANO_JOIN").write_parquet(buf_fs, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/feature_store/fs_macro_hex_historico.parquet", Body=buf_fs.getvalue())

        # =================================================================
        # 6. JOIN FINAL E REDUÇÃO DE DIMENSIONALIDADE (MACRO CLASSES)
        # =================================================================
        print("🏙️ Agrupando CNAEs em Macro Classes de Risco...", flush=True)
        df_final = df_gold.join(df_universo_h3, on="H3_INDEX", how="left").join(df_fs.drop("ANO_OCORRENCIA"), left_on=["H3_INDEX", "ANO_OCORRENCIA"], right_on=["H3_INDEX", "ANO_JOIN"], how="left")
        
        # Limpeza de nulos inicial
        cols_fill = [c for c in df_final.columns if any(x in c for x in ["INFRA_", "CENSO_", "FS_"])]
        df_final = df_final.with_columns([pl.col(c).fill_null(0) for c in cols_fill])

        # 6.1 Dicionário de Agrupamento
        cnae_macros = {
            "MACRO_FINANCEIRO": ["INFRA_DIV_64", "INFRA_DIV_65", "INFRA_DIV_66"],
            "MACRO_LAZER_NOTURNO": ["INFRA_DIV_56", "INFRA_DIV_90", "INFRA_DIV_93"],
            "MACRO_VAREJO": ["INFRA_DIV_45", "INFRA_DIV_47"],
            "MACRO_LOGISTICA_INDUSTRIA": ["INFRA_DIV_49", "INFRA_DIV_52", "INFRA_DIV_53"] + [f"INFRA_DIV_{i}" for i in range(10, 34)],
            "MACRO_SERVICOS_BASE": ["INFRA_DIV_84", "INFRA_DIV_85", "INFRA_DIV_86"]
        }

        # 6.2 Criação das colunas sintéticas
        for macro_name, div_list in cnae_macros.items():
            cols_existentes = [c for c in div_list if c in df_final.columns]
            if cols_existentes:
                df_final = df_final.with_columns(pl.sum_horizontal(cols_existentes).alias(macro_name))
            else:
                df_final = df_final.with_columns(pl.lit(0).alias(macro_name))

        # 6.3 Expurgo de Variáveis Brutas (Para forçar o modelo a usar as Macros)
        cols_cnae_brutos = [c for c in df_final.columns if c.startswith("INFRA_DIV_")]
        df_final = df_final.drop(cols_cnae_brutos + ["RUBRICA_UPPER", "ANO_OCORRENCIA"])

        # =================================================================
        # 7. SALVAMENTO DA ABT MASTER
        # =================================================================
        buf_abt = io.BytesIO(); df_final.write_parquet(buf_abt, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/safedriver_abt_treino.parquet", Body=buf_abt.getvalue())
        print(f"✅ Ouro Pronta: {df_final.height} linhas. {len(df_final.columns)} Features. Dimensionalidade reduzida com sucesso.")

if __name__ == "__main__":
    ArquitetoSafeDriverOuro().construir_abt_final()
