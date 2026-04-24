import os
import boto3
import polars as pl
import io
import time
import requests
import json
import unicodedata
import re
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

    def _notificar_discord(self, msg):
        if self.webhook_url:
            try: requests.post(self.webhook_url, json={"content": msg}, timeout=10)
            except: pass

    def _limpar_tabela_toda(self, df):
        """Passa o rolo compressor em todas as colunas de texto da tabela."""
        cols_texto = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Utf8]
        if not cols_texto:
            return df
        
        return df.with_columns([
            pl.col(c)
            .str.to_uppercase()
            .str.strip_chars()
            .str.replace_all(r"[ÁÀÂÃÄ]", "A")
            .str.replace_all(r"[ÉÈÊË]", "E")
            .str.replace_all(r"[ÍÌÎÏ]", "I")
            .str.replace_all(r"[ÓÒÔÕÖ]", "O")
            .str.replace_all(r"[ÚÙÛÜ]", "U")
            .str.replace_all(r"[Ç]", "C")
            .str.replace_all(r"[^A-Z0-9\s_]", " ") # Exclui TUDO que não é Alfanumérico ou underscore
            .str.replace_all(r"\s+", " ")
            .fill_null("DESCONHECIDO")
            .alias(c)
            for c in cols_texto
        ])

    def _ler_parquet_r2(self, key):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return pl.read_parquet(io.BytesIO(obj['Body'].read()))
        except Exception as e: 
            print(f"Erro ao ler {key}: {e}")
            return None

    def construir_abt_final(self):
        inicio_timer = time.time()
        print("Iniciando reconstrucao da Camada Ouro com Normalizacao Total...")
        
        # =================================================================
        # 1. CARREGAMENTO E LIMPEZA DA MALHA ESTÁTICA
        # =================================================================
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_social = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_SOCIAL_H3.parquet")
        
        df_infra = self._limpar_tabela_toda(df_infra)
        df_social = self._limpar_tabela_toda(df_social)
        df_universo_h3 = df_infra.join(df_social, on="H3_INDEX", how="full", coalesce=True).fill_null(0)

        # =================================================================
        # 2. CARREGAMENTO E LIMPEZA DOS CRIMES (TRUSTED)
        # =================================================================
        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
        
        lista_crimes = [df for f in crime_files if (df := self._ler_parquet_r2(f)) is not None]
        if not lista_crimes:
            raise ValueError("Nenhum dado de crime encontrado na camada prata.")
            
        df_crimes = pl.concat(lista_crimes, how="diagonal").filter(pl.col("H3_INDEX").is_not_null())
        df_crimes = self._limpar_tabela_toda(df_crimes)

        # =================================================================
        # 3. HORA, TURNOS E SANEAMENTO TEMPORAL
        # =================================================================
        df_crimes = df_crimes.with_columns(
            pl.col("HORAOCORRENCIA").cast(pl.Utf8).str.replace_all(r"\D", "").alias("_tmp_hora")
        ).with_columns(
            pl.when(pl.col("_tmp_hora").str.len_chars() == 3)
            .then(pl.lit("0") + pl.col("_tmp_hora"))
            .otherwise(pl.col("_tmp_hora"))
            .str.slice(0, 2)
            .cast(pl.Int8, strict=False)
            .alias("HORA_INT")
        )

        df_crimes = df_crimes.with_columns([
            pl.col("DATAOCORRENCIA").dt.year().alias("ANO_OCORRENCIA"),
            pl.col("DATAOCORRENCIA").dt.weekday().alias("FEAT_DIA_SEMANA"),
            
            # Recalcula e blinda a sazonalidade baseada na Hora Inteira validada
            pl.when((pl.col("HORA_INT") >= 18) & (pl.col("HORA_INT") <= 23)).then(pl.lit("NOITE"))
            .when((pl.col("HORA_INT") >= 12) & (pl.col("HORA_INT") < 18)).then(pl.lit("TARDE"))
            .when((pl.col("HORA_INT") >= 6) & (pl.col("HORA_INT") < 12)).then(pl.lit("MANHA"))
            .when((pl.col("HORA_INT") >= 0) & (pl.col("HORA_INT") < 6)).then(pl.lit("MADRUGADA"))
            .otherwise(pl.col("SAZON_PERIODO")).alias("SAZON_PERIODO")
        ])

        # Calendario (Final de Semana)
        df_crimes = df_crimes.with_columns([
            pl.when(pl.col("FEAT_DIA_SEMANA").is_in([6, 7])).then(pl.lit("FIM_DE_SEMANA"))
            .otherwise(pl.lit("DIA_UTIL")).alias("FEAT_TIPO_DIA")
        ])

        # =================================================================
        # 4. DOSIMETRIA PENAL E CONTEXTO DE VULNERABILIDADE
        # =================================================================
        df_gold = df_crimes.with_columns([
            pl.when(pl.col("RUBRICA").str.contains(r"VEICULO|CARGA")).then(pl.lit("MOTORISTA"))
            .when(pl.col("RUBRICA").str.contains(r"TRANSEUNTE|CELULAR|PESSOA")).then(pl.lit("PEDESTRE"))
            .otherwise(pl.lit("GERAL")).alias("FEAT_PERFIL_VITIMA"),

            pl.when(pl.col("RUBRICA").str.contains(r"ART.*121|LATROC")).then(pl.lit(10.0))
            .when(pl.col("RUBRICA").str.contains(r"ART.*157|ROUBO")).then(pl.lit(5.0))
            .when(pl.col("RUBRICA").str.contains(r"ART.*155|FURTO")).then(pl.lit(2.0))
            .otherwise(pl.lit(1.0)).alias("LABEL_PESO_RISCO")
        ]).with_columns(
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        )

        # =================================================================
        # 5. FEATURE STORE (HISTÓRICO) E FUSÃO DO UNIVERSO (JOIN FINAL)
        # =================================================================
        df_fs_hex = df_gold.group_by(["H3_INDEX", "ANO_OCORRENCIA"]).agg([
            pl.len().alias("FS_VOL_CRIMES_ANO_ANT"),
            pl.col("LABEL_PESO_RISCO").mean().alias("FS_RISCO_MEDIO_ANO_ANT")
        ]).with_columns((pl.col("ANO_OCORRENCIA") + 1).alias("ANO_JOIN"))

        df_final = df_gold.join(df_universo_h3, on="H3_INDEX", how="left") \
                          .join(df_fs_hex.drop("ANO_OCORRENCIA"), left_on=["H3_INDEX", "ANO_OCORRENCIA"], right_on=["H3_INDEX", "ANO_JOIN"], how="left")
        
        # Preenchimento de nulos em colunas numericas e temporais
        cols_fill = [c for c in df_final.columns if any(x in c for x in ["INFRA_", "CENSO_", "FS_"])]
        df_final = df_final.with_columns([pl.col(c).fill_null(0) for c in cols_fill])

        # Redução de Dimensionalidade (CNAEs -> Macros)
        cnae_macros = {
            "MACRO_FINANCEIRO": ["INFRA_DIV_64", "INFRA_DIV_65", "INFRA_DIV_66"],
            "MACRO_LAZER_NOTURNO": ["INFRA_DIV_56", "INFRA_DIV_90", "INFRA_DIV_93"],
            "MACRO_VAREJO": ["INFRA_DIV_45", "INFRA_DIV_47"]
        }
        for macro_name, div_list in cnae_macros.items():
            existentes = [c for c in div_list if c in df_final.columns]
            if existentes:
                df_final = df_final.with_columns(pl.sum_horizontal(existentes).alias(macro_name)) 
            else:
                df_final = df_final.with_columns(pl.lit(0).alias(macro_name))

        # Uma ultima limpeza de integridade nas colunas de texto geradas
        df_final = self._limpar_tabela_toda(df_final)

        # Drop de colunas auxiliares
        df_final = df_final.drop([c for c in df_final.columns if c.startswith("INFRA_DIV_")] + ["ANO_OCORRENCIA", "HORA_INT", "_tmp_hora"])

        # =================================================================
        # 6. EXPORTACAO DA ABT (ANALYTICAL BASE TABLE) E LOGS
        # =================================================================
        buf = io.BytesIO()
        df_final.write_parquet(buf, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/safedriver_abt_treino.parquet", Body=buf.getvalue())

        counts = df_final.group_by("SAZON_PERIODO").len().to_dict(as_series=False)
        sazon_dict = {k: v for k, v in zip(counts['SAZON_PERIODO'], counts['len'])}
        
        duracao = round(time.time() - inicio_timer, 2)
        report = (
            f"🏆 Relatorio Ouro - Feature Store Finalizada\n"
            f"============================================\n"
            f"- Registros da ABT: {df_final.height}\n"
            f"- Features Estruturadas: {len(df_final.columns)}\n"
            f"- Tempo de Processamento: {duracao}s\n"
            f"--------------------------------------------\n"
            f"DISTRIBUICAO TEMPORAL HOMOGENEIZADA:\n"
            f"  > MANHA: {sazon_dict.get('MANHA', 0)}\n"
            f"  > TARDE: {sazon_dict.get('TARDE', 0)}\n"
            f"  > NOITE: {sazon_dict.get('NOITE', 0)}\n"
            f"  > MADRUGADA: {sazon_dict.get('MADRUGADA', 0)}\n"
            f"  > INCERTO: {sazon_dict.get('INCERTO', 0)}\n"
            f"============================================"
        )
        print(report)
        self._notificar_discord(f"```text\n{report}\n```")

if __name__ == "__main__":
    ArquitetoSafeDriverOuro().construir_abt_final()
