import os
import io
import boto3
import polars as pl
import time
import requests
import json
import numpy as np
from botocore.config import Config
from datetime import datetime

class ArquitetoSafeDriverOuro:
    """
    Engine de Construção da ABT SafeDriver Autobot.
    Reforma: Engenharia Temporal Completa, Contexto Crítico Nativo e Saneamento Espacial.
    """
    def __init__(self):
        self.projeto = "SafeDriver Autobot"
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
        # Atualizado para pl.String (substitui pl.Utf8 nas versões mais recentes do Polars)
        cols_texto = [c for c, t in zip(df.columns, df.dtypes) if t == pl.String or t == pl.Utf8]
        if not cols_texto: return df
        return df.with_columns([
            pl.col(c).str.to_uppercase().str.strip_chars()
            .str.replace_all(r"[ÁÀÂÃÄ]", "A").str.replace_all(r"[ÉÈÊË]", "E")
            .str.replace_all(r"[ÍÌÎÏ]", "I").str.replace_all(r"[ÓÒÔÕÖ]", "O")
            .str.replace_all(r"[ÚÙÛÜ]", "U").str.replace_all(r"[Ç]", "C")
            .str.replace_all(r"[^A-Z0-9\s_]", " ").str.replace_all(r"\s+", " ")
            .fill_null("DESCONHECIDO").alias(c)
            for c in cols_texto
        ])

    def _ler_parquet_r2(self, key):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            return pl.read_parquet(io.BytesIO(obj['Body'].read()))
        except Exception: 
            return None

    def construir_abt_final(self):
        inicio_timer = time.time()
        print(f"[INFO] Iniciando construcao da Camada Ouro: {self.projeto}")

        # =================================================================
        # 1. CARREGAMENTO (MALHA GEOGRAFICA)
        # =================================================================
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_social = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_SOCIAL_H3.parquet")
        
        df_universo_h3 = df_infra.join(df_social, on="H3_INDEX", how="full", coalesce=True).fill_null(0)

        if "CIDADE" in df_social.columns and "BAIRRO" in df_social.columns:
            df_dim_bairro = df_social.select(["H3_INDEX", "CIDADE", "BAIRRO"]).unique("H3_INDEX")
        else:
            df_dim_bairro = pl.DataFrame({"H3_INDEX": [], "CIDADE": [], "BAIRRO": []})

        # =================================================================
        # 2. CONSOLIDAÇÃO DE CRIMES E APLICAÇÃO GEOGRÁFICA
        # =================================================================
        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
        lista_crimes = [df for f in crime_files if (df := self._ler_parquet_r2(f)) is not None]
        df_crimes = pl.concat(lista_crimes, how="diagonal").filter(pl.col("H3_INDEX").is_not_null())
        df_crimes = self._limpar_tabela_toda(df_crimes)

        if not df_dim_bairro.is_empty():
            df_crimes = df_crimes.drop(["CIDADE", "BAIRRO"], strict=False).join(df_dim_bairro, on="H3_INDEX", how="left")

        # =================================================================
        # 3. ENGENHARIA TEMPORAL COMPLETA (Reforma)
        # =================================================================
        # Extração limpa de hora, dia da semana e tipos de dia para o CatBoost
        df_crimes = df_crimes.with_columns(
            pl.col("HORAOCORRENCIA").cast(pl.String).str.replace_all(r"\D", "").alias("_tmp_hora")
        ).with_columns(
            pl.when(pl.col("_tmp_hora").str.len_chars() == 3).then(pl.lit("0") + pl.col("_tmp_hora"))
            .otherwise(pl.col("_tmp_hora")).str.slice(0, 2).cast(pl.Int8, strict=False).alias("HORA_INT")
        ).with_columns([
            pl.col("DATAOCORRENCIA").dt.year().alias("ANO_OCORRENCIA"),
            pl.col("DATAOCORRENCIA").dt.month().alias("FEAT_MES"),
            pl.col("DATAOCORRENCIA").dt.weekday().alias("FEAT_DIA_SEMANA"), # 1 (Seg) a 7 (Dom)
            (pl.col("DATAOCORRENCIA").dt.year() * 12 + pl.col("DATAOCORRENCIA").dt.month()).alias("MES_ABSOLUTO"),
            pl.when((pl.col("HORA_INT") >= 18) & (pl.col("HORA_INT") <= 23)).then(pl.lit("NOITE"))
            .when((pl.col("HORA_INT") >= 12) & (pl.col("HORA_INT") < 18)).then(pl.lit("TARDE"))
            .when((pl.col("HORA_INT") >= 6) & (pl.col("HORA_INT") < 12)).then(pl.lit("MANHA"))
            .otherwise(pl.lit("MADRUGADA")).alias("SAZON_PERIODO")
        ])

        # Flags Booleanas/Categóricas exigidas pelo modelo preditivo
        df_crimes = df_crimes.with_columns([
            pl.when(pl.col("FEAT_DIA_SEMANA") >= 6).then(pl.lit("SIM")).otherwise(pl.lit("NAO")).alias("FEAT_IS_FIM_DE_SEMANA"),
            pl.when(pl.col("FEAT_DIA_SEMANA") >= 6).then(pl.lit("FIM_DE_SEMANA")).otherwise(pl.lit("DIA_UTIL")).alias("FEAT_TIPO_DIA")
        ])

        # =================================================================
        # 4. DOSIMETRIA PENAL E CONTEXTO CRÍTICO
        # =================================================================
        df_gold = df_crimes.with_columns([
            pl.when(pl.col("RUBRICA").str.contains(r"VEICULO|CARGA")).then(pl.lit("MOTORISTA"))
            .when(pl.col("RUBRICA").str.contains(r"TRANSEUNTE|CELULAR|PESSOA")).then(pl.lit("PEDESTRE"))
            .otherwise(pl.lit("GERAL")).alias("FEAT_PERFIL_VITIMA"),
            
            pl.when(pl.col("RUBRICA").str.contains(r"ART.*121|LATROC|HOMICIDIO")).then(pl.lit(10.0))
            .when(pl.col("RUBRICA").str.contains(r"ART.*157|ROUBO")).then(pl.lit(5.0))
            .when(pl.col("RUBRICA").str.contains(r"ART.*155|FURTO")).then(pl.lit(2.0))
            .otherwise(pl.lit(1.0)).alias("LABEL_PESO_RISCO")
        ])

        # Feature Cross nativa na base Ouro
        df_gold = df_gold.with_columns(
            pl.concat_str([pl.col("SAZON_PERIODO"), pl.lit("_"), pl.col("FEAT_PERFIL_VITIMA")]).alias("FEAT_CONTEXTO_CRITICO")
        )

        # =================================================================
        # 5. FEATURE STORE (MOMENTUM E MASSA CRIMINAL)
        # =================================================================
        df_fs_ano = df_gold.group_by(["H3_INDEX", "ANO_OCORRENCIA"]).agg([
            pl.col("LABEL_PESO_RISCO").mean().alias("FS_RISCO_MEDIO_ANO_ANT"),
            pl.len().alias("FS_VOL_CRIMES_ANO_ANT") 
        ]).with_columns((pl.col("ANO_OCORRENCIA") + 1).alias("ANO_JOIN"))

        df_fs_mes = df_gold.group_by(["H3_INDEX", "MES_ABSOLUTO"]).agg([
            pl.col("LABEL_PESO_RISCO").mean().alias("FS_RISCO_MEDIO_MES_ANT")
        ]).with_columns((pl.col("MES_ABSOLUTO") + 1).alias("MES_JOIN"))

        df_final = df_gold.join(df_universo_h3, on="H3_INDEX", how="left") \
                          .join(df_fs_ano.drop("ANO_OCORRENCIA"), left_on=["H3_INDEX", "ANO_OCORRENCIA"], right_on=["H3_INDEX", "ANO_JOIN"], how="left") \
                          .join(df_fs_mes.drop("MES_ABSOLUTO"), left_on=["H3_INDEX", "MES_ABSOLUTO"], right_on=["H3_INDEX", "MES_JOIN"], how="left")
        
        # Saneamento robusto de nulos para garantir a matemática no Looker e no CatBoost
        if "CIDADE" in df_final.columns and "BAIRRO" in df_final.columns:
            df_final = df_final.with_columns([
                pl.col("CIDADE").fill_null("DESCONHECIDO"),
                pl.col("BAIRRO").fill_null("DESCONHECIDO")
            ])

        df_final = df_final.with_columns([
            pl.col("FS_RISCO_MEDIO_ANO_ANT").fill_null(0.1),
            pl.col("FS_RISCO_MEDIO_MES_ANT").fill_null(0.1),
            pl.col("FS_VOL_CRIMES_ANO_ANT").fill_null(0) 
        ]).with_columns(
            (pl.col("FS_RISCO_MEDIO_MES_ANT") / pl.col("FS_RISCO_MEDIO_ANO_ANT")).alias("FS_MOMENTUM_RISCO")
        )

        # =================================================================
        # 6. EXPORTACAO (Compressao maxima aplicada)
        # =================================================================
        key_final = f"{self.ouro_dir}/safedriver_abt_treino.parquet"
        buf = io.BytesIO()
        df_final.write_parquet(buf, compression="zstd", compression_level=22) 
        self.s3.put_object(Bucket=self.bucket, Key=key_final, Body=buf.getvalue())

        # --- 7. SISTEMA DE RETORNO (AUDITORIA TECNICA) ---
        duracao = round(time.time() - inicio_timer, 2)
        top_rubricas = df_final["RUBRICA"].value_counts().sort("count", descending=True).head(10)
        mapeamento_rubricas = df_final.group_by("RUBRICA").agg(pl.col("LABEL_PESO_RISCO").first()).join(top_rubricas, on="RUBRICA").sort("count", descending=True)

        report_lines = [
            f"RELATORIO DE PROCESSAMENTO: CAMADA OURO - {self.projeto.upper()}",
            "--------------------------------------------------",
            f"Volume Total Processado: {df_final.height} registros",
            f"Tempo de Execucao: {duracao} segundos",
            "",
            "ANALISE DE DISTRIBUICAO (TOP 10 RUBRICAS):"
        ]
        
        for row in mapeamento_rubricas.iter_rows():
            rubrica_nome = row[0][:40].ljust(45)
            peso_val = f"Peso: {row[1]:.1f}".ljust(12)
            qtd_val = f"Qtd: {row[2]}"
            report_lines.append(f"  - {rubrica_nome} | {peso_val} | {qtd_val}")
        
        teto_encontrado = df_final["LABEL_PESO_RISCO"].max()
        report_lines.extend([
            "",
            f"Limite Superior de Dosimetria Aplicado: {teto_encontrado:.1f}"
        ])
        
        if teto_encontrado < 10.0:
            report_lines.append("ALERTA DE INTEGRIDADE: Ausencia de registros com classificacao de severidade maxima (Peso 10.0) na presente amostragem.")

        report = "\n".join(report_lines)
        print(report)
        self._notificar_discord(f"```text\n{report}\n```")
        print("[INFO] Pipeline executado com sucesso. Arquivo exportado.")

if __name__ == "__main__":
    ArquitetoSafeDriverOuro().construir_abt_final()
