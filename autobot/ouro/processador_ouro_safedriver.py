import os
import boto3
import polars as pl
import io
import time
import requests
import json
from botocore.config import Config
from datetime import datetime

class ArquitetoSafeDriverOuro:
    """
    Componente responsavel pela consolidacao da Analytical Base Table (ABT).
    Esta versao esta equalizada com o Ingestor Prata, aproveitando as colunas
    ja normalizadas e aplicando uma rede de seguranca para a Sazonalidade Temporal.
    """
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
            "fase": "ABT Master Equalizada",
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
        print("Iniciando Consolidacao Ouro (Sincronizada com a Prata)...", flush=True)
        
        # 1. CARREGAMENTO DAS BASES PROVENIENTES DA PRATA
        df_infra = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_INFRA_AGREGADA.parquet")
        df_social = self._ler_parquet_r2(f"{self.prata_malha}/PRATA_MALHA_SOCIAL_H3.parquet")
        
        # Uniao da Malha Estatica (H3 como chave primaria)
        df_universo_h3 = df_infra.join(df_social, on="H3_INDEX", how="full", coalesce=True).fill_null(0)

        # 2. CARREGAMENTO DOS CRIMES (Ssp_trusted_*.parquet)
        paginator = self.s3.get_paginator('list_objects_v2')
        crime_files = [
            obj['Key'] for p in paginator.paginate(Bucket=self.bucket, Prefix=f"{self.prata_crimes}/")
            for obj in p.get('Contents', []) if obj['Key'].endswith('.parquet')
        ]
        
        lista_crimes = [df for f in crime_files if (df := self._ler_parquet_r2(f)) is not None]
        df_crimes = pl.concat(lista_crimes, how="diagonal").filter(pl.col("H3_INDEX").is_not_null())

        # 3. REDE DE SEGURANÇA: RECALCULO DE SAZONALIDADE
        # Se a Prata falhou na extração da hora por tipo de dado, a Ouro corrige aqui.
        print("Aplicando rede de seguranca na Sazonalidade Temporal...", flush=True)
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
            
            pl.when((pl.col("HORA_INT") >= 18) & (pl.col("HORA_INT") <= 23)).then(pl.lit("NOITE"))
            .when((pl.col("HORA_INT") >= 12) & (pl.col("HORA_INT") < 18)).then(pl.lit("TARDE"))
            .when((pl.col("HORA_INT") >= 6) & (pl.col("HORA_INT") < 12)).then(pl.lit("MANHA"))
            .when((pl.col("HORA_INT") >= 0) & (pl.col("HORA_INT") < 6)).then(pl.lit("MADRUGADA"))
            # Fallback: Se o calculo falhar mas a Prata ja tinha a coluna preenchida, usa a da Prata
            .otherwise(pl.col("SAZON_PERIODO")).alias("SAZON_PERIODO")
        ])

        # 4. ENGENHARIA DE ATRIBUTOS (UNIFICAÇÃO DE CALENDÁRIO)
        # Nota: A Prata ja traz a DATAOCORRENCIA tipada.
        df_crimes = df_crimes.with_columns([
            pl.when(pl.col("FEAT_DIA_SEMANA").is_in([6, 7])).then(pl.lit("FIM_DE_SEMANA"))
            .otherwise(pl.lit("DIA_UTIL")).alias("FEAT_TIPO_DIA")
        ])

        # 5. DOSIMETRIA PENAL (Baseada na RUBRICA ja normalizada pela Prata)
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

        # 6. CONSTRUÇÃO DA FEATURE STORE (HISTÓRICO)
        df_fs_hex = df_gold.group_by(["H3_INDEX", "ANO_OCORRENCIA"]).agg([
            pl.len().alias("FS_VOL_CRIMES_ANO_ANT"),
            pl.col("LABEL_PESO_RISCO").mean().alias("FS_RISCO_MEDIO_ANO_ANT")
        ]).with_columns((pl.col("ANO_OCORRENCIA") + 1).alias("ANO_JOIN"))

        # 7. JOIN FINAL (FUSAO MULTIDIMENSIONAL)
        df_final = df_gold.join(df_universo_h3, on="H3_INDEX", how="left") \
                          .join(df_fs_hex.drop("ANO_OCORRENCIA"), left_on=["H3_INDEX", "ANO_OCORRENCIA"], right_on=["H3_INDEX", "ANO_JOIN"], how="left")
        
        # Limpeza de Nulos e Consolidacao de Macros CNAE
        cols_fill = [c for c in df_final.columns if any(x in c for x in ["INFRA_", "CENSO_", "FS_"])]
        df_final = df_final.with_columns([pl.col(c).fill_null(0) for c in cols_fill])

        cnae_macros = {
            "MACRO_FINANCEIRO": ["INFRA_DIV_64", "INFRA_DIV_65", "INFRA_DIV_66"],
            "MACRO_LAZER_NOTURNO": ["INFRA_DIV_56", "INFRA_DIV_90", "INFRA_DIV_93"],
            "MACRO_VAREJO": ["INFRA_DIV_45", "INFRA_DIV_47"]
        }
        for macro_name, div_list in cnae_macros.items():
            existentes = [c for c in div_list if c in df_final.columns]
            df_final = df_final.with_columns(pl.sum_horizontal(existentes).alias(macro_name)) if existentes else df_final.with_columns(pl.lit(0).alias(macro_name))

        # Drop de colunas auxiliares
        df_final = df_final.drop([c for c in df_final.columns if c.startswith("INFRA_DIV_")] + ["ANO_OCORRENCIA", "HORA_INT", "_tmp_hora"])

        # 8. EXPORTAÇÃO E AUDITORIA
        buf_abt = io.BytesIO()
        df_final.write_parquet(buf_abt, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.ouro_dir}/safedriver_abt_treino.parquet", Body=buf_abt.getvalue())

        # Auditoria para o Log do Discord/GitHub
        counts = df_final.group_by("SAZON_PERIODO").len().to_dict(as_series=False)
        sazon_dict = {k: v for k, v in zip(counts['SAZON_PERIODO'], counts['len'])}
        
        duracao = round(time.time() - inicio_timer, 2)
        report = (
            f"Relatorio de Consolidacao - Camada Ouro Finalizada\n"
            f"================================================\n"
            f"- Registros Processados    : {df_final.height}\n"
            f"- Features Estruturadas    : {len(df_final.columns)}\n"
            f"- Tempo de Execucao (s)    : {duracao}s\n"
            f"------------------------------------------------\n"
            f"AUDITORIA DE PERIODOS (SAZONALIDADE):\n"
            f"  > MANHA      : {sazon_dict.get('MANHA', 0)}\n"
            f"  > TARDE      : {sazon_dict.get('TARDE', 0)}\n"
            f"  > NOITE      : {sazon_dict.get('NOITE', 0)}\n"
            f"  > MADRUGADA  : {sazon_dict.get('MADRUGADA', 0)}\n"
            f"  > INCERTO    : {sazon_dict.get('INCERTO', 0)}\n"
            f"================================================"
        )
        print(report)
        self._notificar_discord(f"```text\n{report}\n```")

if __name__ == "__main__":
    ArquitetoSafeDriverOuro().construir_abt_final()
