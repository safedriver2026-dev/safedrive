import os
import boto3
import requests
import time
import sys
import io
import hashlib
import re
import unicodedata
import polars as pl
import h3
import fastexcel
import json
from botocore.config import Config
from datetime import datetime

class IngestorCrimesPrata:
    def __init__(self):
        self.bucket = os.getenv("R2_BUCKET_NAME")
        self.pepper = os.getenv("LGPD_PEPPER", "safedriver_secret_2026")
        self.webhook_discord = os.getenv("DISCORD_SUCESSO")
        self.resolucao_h3 = 9
        
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        if endpoint.endswith(f"/{self.bucket}"):
            endpoint = endpoint[: -len(f"/{self.bucket}")]
            
        self.s3 = boto3.client(
            's3', endpoint_url=endpoint, 
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY").strip(), 
            config=Config(signature_version='s3v4')
        )

        self.mapa_colunas = {
            "NUM_BO": [r"NUM.*BO", r"BO_NUMERO"],
            "MUNICIPIO": [r"MUNIC.PIO", r"CIDADE", r"NM_MUN", r"NOME_MUNICIPIO$"],
            "BAIRRO": [r"BAIRRO", r"NM_BAIRRO"],
            "LOGRADOURO": [r"LOGRADOURO", r"RUA", r"DESCR_LOG", r"ENDERECO"],
            "DATAOCORRENCIA": [r"DATA_OCORRENCIA_BO", r"DT_OCORR", r"DATA_OCORRENCIA"],
            "DATAREGISTRO": [r"DATA_REGISTRO", r"DT_REGISTRO"],
            "HORAOCORRENCIA": [r"HORA_OCORRENCIA_BO", r"HR_OCORR", r"HORA_OCORRENCIA"],
            "RUBRICA": [r"RUBRICA", r"NATUREZA", r"DESCR_RUBRICA"],
            "LATITUDE": [r"LATITUDE", r"COORDENADA_X", r"LATITUD"],
            "LONGITUDE": [r"LONGITUDE", r"COORDENADA_Y", r"LONGITUD"],
            "PERIODO_NATIVO": [r"DESC_PERIODO", r"PERIODO", r"DS_PERIODO"]
        }
        self.df_lookup_vias = None
        self.audit_stats = []

    def _limpar_tabela_toda(self, df):
        """Limpeza pesada em todas as colunas de texto da tabela."""
        cols_texto = [c for c, t in zip(df.columns, df.dtypes) if t == pl.Utf8]
        return df.with_columns([
            pl.col(c)
            .str.to_uppercase()
            .str.strip_chars()
            .map_elements(lambda x: "".join(c for c in unicodedata.normalize('NFKD', str(x)) if unicodedata.category(c) != 'Mn'), return_dtype=pl.Utf8)
            .str.replace_all(r"[^A-Z0-9\s_]", " ")
            .str.replace_all(r"\s+", " ")
            .fill_null("DESCONHECIDO")
            .alias(c)
            for c in cols_texto
        ])

    def _carregar_malha(self):
        print("Lendo malha de referencia para resgate de GPS...")
        path = "datalake/prata/malha_trusted/PRATA_MALHA_GEOGRAFICA_VIAS.parquet"
        obj = self.s3.get_object(Bucket=self.bucket, Key=path)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read())).explode("BAIRROS").unnest("BAIRROS").explode("LOGRADOUROS").unnest("LOGRADOUROS")
        self.df_lookup_vias = df.with_columns([
            pl.col("H3_LIST").list.first().alias("H3_INDEX"),
            pl.col("CIDADE").alias("MUN_L"), pl.col("BAIRRO").alias("BAI_L"), pl.col("RUA").alias("LOG_L")
        ]).select(["MUN_L", "BAI_L", "LOG_L", "H3_INDEX"]).unique()

    def processar_ano(self, ano):
        print(f"Processando crimes de {ano}...")
        path_in = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_out = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"
        
        if self.df_lookup_vias is None: self._carregar_malha()
        
        obj = self.s3.get_object(Bucket=self.bucket, Key=path_in)
        excel_bytes = obj['Body'].read()
        reader = fastexcel.read_excel(excel_bytes)
        abas = [n for n in reader.sheet_names if not any(x in n.upper() for x in ["CAPA", "DICIONARIO", "SPDADOS"])]
        
        dfs_lista = []
        for aba in abas:
            df = pl.read_excel(excel_bytes, sheet_name=aba, engine="calamine").with_columns(pl.all().cast(pl.Utf8))
            
            # Mapeia colunas dinamicamente
            mapeado = {}
            for i, col in enumerate(df.columns):
                col_up = col.upper()
                for alvo, regexes in self.mapa_colunas.items():
                    if any(re.search(r, col_up) for r in regexes) and alvo not in mapeado.values():
                        mapeado[col] = alvo
            
            if mapeado:
                df = df.select(list(mapeado.keys())).rename(mapeado)
                dfs_lista.append(df)

        if not dfs_lista: return
        
        df_full = pl.concat(dfs_lista, how="diagonal")
        
        # --- NORMALIZAÇÃO TOTAL ---
        df_full = self._limpar_tabela_toda(df_full)

        # Trata Datas
        df_full = df_full.with_columns(pl.col("DATAOCORRENCIA").str.to_date(strict=False))

        # H3 por GPS original
        df_full = df_full.with_columns([
            pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).alias("_lat"),
            pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).alias("_lon")
        ]).with_columns(
            pl.struct(["_lat", "_lon"]).map_elements(
                lambda x: h3.latlng_to_cell(x["_lat"], x["_lon"], self.resolucao_h3) if x["_lat"] and x["_lat"] < -10 else None,
                return_dtype=pl.Utf8
            ).alias("H3_INDEX")
        )

        # Resgate pela Malha (Join)
        df_full = df_full.join(self.df_lookup_vias, left_on=["MUNICIPIO", "BAIRRO", "LOGRADOURO"], right_on=["MUN_L", "BAI_L", "LOG_L"], how="left")
        df_full = df_full.with_columns(pl.col("H3_INDEX").fill_null(pl.col("H3_INDEX_right"))).drop(["H3_INDEX_right", "_lat", "_lon"])

        # Período (Sazonalidade)
        df_full = df_full.with_columns([
            pl.col("HORAOCORRENCIA").str.extract(r"(\d{1,2})", 1).cast(pl.Int32, strict=False).alias("_hr"),
            pl.col("PERIODO_NATIVO").alias("_per")
        ]).with_columns(
            pl.when(pl.col("_hr").is_between(0, 5) | pl.col("_per").str.contains("MADRUGADA")).then(pl.lit("MADRUGADA"))
            .when(pl.col("_hr").is_between(6, 11) | pl.col("_per").str.contains("MANH")).then(pl.lit("MANHA"))
            .when(pl.col("_hr").is_between(12, 17) | pl.col("_per").str.contains("TARDE")).then(pl.lit("TARDE"))
            .when(pl.col("_hr").is_between(18, 23) | pl.col("_per").str.contains("NOITE")).then(pl.lit("NOITE"))
            .otherwise(pl.lit("INCERTO")).alias("SAZON_PERIODO")
        ).drop(["_hr", "_per"])

        # Anonimização LGPD
        df_full = df_full.with_columns(
            pl.col("NUM_BO").map_elements(lambda v: hashlib.sha256(f"{str(v)}{self.pepper}".encode()).hexdigest(), return_dtype=pl.Utf8)
        )

        # Salva no R2
        buf = io.BytesIO()
        df_full.write_parquet(buf, compression="zstd")
        self.s3.put_object(Bucket=self.bucket, Key=path_out, Body=buf.getvalue())
        print(f"Arquivo {path_out} salvo.")

if __name__ == "__main__":
    ingestor = IngestorCrimesPrata()
    ano_atual = datetime.now().year
    for ano in range(2022, ano_atual + 1):
        try: ingestor.processar_ano(ano)
        except Exception as e: print(f"Erro no ano {ano}: {e}")
