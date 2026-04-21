import os
import boto3
import requests
import logging
import time
import sys
import io
import hashlib
import re
import unicodedata
import botocore.exceptions
import polars as pl
import h3
import fastexcel
from botocore.config import Config
from datetime import datetime

# Configuração de Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ConfiguracaoIngestao:
    NOME_BUCKET = os.getenv("R2_BUCKET_NAME")
    ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
    ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
    SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
    PEPPER = os.getenv("LGPD_PEPPER", "safedriver_secret_2026")

    URL_BASE_SSP = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
    RESOLUCAO_H3 = 9
    
    MAPA_COLUNAS = {
        "NUM_BO": [r"NUM.*BO", r"N.MERO.*BO", r"BO_NUMERO"],
        "MUNICIPIO": [r"MUNIC.PIO", r"CIDADE", r"NM_MUN", r"NOME_MUN"],
        "BAIRRO": [r"BAIRRO", r"NM_BAIRRO"],
        "LOGRADOURO": [r"LOGRADOURO", r"RUA", r"DESCR_LOG", r"NM_LOG", r"ENDERECO"],
        "DATAOCORRENCIA": [r"DATA.*OCORR", r"DT_OCORR", r"DATA_OCORRENCIA"],
        "HORAOCORRENCIA": [r"HORA.*OCORR", r"HR_OCORR", r"HORA_OCORRENCIA"],
        "RUBRICA": [r"RUBRICA", r"NATUREZA", r"DESCR_RUBRICA"],
        "LATITUDE": [r"LATITUDE", r"LAT.*GEO", r"^LAT$", r"COORDENADA_X", r"LATITUD"],
        "LONGITUDE": [r"LONGITUDE", r"LON.*GEO", r"^LON$", r"COORDENADA_Y", r"LONGITUD"],
        "DESCR_TIPOLOCAL": [r"TIPOLOCAL", r"LOCAL_OCORR", r"DESCR_TIPOLOCAL", r"LOCAL"]
    }

class IngestorSafeDriver:
    def __init__(self):
        self.config = ConfiguracaoIngestao()
        self.s3 = self._inicializar_s3()
        self.ano_atual = datetime.now().year

    def _inicializar_s3(self):
        # FIX: Limpeza idêntica à malha para não duplicar o bucket no R2
        endpoint = self.config.ENDPOINT_URL.strip().rstrip('/')
        if endpoint.endswith(f"/{self.config.NOME_BUCKET}"):
            endpoint = endpoint[: -len(f"/{self.config.NOME_BUCKET}")]
            
        return boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=self.config.ACCESS_KEY.strip(),
            aws_secret_access_key=self.config.SECRET_KEY.strip(),
            config=Config(signature_version='s3v4')
        )

    def _arquivo_existe(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.config.NOME_BUCKET, Key=key)
            return True
        except: return False

    def _normalizar_texto(self, valor):
        """Remove acentos, caracteres especiais e deixa em maiúsculo."""
        if valor is None or str(valor).upper() in ["NULL", "NAN", ".", "", "NONE"]: 
            return "NAO INFORMADO"
        # Normalização NFKD para separar acentos das letras
        texto = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        # Remove tudo que não for letra, número ou espaço e limpa as pontas
        return re.sub(r'[^a-zA-Z0-9\s]', '', texto).upper().strip()

    def _resolver_mapeamento(self, lista_colunas):
        mapeamento = {}
        for i, nome_col in enumerate(lista_colunas):
            if nome_col is None: continue
            col_limpa = "".join(c for c in str(nome_col).upper() if c.isalnum() or c == '_')
            for alvo, padroes in self.config.MAPA_COLUNAS.items():
                if alvo in mapeamento.values(): continue
                for p in padroes:
                    if re.search(p, col_limpa, re.IGNORECASE):
                        mapeamento[i] = alvo
                        break
        return mapeamento

    def _limpar_e_tipar(self, df: pl.DataFrame) -> pl.DataFrame:
        # FIX: LOGRADOURO incluído na normalização de texto para que o Join com a Ouro funcione perfeitamente
        colunas_texto = [c for c in df.columns if c not in ["LATITUDE", "LONGITUDE", "NUM_BO"]]
        
        for col in colunas_texto:
            df = df.with_columns(
                pl.col(col).map_elements(self._normalizar_texto, return_dtype=pl.Utf8).alias(col)
            )

        # 2. LAT/LON para Float
        df = df.with_columns([
            pl.col("LATITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False),
            pl.col("LONGITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False)
        ])
        return df.filter((pl.col("LATITUDE").is_not_null()) & (pl.col("LATITUDE") < -10))

    def extrair_bronze(self, ano: int):
        path = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        if ano < self.ano_atual and self._arquivo_existe(path): return
        url = self.config.URL_BASE_SSP.format(ano=ano)
        try:
            res = requests.get(url, timeout=600)
            if res.status_code == 200:
                self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path, Body=res.content)
                logger.info(f"✅ Bronze {ano} salva.")
        except Exception as e: logger.error(f"❌ Erro Download {ano}: {e}")

    def processar_prata(self, ano: int):
        path_b = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_p = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"

        if self._arquivo_existe(path_p):
            logger.info(f"⏭️ Prata {ano} já pronta.")
            return

        try:
            logger.info(f"🥈 [Ano {ano}] Iniciando Normalização Universal...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=path_b)
            excel_bytes = obj['Body'].read()
            
            excel_reader = fastexcel.read_excel(excel_bytes)
            abas = excel_reader.sheet_names
            list_dfs = []

            for nome_aba in abas:
                if any(x in nome_aba.upper() for x in ["CAPA", "DICIONARIO", "LEGENDA", "CAMPOS", "SPDADOS"]):
                    continue

                try:
                    df_raw = pl.read_excel(excel_bytes, sheet_name=nome_aba, engine="calamine")
                    indices_map = self._resolver_mapeamento(df_raw.columns)
                    
                    if "LATITUDE" not in indices_map.values():
                        found = False
                        for row_idx in range(min(30, df_raw.height)):
                            row_data = df_raw.row(row_idx)
                            indices_map = self._resolver_mapeamento(row_data)
                            if "LATITUDE" in indices_map.values() and "LONGITUDE" in indices_map.values():
                                df_real = pl.read_excel(excel_bytes, sheet_name=nome_aba, engine="calamine", 
                                                        read_options={"skip_rows": row_idx + 1})
                                indices_map = self._resolver_mapeamento(df_real.columns)
                                df_mes = df_real.select([df_real.columns[i] for i in indices_map.keys()])
                                df_mes.columns = [indices_map[i] for i in indices_map.keys()]
                                found = True
                                break
                        if not found: continue
                    else:
                        df_mes = df_raw.select([df_raw.columns[i] for i in indices_map.keys()])
                        df_mes.columns = [indices_map[i] for i in indices_map.keys()]

                    df_mes = df_mes.with_columns(pl.all().cast(pl.Utf8))
                    list_dfs.append(df_mes)

                except Exception as e:
                    logger.warning(f"   ⚠️ Falha na aba {nome_aba}: {e}")

            if not list_dfs: raise ValueError(f"Dados não localizados em {ano}.")

            df_final = pl.concat(list_dfs, how="diagonal")
            
            # APLICA NORMALIZAÇÃO (MAIÚSCULO E SEM ACENTO PARA TUDO, INCLUINDO LOGRADOURO)
            df_final = self._limpar_e_tipar(df_final)
            
            # FIX: Apenas o BO recebe hash para anonimização (LGPD). A rua permanece em texto claro para geocoding!
            pepper = self.config.PEPPER
            df_final = df_final.with_columns([
                pl.col("NUM_BO").map_elements(lambda v: hashlib.sha256(f"{str(v).upper()}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8)
            ])

            # H3
            res_h3 = self.config.RESOLUCAO_H3
            df_final = df_final.with_columns(
                pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                    lambda x: h3.latlng_to_cell(x["LATITUDE"], x["LONGITUDE"], res_h3),
                    return_dtype=pl.Utf8
                ).alias("H3_INDEX")
            )

            buf = io.BytesIO()
            df_final.write_parquet(buf, compression="zstd")
            self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_p, Body=buf.getvalue())
            logger.info(f"✨ Prata {ano} normalizada com SUCESSO.")

        except Exception as e:
            logger.error(f"💥 Erro fatal em {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    for ano in range(2022, ingestor.ano_atual + 1):
        if modo in ["bronze", "tudo"]: ingestor.extrair_bronze(ano)
        if modo in ["prata", "tudo"]: ingestor.processar_prata(ano)
