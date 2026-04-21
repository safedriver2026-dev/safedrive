import os
import boto3
import requests
import logging
import time
import sys
import io
import hashlib
import re
import botocore.exceptions
import polars as pl
import h3
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
    
    # Mapeamento Ultra Flexível (SSP adora mudar esses nomes)
    MAPA_COLUNAS = {
        "NUM_BO": [r"NUM.*BO", r"N.MERO.*BO"],
        "MUNICIPIO": [r"MUNIC.PIO", r"CIDADE"],
        "BAIRRO": [r"BAIRRO"],
        "LOGRADOURO": [r"LOGRADOURO", r"RUA", r"DESCR_LOG"],
        "DATAOCORRENCIA": [r"DATA.*OCORR", r"DT_OCORR"],
        "HORAOCORRENCIA": [r"HORA.*OCORR", r"HR_OCORR"],
        "RUBRICA": [r"RUBRICA", r"NATUREZA", r"DESCR_RUBRICA"],
        "LATITUDE": [r"LATITUDE", r"LAT.*GEO", r"^LAT$"],
        "LONGITUDE": [r"LONGITUDE", r"LON.*GEO", r"^LON$", r"^LONG$"],
        "DESCR_TIPOLOCAL": [r"TIPOLOCAL", r"LOCAL_OCORR"]
    }

class IngestorSafeDriver:
    def __init__(self):
        self.config = ConfiguracaoIngestao()
        self.s3 = self._inicializar_s3()
        self.ano_atual = datetime.now().year

    def _inicializar_s3(self):
        endpoint = self.config.ENDPOINT_URL.strip().rstrip('/')
        return boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=self.config.ACCESS_KEY.strip(),
            aws_secret_access_key=self.config.SECRET_KEY.strip(),
            config=Config(signature_version='s3v4', retries={'max_attempts': 5})
        )

    def _arquivo_existe(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.config.NOME_BUCKET, Key=key)
            return True
        except: return False

    def _resolver_colunas_fuzzy(self, colunas_reais):
        """Usa Regex para mapear colunas mesmo com nomes zoados pela SSP."""
        mapeamento_final = {}
        for alvo, padroes in self.config.MAPA_COLUNAS.items():
            for col in colunas_reais:
                for p in padroes:
                    if re.search(p, col, re.IGNORECASE):
                        mapeamento_final[col] = alvo
                        break
                if col in mapeamento_final: break
        return mapeamento_final

    def _limpar_e_tipar(self, df: pl.DataFrame) -> pl.DataFrame:
        print("   ⚙️ Normalizando tipos e filtrando coordenadas...", flush=True)
        # 1. LAT/LON para Float (SSP usa vírgula)
        df = df.with_columns([
            pl.col("LATITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False),
            pl.col("LONGITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False)
        ])

        # 2. Filtro de Segurança Espacial
        df = df.filter(
            (pl.col("LATITUDE").is_not_null()) & 
            (pl.col("LATITUDE") < -10) & # Filtra coordenadas fora de SP/Brasil
            (pl.col("LONGITUDE").is_not_null())
        )
        return df

    def extrair_bronze(self, ano: int):
        path = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        if ano < self.ano_atual and self._arquivo_existe(path):
            logger.info(f"⏭️ Bronze {ano} já existe.")
            return
        url = self.config.URL_BASE_SSP.format(ano=ano)
        try:
            res = requests.get(url, timeout=600)
            if res.status_code == 200:
                self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path, Body=res.content)
                logger.info(f"✅ Bronze {ano} salva.")
            else: logger.error(f"❌ Erro SSP {ano}: {res.status_code}")
        except Exception as e: logger.error(f"❌ Falha no Download {ano}: {e}")

    def processar_prata(self, ano: int):
        path_b = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_p = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"

        if self._arquivo_existe(path_p):
            logger.info(f"⏭️ Prata {ano} já processada.")
            return

        try:
            logger.info(f"🥈 Analisando estrutura de {ano}...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=path_b)
            excel_data = obj['Body'].read()
            
            # FAREJADOR DE ABAS: Tenta as 3 primeiras abas (Ocorrências costuma mudar de lugar)
            df = None
            mapeamento = {}
            for sheet in [1, 2, 3]:
                try:
                    temp_df = pl.read_excel(io.BytesIO(excel_data), sheet_id=sheet, engine="calamine")
                    temp_df.columns = [c.upper().strip() for c in temp_df.columns]
                    mapeamento = self._resolver_colunas_fuzzy(temp_df.columns)
                    
                    if "LATITUDE" in mapeamento.values() and "LONGITUDE" in mapeamento.values():
                        logger.info(f"   🎯 Estrutura encontrada na ABA {sheet}!")
                        df = temp_df
                        break
                except: continue

            if df is None:
                raise ValueError(f"Não foi possível localizar colunas de Latitude/Longitude em nenhuma aba de {ano}!")

            # Seleciona e renomeia
            df = df.select(list(mapeamento.keys())).rename(mapeamento)
            df = df.with_columns(pl.all().cast(pl.Utf8)) # Schema-on-read: tudo pra string
            
            # Tratamento Prata
            df = self._limpar_e_tipar(df)
            
            # LGPD
            def hash_lgpd(v):
                return hashlib.sha256(f"{v}{self.config.PEPPER}".encode()).hexdigest()
            
            df = df.with_columns([
                pl.col("NUM_BO").map_elements(hash_lgpd, return_dtype=pl.Utf8),
                pl.col("LOGRADOURO").map_elements(hash_lgpd, return_dtype=pl.Utf8)
            ])

            # H3
            df = df.with_columns(
                pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                    lambda x: h3.latlng_to_cell(x["LATITUDE"], x["LONGITUDE"], self.config.RESOLUCAO_H3),
                    return_dtype=pl.Utf8
                ).alias("H3_INDEX")
            )

            # Salvar
            buf = io.BytesIO()
            df.write_parquet(buf, compression="zstd")
            self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_p, Body=buf.getvalue())
            logger.info(f"✨ Prata {ano} finalizada: {df.height} ocorrências.")

        except Exception as e:
            logger.error(f"💥 Falha na Prata {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    anos = range(2022, ingestor.ano_atual + 1)
    for ano in anos:
        if modo in ["bronze", "tudo"]: ingestor.extrair_bronze(ano)
        if modo in ["prata", "tudo"]: ingestor.processar_prata(ano)
