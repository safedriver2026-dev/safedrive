import os
import boto3
import requests
import logging
import io
import polars as pl
import h3
import hashlib
from botocore.config import Config
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ConfiguracaoIngestao:
    # Nomes batendo com o YAML
    NOME_BUCKET = os.getenv("R2_BUCKET_NAME")
    ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
    ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
    SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
    
    PEPPER = os.getenv("LGPD_PEPPER", "default_pepper")
    SALT = os.getenv("LGPD_SALT", "default_salt")

    URL_BASE_SSP = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
    RESOLUCAO_H3 = 9
    
    COLUNAS_ALVO = [
        "NUM_BO", "MUNICIPIO", "CIDADE", "NOME_MUNICIPIO", "BAIRRO", "LOGRADOURO",
        "DATA_OCORRENCIA_BO", "DATA_OCORRENCIA", "HORA_OCORRENCIA_BO", 
        "DESC_PERIODO", "RUBRICA", "LATITUDE", "LONGITUDE", 
        "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL"
    ]
    ANCORAS_CABECALHO = {"LOGRADOURO", "MUNICIPIO", "RUBRICA", "LATITUDE", "DESC_PERIODO"}

class IngestorSafeDriver:
    def __init__(self):
        self.config = ConfiguracaoIngestao()
        self._validar_ambiente()
        self.s3 = self._inicializar_s3()
        self.sessao = self._inicializar_http()

    def _validar_ambiente(self):
        """Impede o erro de 'length 0' logo no início"""
        if not self.config.ACCESS_KEY or len(self.config.ACCESS_KEY) == 0:
            raise ValueError("❌ R2_ACCESS_KEY_ID não encontrada no ambiente!")

    def _inicializar_s3(self):
        return boto3.client(
            's3', 
            endpoint_url=self.config.ENDPOINT_URL.strip().rstrip('/'),
            aws_access_key_id=self.config.ACCESS_KEY.strip(),
            aws_secret_access_key=self.config.SECRET_KEY.strip(),
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )

    def _inicializar_http(self):
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=retries))
        return s

    def _gerar_hash_lgpd(self, logradouro: str, id_bo: str) -> str:
        if not logradouro or str(logradouro).upper() in ["NONE", "NULL", "NAN"]:
            return "LOCAL_ANONIMIZADO"
        payload = f"{self.config.PEPPER}{logradouro}{id_bo}{self.config.SALT}".encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def _calcular_h3(self, lat, lon):
        try:
            lat, lon = float(lat), float(lon)
            if lat == 0 or lon == 0 or abs(lat) > 90 or abs(lon) > 180: return None
            return h3.latlng_to_cell(lat, lon, self.config.RESOLUCAO_H3)
        except: return None

    def _processar_e_proteger(self, bytes_excel: bytes) -> Optional[pl.DataFrame]:
        fluxo = io.BytesIO(bytes_excel)
        dfs = []
        for aba in range(1, 4):
            try:
                # O motor calamine é essencial para o XLSX da SSP
                df = pl.read_excel(fluxo, sheet_id=aba, engine="calamine")
                df.columns = [c.upper().strip() for c in df.columns]
                cols = [c for c in df.columns if c in self.config.COLUNAS_ALVO]
                df = df.select(cols).with_columns(pl.all().cast(pl.String))
                dfs.append(df)
            except: continue

        if not dfs: return None
        df_final = pl.concat(dfs, how="diagonal")

        # 1. Coordenadas
        df_final = df_final.with_columns([
            pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0)
        ])

        # 2. LGPD
        df_final = df_final.with_columns(
            pl.struct(["LOGRADOURO", "NUM_BO"]).map_elements(
                lambda x: self._gerar_hash_lgpd(x["LOGRADOURO"], x["NUM_BO"]),
                return_dtype=pl.String
            ).alias("HASH_LOCAL_UNICO")
        ).drop(["LOGRADOURO", "NUM_BO"])

        # 3. H3
        geo_map = df_final.select(["LATITUDE", "LONGITUDE"]).unique().to_dicts()
        for d in geo_map:
            d["H3_INDEX"] = self._calcular_h3(d["LATITUDE"], d["LONGITUDE"])
        
        return df_final.join(pl.DataFrame(geo_map), on=["LATITUDE", "LONGITUDE"], how="left")

    def rodar_ano(self, ano: int):
        path_bronze = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_prata = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"
        url = self.config.URL_BASE_SSP.format(ano=ano)

        try:
            logger.info(f"[{ano}] Iniciando download...")
            res = self.sessao.get(url, timeout=400)
            if res.status_code != 200: return

            # Bronze
            self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_bronze, Body=res.content)
            logger.info(f"📁 [{ano}] Bronze salva.")

            # Prata
            df_trusted = self._processar_e_proteger(res.content)
            if df_trusted is not None:
                buffer = io.BytesIO()
                df_trusted.write_parquet(buffer, compression="lz4")
                self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_prata, Body=buffer.getvalue())
                logger.info(f"✨ [{ano}] Prata salva.")

        except Exception as e:
            logger.error(f"❌ Erro no ciclo do ano {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    ano_atual = datetime.now().year
    for ano in range(2022, ano_atual + 1):
        ingestor.rodar_ano(ano)
