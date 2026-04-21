import os
import boto3
import requests
import logging
import time
import sys
import io
import hashlib
import botocore.exceptions
import polars as pl
import h3
from botocore.config import Config
from datetime import datetime

# Configuração de Log Profissional
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
    
    # Mapeamento Flexível (Gaps de nomes que a SSP costuma mudar)
    MAPA_COLUNAS = {
        "NUM_BO": ["NUM_BO", "NÚMERO_BO", "NUMERO_BO"],
        "MUNICIPIO": ["MUNICIPIO", "CIDADE", "NOME_MUNICIPIO"],
        "BAIRRO": ["BAIRRO", "NOME_BAIRRO"],
        "LOGRADOURO": ["LOGRADOURO", "DESCR_LOGRADOURO", "RUA"],
        "DATAOCORRENCIA": ["DATAOCORRENCIA", "DATA_OCORRENCIA", "DATA_OCORRENCIA_BO"],
        "HORAOCORRENCIA": ["HORAOCORRENCIA", "HORA_OCORRENCIA", "HORA_OCORRENCIA_BO"],
        "RUBRICA": ["RUBRICA", "DESCR_RUBRICA", "NATUREZA"],
        "LATITUDE": ["LATITUDE", "LAT"],
        "LONGITUDE": ["LONGITUDE", "LON", "LONG"],
        "DESCR_TIPOLOCAL": ["DESCR_TIPOLOCAL", "TIPO_LOCAL", "LOCAL"]
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

    def _limpar_e_tipar(self, df: pl.DataFrame) -> pl.DataFrame:
        """Trata as strings da Bronze para os tipos reais da Prata."""
        print("   ⚙️ Refinando tipos e limpando coordenadas...", flush=True)
        
        # 1. Garantir que Lat/Lon sejam números (SSP usa vírgula)
        df = df.with_columns([
            pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False),
            pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False)
        ])

        # 2. Filtro de Qualidade (Crime sem coordenada não treina modelo de mapa)
        df = df.filter(
            (pl.col("LATITUDE").is_not_null()) & 
            (pl.col("LATITUDE") != 0) &
            (pl.col("LATITUDE") < -10) # Coordenada válida para SP
        )

        # 3. Normalização de Texto
        df = df.with_columns([
            pl.all().exclude(["LATITUDE", "LONGITUDE"]).map_elements(
                lambda x: str(x).upper().strip() if x is not None else "NAO INFORMADO",
                return_dtype=pl.Utf8
            )
        ])
        
        return df

    def _aplicar_lgpd(self, df: pl.DataFrame) -> pl.DataFrame:
        """Pseudonimização irreversível."""
        def hash_v(val):
            return hashlib.sha256(f"{val}{self.config.PEPPER}".encode()).hexdigest()

        return df.with_columns([
            pl.col("NUM_BO").map_elements(hash_v, return_dtype=pl.Utf8),
            pl.col("LOGRADOURO").map_elements(hash_v, return_dtype=pl.Utf8)
        ])

    def _resolver_colunas(self, df_cols):
        """Inteligência para achar as colunas certas mesmo se a SSP mudar o nome."""
        selecao = {}
        for alvo, variantes in self.config.MAPA_COLUNAS.items():
            for v in variantes:
                if v in df_cols:
                    selecao[v] = alvo
                    break
        return selecao

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
            else: logger.error(f"❌ Falha SSP {ano}: {res.status_code}")
        except Exception as e: logger.error(f"❌ Erro Download {ano}: {e}")

    def processar_prata(self, ano: int):
        path_b = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_p = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"

        if self._arquivo_existe(path_p):
            logger.info(f"⏭️ Prata {ano} já processada.")
            return

        try:
            logger.info(f"🥈 Processando Prata {ano}...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=path_b)
            
            # Tenta ler com calamine (motor mais estável para Excel sujo)
            df = pl.read_excel(io.BytesIO(obj['Body'].read()), sheet_id=1, engine="calamine")
            
            # Achata tudo para String imediatamente para evitar erros de esquema (Schema-on-read)
            df = df.with_columns(pl.all().cast(pl.Utf8))
            
            # Resolve nomes de colunas dinamicamente
            df.columns = [c.upper().replace(" ", "_").strip() for c in df.columns]
            mapeamento = self._resolver_colunas(df.columns)
            
            if "LATITUDE" not in mapeamento.values():
                raise ValueError(f"Arquivo de {ano} não possui coluna de Latitude identificável!")

            df = df.select(list(mapeamento.keys())).rename(mapeamento)

            # Transformações e Inteligência
            df = self._limpar_e_tipar(df)
            df = self._aplicar_lgpd(df)
            
            # H3 Index
            df = df.with_columns(
                pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                    lambda x: h3.latlng_to_cell(x["LATITUDE"], x["LONGITUDE"], self.config.RESOLUCAO_H3),
                    return_dtype=pl.Utf8
                ).alias("H3_INDEX")
            )

            # Upload Parquet
            buf = io.BytesIO()
            df.write_parquet(buf, compression="zstd")
            self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_p, Body=buf.getvalue())
            logger.info(f"✨ Prata {ano} finalizada: {df.height} linhas.")

        except Exception as e:
            logger.error(f"💥 Falha na Prata {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    anos = range(2022, ingestor.ano_atual + 1)

    for ano in anos:
        if modo in ["bronze", "tudo"]: ingestor.extrair_bronze(ano)
        if modo in ["prata", "tudo"]: ingestor.processar_prata(ano)
