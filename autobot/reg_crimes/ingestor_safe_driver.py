import os
import boto3
import requests
import logging
import time
import sys
import botocore.exceptions
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

class IngestorSafeDriver:
    def __init__(self):
        self.config = ConfiguracaoIngestao()
        self.s3 = self._inicializar_s3()
        self.ano_atual = datetime.now().year

    def _inicializar_s3(self):
        return boto3.client(
            's3', 
            endpoint_url=self.config.ENDPOINT_URL.strip().rstrip('/'),
            aws_access_key_id=self.config.ACCESS_KEY.strip(),
            aws_secret_access_key=self.config.SECRET_KEY.strip(),
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )

    def _arquivo_existe(self, key: str) -> bool:
        """Verifica se o arquivo já existe no R2 usando apenas os metadados (head_object)"""
        try:
            self.s3.head_object(Bucket=self.config.NOME_BUCKET, Key=key)
            return True
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            return False

    def _gerar_hash_lgpd(self, logradouro: str, id_bo: str) -> str:
        import hashlib # Lazy import isolado
        if not logradouro or str(logradouro).upper() in ["NONE", "NULL", "NAN"]:
            return "LOCAL_ANONIMIZADO"
        payload = f"{self.config.PEPPER}{logradouro}{id_bo}{self.config.SALT}".encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def _calcular_h3(self, lat, lon):
        import h3 # Lazy import isolado
        try:
            lat, lon = float(lat), float(lon)
            if lat == 0 or lon == 0 or abs(lat) > 90 or abs(lon) > 180: return None
            return h3.latlng_to_cell(lat, lon, self.config.RESOLUCAO_H3)
        except: return None

    # --- ETAPA 1: BRONZE (LEVE E PURA) ---
    def extrair_bronze(self, ano: int):
        path_bronze = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        
        # Inteligência de Idempotência
        if ano < self.ano_atual and self._arquivo_existe(path_bronze):
            logger.info(f"⏭️ [BRONZE] Arquivo de {ano} já existe no R2. Pulando download.")
            return

        url = self.config.URL_BASE_SSP.format(ano=ano)
        tentativa = 1

        while True:
            try:
                logger.info(f"🚀 [BRONZE] {ano} - Tentativa {tentativa}")
                res = requests.get(url, timeout=600)
                if res.status_code == 200:
                    self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_bronze, Body=res.content)
                    logger.info(f"✅ [BRONZE] {ano} salva no R2.")
                    break
                logger.warning(f"⚠️ [BRONZE] Status {res.status_code}. Re-tentando...")
            except Exception as e:
                logger.error(f"❌ [BRONZE] Falha: {e}")
            
            tentativa += 1
            time.sleep(20)

    # --- ETAPA 2: PRATA (PESADA E ANALÍTICA) ---
    def processar_prata(self, ano: int):
        path_bronze = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_prata = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"

        # Inteligência de Idempotência
        if ano < self.ano_atual and self._arquivo_existe(path_prata):
            logger.info(f"⏭️ [PRATA] Parquet de {ano} já está pronto. Pulando processamento.")
            return

        import io # Lazy import isolado
        import polars as pl # Lazy import isolado

        try:
            logger.info(f"🥈 [PRATA] Lendo Bronze {ano} do R2...")
            
            # Se a Bronze não existir (ex: erro no download), aborta a Prata com graciosidade
            if not self._arquivo_existe(path_bronze):
                logger.error(f"❌ [PRATA] Arquivo base de {ano} não encontrado na Bronze. Abortando ano.")
                return

            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=path_bronze)
            bytes_excel = obj['Body'].read()

            fluxo = io.BytesIO(bytes_excel)
            dfs = []
            for aba in range(1, 4):
                try:
                    df = pl.read_excel(fluxo, sheet_id=aba, engine="calamine")
                    df.columns = [c.upper().strip() for c in df.columns]
                    cols = [c for c in df.columns if c in self.config.COLUNAS_ALVO]
                    df = df.select(cols).with_columns(pl.all().cast(pl.String))
                    dfs.append(df)
                except: continue

            if not dfs: return

            df_final = pl.concat(dfs, how="diagonal")
            df_final = df_final.with_columns([
                pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0)
            ])

            # LGPD e H3
            df_final = df_final.with_columns(
                pl.struct(["LOGRADOURO", "NUM_BO"]).map_elements(
                    lambda x: self._gerar_hash_lgpd(x["LOGRADOURO"], x["NUM_BO"]),
                    return_dtype=pl.String
                ).alias("HASH_LOCAL_UNICO")
            ).drop(["LOGRADOURO", "NUM_BO"])

            geo_map = df_final.select(["LATITUDE", "LONGITUDE"]).unique().to_dicts()
            for d in geo_map:
                d["H3_INDEX"] = self._calcular_h3(d["LATITUDE"], d["LONGITUDE"])
            
            df_trusted = df_final.join(pl.DataFrame(geo_map), on=["LATITUDE", "LONGITUDE"], how="left")

            buffer = io.BytesIO()
            df_trusted.write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_prata, Body=buffer.getvalue())
            logger.info(f"✨ [PRATA] {ano} finalizada.")

        except Exception as e:
            logger.error(f"❌ [PRATA] Erro no ano {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    anos = range(2022, ingestor.ano_atual + 1)

    for ano in anos:
        if modo in ["bronze", "tudo"]:
            ingestor.extrair_bronze(ano)
        
        if modo in ["prata", "tudo"]:
            ingestor.processar_prata(ano)
