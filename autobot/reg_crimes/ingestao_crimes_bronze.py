import os
import boto3
import requests
import logging
import io
import time
import polars as pl
import h3
import hashlib
from botocore.config import Config
from datetime import datetime
from typing import Optional

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

    def _inicializar_s3(self):
        return boto3.client(
            's3', 
            endpoint_url=self.config.ENDPOINT_URL.strip().rstrip('/'),
            aws_access_key_id=self.config.ACCESS_KEY.strip(),
            aws_secret_access_key=self.config.SECRET_KEY.strip(),
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )

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

    # --- ETAPA 1: BRONZE (INGESTÃO PURA) ---
    def extrair_bronze(self, ano: int, max_tentativas: int = 0):
        """
        Baixa o arquivo da SSP e salva 'as-is' no R2.
        max_tentativas = 0 significa infinito.
        """
        url = self.config.URL_BASE_SSP.format(ano=ano)
        path_bronze = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        tentativa = 1

        while True:
            try:
                logger.info(f"🚀 [{ano}] Tentativa {tentativa}: Baixando...")
                res = requests.get(url, timeout=600) # Timeout alto para arquivos grandes
                
                if res.status_code == 200:
                    self.s3.put_object(
                        Bucket=self.config.NOME_BUCKET, 
                        Key=path_bronze, 
                        Body=res.content
                    )
                    logger.info(f"✅ [{ano}] Bronze salva com sucesso no R2.")
                    break
                else:
                    logger.warning(f"⚠️ [{ano}] Erro HTTP {res.status_code}. Re-tentando em 30s...")
            
            except Exception as e:
                logger.error(f"❌ [{ano}] Falha na conexão: {e}. Re-tentando...")
            
            if 0 < max_tentativas <= tentativa:
                logger.error(f"🛑 [{ano}] Limite de tentativas atingido.")
                break
            
            tentativa += 1
            time.sleep(30)

    # --- ETAPA 2: PRATA (TRANSFORMAÇÃO) ---
    def processar_prata(self, ano: int):
        """
        Lê o arquivo da Bronze no R2, processa e salva na Prata em Parquet.
        """
        path_bronze = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_prata = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"

        try:
            logger.info(f"🥈 [{ano}] Lendo Bronze do R2 para processamento...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=path_bronze)
            bytes_excel = obj['Body'].read()

            # Processamento Polars
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

            if not dfs:
                logger.error(f"❌ [{ano}] Nenhuma aba válida encontrada no Excel.")
                return

            df_final = pl.concat(dfs, how="diagonal")

            # Tratamento Geográfico e LGPD
            df_final = df_final.with_columns([
                pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0)
            ])

            df_final = df_final.with_columns(
                pl.struct(["LOGRADOURO", "NUM_BO"]).map_elements(
                    lambda x: self._gerar_hash_lgpd(x["LOGRADOURO"], x["NUM_BO"]),
                    return_dtype=pl.String
                ).alias("HASH_LOCAL_UNICO")
            ).drop(["LOGRADOURO", "NUM_BO"])

            # Cálculo de H3 Otimizado
            geo_map = df_final.select(["LATITUDE", "LONGITUDE"]).unique().to_dicts()
            for d in geo_map:
                d["H3_INDEX"] = self._calcular_h3(d["LATITUDE"], d["LONGITUDE"])
            
            df_trusted = df_final.join(pl.DataFrame(geo_map), on=["LATITUDE", "LONGITUDE"], how="left")

            # Escrita na Prata
            buffer = io.BytesIO()
            df_trusted.write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_prata, Body=buffer.getvalue())
            logger.info(f"✨ [{ano}] Prata gerada e salva com sucesso.")

        except Exception as e:
            logger.error(f"❌ [{ano}] Erro ao processar Prata: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    
    # Exemplo: Rodar Bronze e Prata para os últimos anos
    for ano in [2024, 2025, 2026]:
        # Você pode comentar uma das linhas abaixo se quiser rodar só uma etapa
        ingestor.extrair_bronze(ano) 
        ingestor.processar_prata(ano)
