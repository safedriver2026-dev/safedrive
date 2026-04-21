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
    
    # Mapeamento Ultra Flexível (RegEx)
    MAPA_COLUNAS = {
        "NUM_BO": [r"NUM.*BO", r"N.MERO.*BO", r"BO_NUMERO"],
        "MUNICIPIO": [r"MUNIC.PIO", r"CIDADE", r"NM_MUN"],
        "BAIRRO": [r"BAIRRO", r"NM_BAIRRO"],
        "LOGRADOURO": [r"LOGRADOURO", r"RUA", r"DESCR_LOG", r"NM_LOG"],
        "DATAOCORRENCIA": [r"DATA.*OCORR", r"DT_OCORR", r"DATA_OCORRENCIA"],
        "HORAOCORRENCIA": [r"HORA.*OCORR", r"HR_OCORR", r"HORA_OCORRENCIA"],
        "RUBRICA": [r"RUBRICA", r"NATUREZA", r"DESCR_RUBRICA"],
        "LATITUDE": [r"LATITUDE", r"LAT.*GEO", r"^LAT$", r"COORDENADA_X"],
        "LONGITUDE": [r"LONGITUDE", r"LON.*GEO", r"^LON$", r"COORDENADA_Y"],
        "DESCR_TIPOLOCAL": [r"TIPOLOCAL", r"LOCAL_OCORR", r"DESCR_TIPOLOCAL"]
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
            config=Config(signature_version='s3v4')
        )

    def _arquivo_existe(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.config.NOME_BUCKET, Key=key)
            return True
        except: return False

    def _resolver_colunas_fuzzy(self, colunas_reais):
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
        # 1. LAT/LON para Float (Trata string com vírgula e lixo de texto)
        df = df.with_columns([
            pl.col("LATITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False),
            pl.col("LONGITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False)
        ])

        # 2. Filtro: Se não tem coordenada válida, não serve para o SafeDriver
        return df.filter(
            (pl.col("LATITUDE").is_not_null()) & (pl.col("LATITUDE") < -10)
        )

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
            logger.info(f"🥈 Analisando estrutura dinâmica de {ano}...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=path_b)
            excel_data = obj['Body'].read()
            
            # POLARS EXTREME: Tenta descobrir quais abas existem
            # Como polars read_excel com calamine não lista abas facilmente sem carregar, 
            # vamos iterar por tentativa e erro em um range maior
            df = None
            mapeamento = {}
            
            for sheet_idx in range(1, 10): # Tenta até a décima aba
                try:
                    # Tenta ler a aba pulando de 0 a 3 linhas (caso a SSP tenha colocado títulos no topo)
                    for skip in [0, 1, 2, 3]:
                        temp_df = pl.read_excel(io.BytesIO(excel_data), sheet_id=sheet_idx, engine="calamine", read_options={"skip_rows": skip})
                        temp_df.columns = [str(c).upper().strip() for c in temp_df.columns]
                        mapeamento = self._resolver_colunas_fuzzy(temp_df.columns)
                        
                        if "LATITUDE" in mapeamento.values() and "LONGITUDE" in mapeamento.values():
                            logger.info(f"   🎯 Sucesso! Dados encontrados na ABA {sheet_idx} (pulando {skip} linhas)")
                            df = temp_df
                            break
                    if df is not None: break
                except Exception:
                    continue

            if df is None:
                # Log de debug para você ver o que veio no arquivo
                logger.error(f"❌ Nenhuma aba de {ano} contém LAT/LON. Verificando cabeçalhos da Aba 1 para debug:")
                debug_df = pl.read_excel(io.BytesIO(excel_data), sheet_id=1, engine="calamine")
                logger.info(f"   Colunas encontradas na Aba 1: {debug_df.columns[:15]}")
                raise ValueError(f"Estrutura irreconhecível em {ano}")

            # Processamento final
            df = df.select(list(mapeamento.keys())).rename(mapeamento)
            df = df.with_columns(pl.all().cast(pl.Utf8)) 
            df = self._limpar_e_tipar(df)
            
            # LGPD
            pepper = self.config.PEPPER
            df = df.with_columns([
                pl.col("NUM_BO").map_elements(lambda v: hashlib.sha256(f"{v}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8),
                pl.col("LOGRADOURO").map_elements(lambda v: hashlib.sha256(f"{v}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8)
            ])

            # H3
            res = self.config.RESOLUCAO_H3
            df = df.with_columns(
                pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                    lambda x: h3.latlng_to_cell(x["LATITUDE"], x["LONGITUDE"], res),
                    return_dtype=pl.Utf8
                ).alias("H3_INDEX")
            )

            buf = io.BytesIO()
            df.write_parquet(buf, compression="zstd")
            self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_p, Body=buf.getvalue())
            logger.info(f"✨ Prata {ano} salva: {df.height} linhas.")

        except Exception as e:
            logger.error(f"💥 Falha na Prata {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    for ano in range(2022, ingestor.ano_atual + 1):
        if modo in ["bronze", "tudo"]: ingestor.extrair_bronze(ano)
        if modo in ["prata", "tudo"]: ingestor.processar_prata(ano)
