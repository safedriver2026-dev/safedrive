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
    
    # Mapeamento Ultra Flexível (RegEx) para encontrar as colunas independente do nome
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
                col_str = str(col).upper().strip()
                for p in padroes:
                    if re.search(p, col_str, re.IGNORECASE):
                        mapeamento_final[col] = alvo
                        break
                if col in mapeamento_final: break
        return mapeamento_final

    def _encontrar_cabecalho_real(self, excel_data, sheet_id):
        """Varre as primeiras 20 linhas da aba para achar onde começa o cabeçalho real."""
        try:
            # Lê as primeiras 20 linhas como string para analisar
            df_teste = pl.read_excel(io.BytesIO(excel_data), sheet_id=sheet_id, engine="calamine", read_options={"n_rows": 20, "has_header": False})
            
            for i, row in enumerate(df_teste.iter_rows()):
                row_str = [str(cell).upper() for cell in row]
                # Se encontrarmos pelo menos 3 colunas chaves, essa é a linha do cabeçalho
                matches = 0
                for cell in row_str:
                    if any(re.search(p, cell) for padroes in self.config.MAPA_COLUNAS.values() for p in padroes):
                        matches += 1
                
                if matches >= 4: # Encontrou o cabeçalho
                    return i
            return None
        except:
            return None

    def _limpar_e_tipar(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.with_columns([
            pl.col("LATITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False),
            pl.col("LONGITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False)
        ])
        return df.filter((pl.col("LATITUDE").is_not_null()) & (pl.col("LATITUDE") < -10))

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
            logger.info(f"🥈 Caçando dados na estrutura de {ano}...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=path_b)
            excel_data = obj['Body'].read()
            
            df_final = None
            mapeamento = {}

            # Varre as abas (Sheet IDs em calamine começam em 1)
            for sheet_idx in range(1, 10):
                skip_linha = self._encontrar_cabecalho_real(excel_data, sheet_idx)
                
                if skip_linha is not None:
                    logger.info(f"   🎯 Cabeçalho encontrado na Aba {sheet_idx}, linha {skip_linha + 1}")
                    
                    # Lê a aba pulando as linhas inúteis da "capa"
                    df_raw = pl.read_excel(
                        io.BytesIO(excel_data), 
                        sheet_id=sheet_idx, 
                        engine="calamine", 
                        read_options={"skip_rows": skip_linha}
                    )
                    
                    # Normaliza nomes de colunas encontrados
                    df_raw.columns = [str(c).upper().strip() for c in df_raw.columns]
                    mapeamento = self._resolver_colunas_fuzzy(df_raw.columns)
                    
                    if "LATITUDE" in mapeamento.values() and "LONGITUDE" in mapeamento.values():
                        df_final = df_raw.select(list(mapeamento.keys())).rename(mapeamento)
                        break

            if df_final is None:
                raise ValueError(f"Não foi possível encontrar a tabela de crimes em nenhuma aba de {ano}.")

            # Tratamento Prata
            df_final = df_final.with_columns(pl.all().cast(pl.Utf8))
            df_final = self._limpar_e_tipar(df_final)
            
            # LGPD (Hash)
            pepper = self.config.PEPPER
            df_final = df_final.with_columns([
                pl.col("NUM_BO").map_elements(lambda v: hashlib.sha256(f"{v}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8),
                pl.col("LOGRADOURO").map_elements(lambda v: hashlib.sha256(f"{v}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8)
            ])

            # H3 Index
            res_h3 = self.config.RESOLUCAO_H3
            df_final = df_final.with_columns(
                pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                    lambda x: h3.latlng_to_cell(x["LATITUDE"], x["LONGITUDE"], res_h3),
                    return_dtype=pl.Utf8
                ).alias("H3_INDEX")
            )

            # Upload
            buf = io.BytesIO()
            df_final.write_parquet(buf, compression="zstd")
            self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_p, Body=buf.getvalue())
            logger.info(f"✨ Prata {ano} salva: {df_final.height} linhas georreferenciadas.")

        except Exception as e:
            logger.error(f"💥 Falha na Prata {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    for ano in range(2022, ingestor.ano_atual + 1):
        if modo in ["bronze", "tudo"]: ingestor.extrair_bronze(ano)
        if modo in ["prata", "tudo"]: ingestor.processar_prata(ano)
