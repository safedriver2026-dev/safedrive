import os
import boto3
import requests
import logging
import io
import polars as pl
import h3
from botocore.config import Config
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfiguracaoIngestao:
    NOME_BUCKET = os.getenv("R2_BUCKET_NAME", "safedriver").strip()
    URL_BASE_SSP = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
    RESOLUCAO_H3 = 9
    
    COLUNAS_ALVO = [
        "MUNICIPIO", "CIDADE", "NOME_MUNICIPIO", "BAIRRO", "LOGRADOURO",
        "DATA_OCORRENCIA_BO", "DATA_OCORRENCIA", "HORA_OCORRENCIA_BO", 
        "DESC_PERIODO", "RUBRICA", "LATITUDE", "LONGITUDE", 
        "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL"
    ]
    ANCORAS_CABECALHO = {"LOGRADOURO", "MUNICIPIO", "RUBRICA", "LATITUDE", "DESC_PERIODO"}
    PADRAO_LIMPEZA = r"(?i)PRESENTE TABELA|FINALIDADE ESCLARECER|CAMPOS CONTIDOS|BASE DE DADOS"

class IngestorSafeDriver:
    def __init__(self):
        self.config = ConfiguracaoIngestao()
        self.s3 = self._inicializar_s3()
        self.sessao = self._inicializar_http()

    def _inicializar_s3(self):
        return boto3.client(
            's3', 
            endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )

    def _inicializar_http(self):
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=retries))
        return s

    def _calcular_h3(self, lat, lon):
        try:
            lat, lon = float(lat), float(lon)
            if lat == 0 or lon == 0 or abs(lat) > 90 or abs(lon) > 180: return None
            return h3.latlng_to_cell(lat, lon, self.config.RESOLUCAO_H3)
        except: return None

    def _processar_planilha(self, bytes_excel: bytes) -> Optional[pl.DataFrame]:
        """Lógica de extração e limpeza dos B.O.s"""
        fluxo = io.BytesIO(bytes_excel)
        dfs = []

        # Varre as abas principais (geralmente as 3 primeiras contêm os dados)
        for aba in range(1, 4):
            try:
                # 1. Encontrar o cabeçalho dinamicamente
                preview = pl.read_excel(fluxo, sheet_id=aba, engine="calamine", has_header=False, read_options={"n_rows": 50})
                idx_header = None
                for i, linha in enumerate(preview.iter_rows()):
                    vals = [str(c).upper() for c in linha if c is not None]
                    if len([v for v in vals if v in self.config.ANCORAS_CABECALHO]) >= 3:
                        idx_header = i
                        break
                
                if idx_header is not None:
                    df = pl.read_excel(fluxo, sheet_id=aba, engine="calamine", read_options={"skip_rows": idx_header})
                    df.columns = [c.upper().strip() for c in df.columns]
                    
                    # 2. Filtragem e Cast
                    cols = [c for c in df.columns if c in self.config.COLUNAS_ALVO]
                    df = df.select(cols).with_columns(pl.all().cast(pl.String))
                    df = df.filter(~pl.any_horizontal(pl.all().str.contains(self.config.PADRAO_LIMPEZA)))
                    dfs.append(df)
            except: continue

        if not dfs: return None

        # 3. Consolidação e Enriquecimento Geo
        df_final = pl.concat(dfs, how="diagonal").with_columns([
            pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0)
        ])

        # Otimização H3 (Calcula apenas para pontos únicos)
        geo_map = df_final.select(["LATITUDE", "LONGITUDE"]).unique().to_dicts()
        for d in geo_map:
            d["H3_INDEX"] = self._calcular_h3(d["LATITUDE"], d["LONGITUDE"])
        
        return df_final.join(pl.DataFrame(geo_map), on=["LATITUDE", "LONGITUDE"], how="left")

    def rodar_ano(self, ano: int):
        # Definição das novas pastas que você pediu
        path_bronze = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_prata = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"
        url = self.config.URL_BASE_SSP.format(ano=ano)

        try:
            logger.info(f"[{ano}] Buscando dados na SSP-SP...")
            res = self.sessao.get(url, timeout=300)
            if res.status_code != 200: return

            # Salva o Bruto (Bronze)
            self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_bronze, Body=res.content)
            logger.info(f"[{ano}] Arquivo RAW salvo em bronze/crimes_raw")

            # Processa e Salva o Trusted (Prata)
            df = self._processar_planilha(res.content)
            if df is not None:
                buffer = io.BytesIO()
                df.write_parquet(buffer, compression="lz4") # LZ4 é excelente para performance
                self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_prata, Body=buffer.getvalue())
                logger.info(f"[{ano}] Dados limpos e H3 salvos em prata/crimes_trusted")

        except Exception as e:
            logger.error(f"Erro no ano {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    ano_atual = datetime.now().year
    # Roda para o ano atual e o anterior
    for ano in [ano_atual - 1, ano_atual]:
        ingestor.rodar_ano(ano)
