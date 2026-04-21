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
    
    # Colunas que o modelo SafeDriver exige
    COLUNAS_ALVO = [
        "NUM_BO", "MUNICIPIO", "BAIRRO", "LOGRADOURO",
        "DATAOCORRENCIA", "HORAOCORRENCIA", "RUBRICA", 
        "LATITUDE", "LONGITUDE", "DESCR_TIPOLOCAL"
    ]

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
        except:
            return False

    def _limpar_e_converter_tipos(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Aqui é onde a mágica acontece: transformamos as Strings da Bronze 
        nos tipos reais da Prata (Float, Int, Date).
        """
        print("   ⚙️ Convertendo strings da Bronze para tipos Prata...", flush=True)
        
        return df.with_columns([
            # 1. LAT/LON: Troca vírgula por ponto e converte pra Float64
            pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False),
            pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False),
            
            # 2. Padronização de textos básicos
            pl.col("MUNICIPIO").str.to_uppercase().fill_null("NAO INFORMADO"),
            pl.col("BAIRRO").str.to_uppercase().fill_null("NAO INFORMADO"),
            
            # 3. Garantir que RUBRICA (tipo de crime) esteja limpo para o XGBoost
            pl.col("RUBRICA").str.to_uppercase().str.strip_chars()
        ]).filter(
            # Remove lixo de coordenadas (linhas sem GPS não servem para o SafeDriver)
            (pl.col("LATITUDE").is_not_null()) & 
            (pl.col("LATITUDE") != 0)
        )

    def _aplicar_lgpd(self, df: pl.DataFrame) -> pl.DataFrame:
        """Anonimização via SHA-256"""
        def hash_value(val):
            if val is None: return "ANONIMIZADO"
            return hashlib.sha256(f"{str(val).upper()}{self.config.PEPPER}".encode()).hexdigest()

        return df.with_columns([
            pl.col("NUM_BO").map_elements(hash_value, return_dtype=pl.Utf8),
            pl.col("LOGRADOURO").map_elements(hash_value, return_dtype=pl.Utf8)
        ])

    def _indexar_h3(self, df: pl.DataFrame) -> pl.DataFrame:
        """Cálculo do Hexágono H3"""
        return df.with_columns(
            pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                lambda x: h3.latlng_to_cell(x["LATITUDE"], x["LONGITUDE"], self.config.RESOLUCAO_H3),
                return_dtype=pl.Utf8
            ).alias("H3_INDEX")
        )

    def extrair_bronze(self, ano: int):
        """Salva o Excel BRUTO (Raw) no R2"""
        path_bronze = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        
        if ano < self.ano_atual and self._arquivo_existe(path_bronze):
            logger.info(f"⏭️ [BRONZE] {ano} já existe. Pulando download.")
            return

        url = self.config.URL_BASE_SSP.format(ano=ano)
        try:
            logger.info(f"🚀 [BRONZE] Baixando ano {ano}...")
            res = requests.get(url, timeout=300)
            if res.status_code == 200:
                self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_bronze, Body=res.content)
                logger.info(f"✅ [BRONZE] {ano} salvo no R2.")
            else:
                logger.error(f"❌ Erro download {ano}: Status {res.status_code}")
        except Exception as e:
            logger.error(f"❌ Falha crítica no download {ano}: {e}")

    def processar_prata(self, ano: int):
        """Lê a Bronze como String e converte para Prata Trusted"""
        path_bronze = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_prata = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"

        if self._arquivo_existe(path_prata):
            logger.info(f"⏭️ [PRATA] {ano} já processado. Pulando.")
            return

        try:
            logger.info(f"🥈 [PRATA] Iniciando tratamento do ano {ano}...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=path_bronze)
            content = obj['Body'].read()
            
            # 1. LER TUDO COMO STRING (A solução para o seu problema)
            # Ao usar o motor 'calamine', o Polars lê o Excel. 
            # Em seguida, forçamos o cast de TODAS as colunas para String imediatamente.
            df = pl.read_excel(io.BytesIO(content), sheet_id=1, engine="calamine")
            df = df.with_columns(pl.all().cast(pl.Utf8)) 
            
            # 2. Padronização de Nomes de Colunas
            df.columns = [c.upper().replace(" ", "_").strip() for c in df.columns]
            
            # 3. Selecionar apenas o que importa (Evita carregar lixo na memória)
            cols_disponiveis = [c for c in self.config.COLUNAS_ALVO if c in df.columns]
            df = df.select(cols_disponiveis)

            # 4. Converter as strings nos tipos corretos (Float, String Limpa, etc)
            df = self._limpar_e_converter_tipos(df)

            # 5. LGPD (Anonimizar antes de salvar na Prata)
            df = self._aplicar_lgpd(df)

            # 6. H3 Index (Espacialização)
            df = self._indexar_h3(df)

            # 7. Salvar em Parquet (Muito mais leve que Excel para o Power BI / ML)
            buffer = io.BytesIO()
            df.write_parquet(buffer, compression="snappy")
            self.s3.put_object(
                Bucket=self.config.NOME_BUCKET, 
                Key=path_prata, 
                Body=buffer.getvalue()
            )
            logger.info(f"✨ [PRATA] {ano} finalizada: {df.height} ocorrências georreferenciadas.")

        except Exception as e:
            logger.error(f"💥 Erro fatal na Prata {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    
    anos = range(2022, ingestor.ano_atual + 1)

    for ano in anos:
        if modo in ["bronze", "tudo"]:
            ingestor.extrair_bronze(ano)
        
        if modo in ["prata", "tudo"]:
            ingestor.processar_prata(ano)
