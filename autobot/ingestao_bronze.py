import os
import boto3
import requests
import logging
import io
import polars as pl
from botocore.config import Config
from botocore.exceptions import ClientError
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class IngestaoBronze:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}))
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0"}

    def executar_ingestao_continua(self):
        logger.info("BRONZE: Iniciando extração com busca dinâmica de cabeçalhos (Header-Hunter).")
        ano_atual = datetime.now().year
        novos_dados = False
        for ano in range(2022, ano_atual + 1):
            if self._verificar_e_baixar(ano): novos_dados = True
        return novos_dados

    def _verificar_e_baixar(self, ano):
        path_raw = f"datalake/bronze/raw/ssp_raw_{ano}.xlsx"
        path_trusted = f"datalake/bronze/trusted/ssp_trusted_{ano}.parquet"
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"

        try:
            # 1. Recuperar o Excel (Prioriza R2, senão baixa)
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path_raw)
                excel_bytes = obj['Body'].read()
                logger.info(f"BRONZE: [{ano}] Usando cache do R2.")
            except:
                logger.info(f"BRONZE: [{ano}] Baixando da SSP-SP...")
                resp = requests.get(url, headers=self.headers, timeout=300)
                excel_bytes = resp.content
                self.s3.put_object(Bucket=self.bucket, Key=path_raw, Body=excel_bytes)

            xlsx_io = io.BytesIO(excel_bytes)
            dfs = []
            
            # 2. Varre as abas com Caçador de Cabeçalhos
            for i in range(1, 7):
                try:
                    # Lê a aba bruta sem assumir cabeçalho
                    df_raw = pl.read_excel(xlsx_io, sheet_id=i, engine="calamine", has_header=False)
                    if df_raw.is_empty(): continue
                    
                    # Procura em qual linha estão as colunas reais (LOGRADOURO ou MUNICIPIO)
                    header_row_index = None
                    for idx, row in enumerate(df_raw.iter_rows()):
                        # Converte a linha para uma lista de strings em maiúsculo
                        row_values = [str(v).upper() if v is not None else "" for v in row]
                        if any("LOGRADOURO" in s for s in row_values) or any("MUNICIPIO" in s for s in row_values) or any("CIDADE" in s for s in row_values):
                            header_row_index = idx
                            break
                    
                    if header_row_index is None: continue
                    
                    # Recarrega a aba pulando o lixo até o cabeçalho encontrado
                    df = pl.read_excel(xlsx_io, sheet_id=i, engine="calamine", 
                                       read_options={"skip_rows": header_row_index})
                    
                    df = df.with_columns(pl.all().cast(pl.String))
                    df.columns = [c.upper().replace("Ç", "C").replace("Ã", "A").strip() for c in df.columns]

                    # Mapeamento de sobrevivência
                    mapeamento = {"CIDADE": "MUNICIPIO", "NOME_MUNICIPIO": "MUNICIPIO", "NOME_MUNICIPIO_CIRCUNSCRICAO": "MUNICIPIO"}
                    for original, novo in mapeamento.items():
                        if original in df.columns: df = df.rename({original: novo})
                    
                    if "MUNICIPIO" in df.columns:
                        dfs.append(df)
                        logger.info(f"BRONZE: [{ano}] Aba {i} processada com sucesso (Header na linha {header_row_index}).")
                except: continue

            if dfs:
                df_trusted = pl.concat(dfs, how="diagonal")
                buffer = io.BytesIO()
                df_trusted.write_parquet(buffer)
                self.s3.put_object(Bucket=self.bucket, Key=path_trusted, Body=buffer.getvalue())
                logger.info(f"BRONZE: [{ano}] Camada Trusted Parquet gerada com sucesso.")
                return True
        except Exception as e:
            logger.error(f"BRONZE: Erro no ano {ano}: {e}")
        return False
