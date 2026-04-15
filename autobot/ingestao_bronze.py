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

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }

    def executar_ingestao_continua(self):
        """Varre os anos de interesse e realiza o CDC e a conversão Trusted."""
        logger.info("BRONZE: Iniciando rotina de extração (Raw) e Refinamento (Trusted).")
        ano_atual = datetime.now().year
        novos_dados_ingeridos = False

        for ano in range(2022, ano_atual + 1):
            if self._verificar_e_baixar(ano):
                novos_dados_ingeridos = True

        return novos_dados_ingeridos

    def _verificar_e_baixar(self, ano):
        # Definindo as duas subcamadas
        path_raw_excel = f"datalake/bronze/raw/ssp_raw_{ano}.xlsx"
        path_trusted_parquet = f"datalake/bronze/trusted/ssp_trusted_{ano}.parquet"
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"

        # 1. CDC (Change Data Capture) via HEAD Request
        tamanho_r2 = 0
        try:
            meta_r2 = self.s3.head_object(Bucket=self.bucket, Key=path_raw_excel)
            tamanho_r2 = meta_r2.get('ContentLength', 0)
        except ClientError:
            tamanho_r2 = 0

        try:
            resp_head = requests.head(url, headers=self.headers, timeout=30)
            if resp_head.status_code == 200:
                tamanho_ssp = int(resp_head.headers.get('Content-Length', -1))
                if tamanho_ssp > 0 and tamanho_ssp == tamanho_r2:
                    logger.info(f"BRONZE: [{ano}] Excel Raw em sincronia. Ignorando download.")
                    return False
            else:
                logger.warning(f"BRONZE: [{ano}] Servidor SSP indisponível (Status {resp_head.status_code}).")
                return False
        except Exception as e:
            logger.error(f"BRONZE: Erro ao validar cabeçalhos na SSP ({ano}): {e}")
            return False

        # 2. Ingestão da Fonte de Verdade (Raw Excel)
        logger.info(f"BRONZE: Atualização detectada para {ano}. Baixando fonte da verdade...")
        try:
            response = requests.get(url, headers=self.headers, timeout=300)
            response.raise_for_status()
            excel_bytes = response.content
            
            # Salva o Excel Original (Garantia de Auditoria)
            logger.info(f"BRONZE: Salvando Excel Original ({ano}) na camada Raw...")
            self.s3.put_object(
                Bucket=self.bucket,
                Key=path_raw_excel,
                Body=excel_bytes
            )

            # 3. Refinamento para a Camada Prata (Trusted Parquet)
            logger.info(f"BRONZE: Convertendo {ano} para formato Colunar (Trusted)...")
            dfs = []
            
            for i in range(1, 6):
                try:
                    df = pl.read_excel(io.BytesIO(excel_bytes), sheet_id=i, engine="calamine")
                    df = df.with_columns(pl.all().cast(pl.String))
                    df.columns = [c.upper().replace("Ç", "C").replace("Ã", "A").strip() for c in df.columns]
                    dfs.append(df)
                except Exception:
                    continue
            
            if dfs:
                df_consolidado = pl.concat(dfs, how="diagonal")
                parquet_buffer = io.BytesIO()
                df_consolidado.write_parquet(parquet_buffer)
                parquet_buffer.seek(0)
                
                logger.info(f"BRONZE: Salvando Parquet Refinado ({ano}) na camada Trusted...")
                self.s3.upload_fileobj(
                    Fileobj=parquet_buffer,
                    Bucket=self.bucket,
                    Key=path_trusted_parquet
                )
            
            logger.info(f"BRONZE: Ciclo completo para {ano} (Raw -> Trusted).")
            return True

        except Exception as e:
            logger.error(f"BRONZE: Falha crítica no processamento do ano {ano}: {e}")
            return False
