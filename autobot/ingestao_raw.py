import os
import boto3
import requests
import logging
from botocore.exceptions import ClientError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class IngestaoRaw:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def executar_ingestao(self, ano):
        path_bronze = f"datalake/bronze/ssp_raw_{ano}.xlsx"
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"

        try:
            try:
                self.s3.head_object(Bucket=self.bucket, Key=path_bronze)
                logger.info(f"BRONZE: Dados de {ano} ja existem no R2.")
                return True
            except ClientError as e:
                if e.response['Error']['Code'] != '404':
                    raise e

            logger.info(f"BRONZE: Iniciando download do ano {ano} da SSP-SP.")
            logger.info(f"Conectando a: {url}")
            
            response = requests.get(url, headers=self.headers, timeout=300)
            
            if response.status_code == 200:
                logger.info(f"Subindo ficheiro bruto de {ano} para a Camada Bronze.")
                
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=path_bronze,
                    Body=response.content
                )
                logger.info(f"BRONZE: Ano {ano} processado com sucesso.")
                return True
            else:
                logger.error(f"Falha ao transferir {ano}. HTTP Status: {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            logger.error(f"Falha na ingestao do ano {ano}: Tempo de limite excedido (300s). O servidor da SSP esta demasiado lento.")
            return False
        except requests.exceptions.ConnectionError:
            logger.error(f"Falha na ligacao para o ano {ano}: O servidor recusou ou a rede falhou.")
            return False
        except Exception as e:
            logger.error(f"Erro inesperado na Camada Bronze ({ano}): {e}")
            return False

if __name__ == "__main__":
    from datetime import datetime
    raw = IngestaoRaw()
    ano_atual = datetime.now().year
    for a in range(2022, ano_atual + 1):
        raw.executar_ingestao(a)
