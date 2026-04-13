import boto3
import requests
import os
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

    def executar_ingestao(self, ano):
        path_destino = f"datalake/bronze/ssp_raw_{ano}.xlsx"

        try:
            self.s3.head_object(Bucket=self.bucket, Key=path_destino)
            logger.info(f"BRONZE: Dados de {ano} ja existem no R2.")
            return True

        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                logger.info(f"BRONZE: Iniciando download do ano {ano} da SSP-SP.")
                return self._processar_extracao(ano, path_destino)
            else:
                logger.error(f"Erro de conexao com o R2: {e}")
                raise

    def _processar_extracao(self, ano, path_destino):
        url_ssp = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        try:
            logger.info(f"Conectando a: {url_ssp}")
            response = requests.get(url_ssp, headers=headers, timeout=120, stream=True)

            if response.status_code == 404:
                logger.warning(f"SSP: Arquivo de {ano} nao disponivel no portal.")
                return False
            
            response.raise_for_status()

            logger.info(f"Subindo arquivo bruto de {ano} para a Camada Bronze.")
            self.s3.put_object(
                Bucket=self.bucket,
                Key=path_destino,
                Body=response.content,
                ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            logger.info(f"Ano {ano} processado com sucesso.")
            return True

        except Exception as e:
            logger.error(f"Falha na ingestao do ano {ano}: {e}")
            return False

if __name__ == "__main__":
    from datetime import datetime
    ingestor = IngestaoRaw()
    for a in range(2022, datetime.now().year + 1):
        ingestor.executar_ingestao(a)
