import os
import boto3
import requests
import logging
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
        """
        Varre os anos de interesse e realiza o download apenas se houver mudanças (CDC).
        Retorna True se algum arquivo novo foi ingerido.
        """
        logger.info("BRONZE: Iniciando rotina de extração e Change Data Capture (CDC) na SSP-SP.")
        ano_atual = datetime.now().year
        novos_dados_ingeridos = False

       
        for ano in range(2022, ano_atual + 1):
            if self._verificar_e_baixar(ano):
                novos_dados_ingeridos = True

        return novos_dados_ingeridos

    def _verificar_e_baixar(self, ano):
        """
        Lógica interna para comparar o tamanho do arquivo local vs remoto antes de baixar.
        """
        path_bronze = f"safedriver/datalake/bronze/ssp_raw_{ano}.xlsx"
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"

  
        tamanho_r2 = 0
        try:
            meta_r2 = self.s3.head_object(Bucket=self.bucket, Key=path_bronze)
            tamanho_r2 = meta_r2.get('ContentLength', 0)
        except ClientError:
         
            tamanho_r2 = 0

     
        try:
            resp_head = requests.head(url, headers=self.headers, timeout=30)
            
            if resp_head.status_code == 200:
                tamanho_ssp = int(resp_head.headers.get('Content-Length', -1))
                
              
                if tamanho_ssp > 0 and tamanho_ssp == tamanho_r2:
                    logger.info(f"BRONZE: [{ano}] Arquivo em sincronia ({tamanho_ssp} bytes). Ignorando download.")
                    return False
            else:
                logger.warning(f"BRONZE: [{ano}] Servidor SSP indisponível ou arquivo não postado (Status {resp_head.status_code}).")
                return False
                
        except Exception as e:
            logger.error(f"BRONZE: Erro ao validar cabeçalhos na SSP ({ano}): {e}")
            return False

       
        logger.info(f"BRONZE: Mudança detectada para {ano}. Iniciando transferência via stream...")
        try:
            with requests.get(url, headers=self.headers, stream=True, timeout=300) as r:
                r.raise_for_status()
                
      
                self.s3.upload_fileobj(
                    Fileobj=r.raw,
                    Bucket=self.bucket,
                    Key=path_bronze
                )
                
            logger.info(f"BRONZE: Ano {ano} persistido com sucesso na Camada Bronze.")
            return True

        except Exception as e:
            logger.error(f"BRONZE: Falha crítica na transferência do ano {ano}: {e}")
            return False
