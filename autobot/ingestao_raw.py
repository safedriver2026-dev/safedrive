import os
import boto3
import requests
import logging
from botocore.config import Config
from botocore.exceptions import ClientError
from datetime import datetime

# Configuração de logging padrão corporativo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class IngestaoBronze:
    def __init__(self):
        # Definições de ambiente para o Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        # Configuração de conectividade blindada
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

    def executar_ingestao_continua(self):
        """Itera sobre a base histórica e retorna True se detetar novos dados (CDC)."""
        logger.info("BRONZE: Iniciando rotina de extração e Change Data Capture na SSP-SP.")
        ano_atual = datetime.now().year
        novos_dados_ingeridos = False

        for ano in range(2022, ano_atual + 1):
            if self._verificar_e_baixar(ano):
                novos_dados_ingeridos = True

        return novos_dados_ingeridos

    def _verificar_e_baixar(self, ano):
        path_bronze = f"safedriver/datalake/bronze/ssp_raw_{ano}.xlsx"
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"

        # 1. Verifica o tamanho do ficheiro atual no Data Lake (R2)
        tamanho_r2 = 0
        try:
            meta_r2 = self.s3.head_object(Bucket=self.bucket, Key=path_bronze)
            tamanho_r2 = meta_r2.get('ContentLength', 0)
        except ClientError:
            pass # Ficheiro não existe, tamanho_r2 mantém-se 0

        # 2. Verifica o tamanho do ficheiro na Origem (SSP-SP) via HTTP HEAD (sem baixar o ficheiro)
        try:
            resp_head = requests.head(url, headers=self.headers, timeout=30)
            if resp_head.status_code == 200:
                tamanho_ssp = int(resp_head.headers.get('Content-Length', -1))
                
                # Regra de CDC: Se os tamanhos forem idênticos, os dados não mudaram. Bypass.
                if tamanho_ssp > 0 and tamanho_ssp == tamanho_r2:
                    logger.info(f"BRONZE: [{ano}] Sem atualizações na SSP. Ficheiro em sincronia ({tamanho_ssp} bytes). Bypass ativado.")
                    return False
            else:
                logger.warning(f"BRONZE: [{ano}] Servidor SSP retornou status {resp_head.status_code} na verificação.")
                return False
                
        except Exception as e:
            logger.error(f"BRONZE: Erro ao validar cabeçalhos HTTP na SSP ({ano}): {e}")
            return False

        # 3. Execução do Download Pesado (Apenas se o CDC aprovar)
        logger.info(f"BRONZE: Atualização detetada para {ano}. Iniciando extração do ficheiro da SSP-SP...")
        try:
            response = requests.get(url, headers=self.headers, timeout=300)
            
            if response.status_code == 200:
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=path_bronze,
                    Body=response.content
                )
                logger.info(f"BRONZE: Ano {ano} transferido e persistido com sucesso na Camada Bronze.")
                return True
            else:
                logger.error(f"Falha na extração de {ano}. HTTP Status: {response.status_code}")
                return False

        except requests.exceptions.Timeout:
            logger.error(f"Timeout (300s) na ingestão do ano {ano}. Servidor da SSP instável.")
            return False
        except Exception as e:
            logger.error(f"Erro inesperado na persistência da Camada Bronze ({ano}): {e}")
            return False

if __name__ == "__main__":
    IngestaoBronze().executar_ingestao_continua()
