import boto3
import requests
import os
import logging
from botocore.exceptions import ClientError

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class IngestaoRaw:
    def __init__(self):
        # 🛡️ Sanitização de credenciais para evitar erros de header
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        # Conexão com Cloudflare R2
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )

    def executar_ingestao(self, ano):
        """
        Orquestra a verificação e o download do arquivo bruto da SSP.
        """
        # Caminho final na Camada Bronze (armazenamos como .xlsx original)
        path_destino = f"datalake/bronze/ssp_raw_{ano}.xlsx"

        try:
            # 1. Verifica se o ano já foi processado (Idempotência)
            self.s3.head_object(Bucket=self.bucket, Key=path_destino)
            logger.info(f"✅ [BRONZE] Dados de {ano} já estão no Data Lake. Pulando download.")
            return True

        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                logger.info(f"📥 [BRONZE] Iniciando download do ano {ano} diretamente da SSP-SP...")
                return self._processar_extracao(ano, path_destino)
            else:
                logger.error(f"❌ Erro de conexão com o R2: {e}")
                raise

    def _processar_extracao(self, ano, path_destino):
        """
        Realiza o download via stream e sobe para o R2 sem salvar no disco local.
        """
        # URL oficial da SSP-SP (conforme o padrão SPDadosCriminais_ANO.xlsx)
        url_ssp = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        
        # Simulação de navegador para evitar bloqueio por parte do servidor do governo
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        try:
            logger.info(f"🔗 Conectando a: {url_ssp}")
            response = requests.get(url_ssp, headers=headers, timeout=120, stream=True)

            # Tratamento de Delay (Ano atual ainda não publicado)
            if response.status_code == 404:
                logger.warning(f"⚠️ [SSP] O arquivo de {ano} ainda não está disponível no portal (erro 404).")
                return False
            
            response.raise_for_status()

            # Upload direto para o R2 usando o conteúdo em memória
            logger.info(f"📤 Subindo arquivo bruto de {ano} para a Camada Bronze...")
            self.s3.put_object(
                Bucket=self.bucket,
                Key=path_destino,
                Body=response.content,
                ContentType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )

            logger.info(f"✨ Ano {ano} processado com sucesso!")
            return True

        except Exception as e:
            logger.error(f"❌ Falha crítica ao baixar dados da SSP ({ano}): {e}")
            return False

# Gatilho de teste para execução isolada
if __name__ == "__main__":
    from datetime import datetime
    ingestor = IngestaoRaw()
    # Tenta rodar do início (2022) até o ano atual
    for a in range(2022, datetime.now().year + 1):
        ingestor.executar_ingestao(a)
