import boto3
import requests
import os
import io
import logging
from botocore.exceptions import ClientError

# Configuração de Logs para rastreamento no GitHub Actions
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class IngestaoRaw:
    def __init__(self):
        # 🛡️ SANITIZAÇÃO TOTAL: .strip() evita erros de conexão por espaços nos segredos
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        # Conexão com Cloudflare R2 usando a API S3
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )

    def executar_ingestao(self, ano):
        """
        Ponto de entrada para qualquer ano (2022, 2023, 2024...).
        Verifica se o dado bruto já existe no R2 para economizar processamento.
        """
        # Nome padronizado para a Camada Bronze
        path_destino = f"datalake/bronze/ssp_raw_{ano}.csv"

        try:
            # 1. Check de Idempotência: Se o arquivo já existe, não baixa de novo
            self.s3.head_object(Bucket=self.bucket, Key=path_destino)
            logger.info(f"✅ [BRONZE] O ano {ano} já está populado. Pulando para a próxima etapa.")
            return True

        except ClientError as e:
            # Erro 404 significa que o arquivo não existe e precisa ser baixado
            if e.response['Error']['Code'] == "404":
                logger.info(f"📥 [BRONZE] Dados de {ano} ausentes. Iniciando extração remota...")
                return self._processar_download_e_upload(ano, path_destino)
            else:
                logger.error(f"❌ Erro de infraestrutura ao acessar o R2: {e}")
                raise

    def _processar_download_e_upload(self, ano, path_destino):
        """
        Realiza a captura do dado da fonte e armazena na Bronze de forma imutável.
        """
        # A URL deve ser dinâmica baseada no ano para ser completa
        url_fonte = f"https://sua-fonte-de-dados.com/ssp/ocorrencias_{ano}.csv"
        
        try:
            # 1. Captura do dado via Streaming (eficiente para arquivos grandes)
            logger.info(f"🔗 Conectando à fonte SSP para o ano {ano}...")
            response = requests.get(url_fonte, timeout=60, stream=True)
            response.raise_for_status() 
            
            # Usamos BytesIO para manter o arquivo em memória e evitar escrita em disco no GitHub
            conteudo_buffer = io.BytesIO(response.content)
            
            # 2. Upload para a Camada Bronze
            logger.info(f"📤 Transferindo dados brutos de {ano} para o Data Lake R2...")
            self.s3.upload_fileobj(
                conteudo_buffer,
                self.bucket,
                path_destino,
                ExtraArgs={'ContentType': 'text/csv'}
            )
            
            logger.info(f"✨ Camada Bronze atualizada: {path_destino}")
            return True

        except Exception as e:
            logger.error(f"❌ Falha crítica ao processar a ingestão do ano {ano}: {e}")
            # Se a extração falhar, o pipeline deve ser interrompido para evitar dados incompletos
            raise e

# ==========================================
# GATILHO DE TESTE (Loop Automático)
# ==========================================
if __name__ == "__main__":
    # Para testes manuais rápidos, ele tenta processar o range completo
    from datetime import datetime
    ingestor = IngestaoRaw()
    for a in range(2022, datetime.now().year + 1):
        ingestor.executar_ingestao(a)
