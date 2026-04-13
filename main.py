import sys
import os
import boto3
import json
import logging
import argparse
from datetime import datetime
from botocore.config import Config
from botocore.exceptions import ClientError
from google.cloud import bigquery
from google.oauth2 import service_account

# Importação dos módulos do ecossistema SafeDriver
from autobot.ingestao_bronze import IngestaoBronze
from autobot.processamento_prata import ProcessamentoPrata
from autobot.treinador_ia import TreinadorEvolutivo
from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
from autobot.calendario_estrategico import CalendarioEstrategico
from autobot.comunicador import ComunicadorSafeDriver

# Configuração de Logging de Graus de Produção
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class SafeDriverMaestro:
    def __init__(self):
        self.inicio = datetime.now()
        self.cal = CalendarioEstrategico()
        self.comunicador = ComunicadorSafeDriver()

    def _dados_essenciais_ausentes(self):
        """Verifica a integridade do Datalake no R2 e a existência da tabela no BigQuery."""
        ano_atual = datetime.now().year
        path_prata = f"safedriver/datalake/prata/ssp_consolidada_{ano_atual}.parquet"
        path_modelo = "safedriver/modelos_ml/latest_cat_motorista.pkl"
        
        # 1. Validação de Infraestrutura R2 (Cloudflare)
        try:
            s3 = boto3.client(
                's3',
                endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
                aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
                aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
                config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
            )
            bucket = os.getenv("R2_BUCKET_NAME", "").strip()
            
            # Tenta ler os metadados dos arquivos vitais
            s3.head_object(Bucket=bucket, Key=path_prata)
            s3.head_object(Bucket=bucket, Key=path_modelo)
            logger.info("INTEGRIDADE: Arquivos detectados no R2.")
        except Exception:
            logger.warning("INTEGRIDADE: Arquivos essenciais ausentes no R2 (Prata ou Modelos).")
            return True 

        # 2. Validação de Infraestrutura BigQuery (Google Cloud)
        try:
            gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
            cred_info = json.loads(gcp_json)
            credentials = service_account.Credentials.from_service_account_info(cred_info)
            client = bigquery.Client(credentials=credentials, project=os.getenv("BQ_PROJECT_ID"))
            
            tabela_ref = f"{os.getenv('BQ_PROJECT_ID')}.{os.getenv('BQ_DATASET_ID')}.fato_risco_predicao_atual"
            client.get_table(tabela_ref)
            logger.info("INTEGRIDADE: Tabela de produção localizada no BigQuery.")
        except Exception as e:
            logger.warning(f"INTEGRIDADE: Tabela no BigQuery não localizada ou inacessível: {e}")
            return True # Gatilho acionado para forçar a criação da tabela na Ouro

        return False # Sistema 100% íntegro

    def run(self, force=False):
        try:
            logger.info("SafeDriver Maestro: Iniciando rotina de verificação e orquestração.")
            
            # ETAPA 0: Ingestão e CDC (Change Data Capture)
            # Verifica se a SSP atualizou os arquivos Excel
            bronze = IngestaoBronze()
            teve_atualizacao_dados = bronze.executar_ingestao_continua()
            
            # Verificação de gatilhos estratégicos e integridade
            deve_executar_calendario = self.cal.deve_rodar_hoje()
            dados_ausentes = self._dados_essenciais_ausentes()
            
            # Lógica de Decisão do Orquestrador
            deve_executar = teve_atualizacao_dados or deve_executar_calendario or dados_ausentes

            if not deve_executar and not force:
                logger.info("Operação suspensa: Ambiente íntegro, sem novos dados e fora de ciclo estratégico.")
                return

            # Log de Motivação da Execução
            if force: logger.info("Gatilho: EXECUÇÃO FORÇADA via parâmetro.")
            elif teve_atualizacao_dados: logger.info("Gatilho: NOVOS DADOS detectados na SSP-SP.")
            elif dados_ausentes: logger.info("Gatilho: RECUPERAÇÃO DE INFRAESTRUTURA (Dados ou Tabelas ausentes).")
            else: logger.info("Gatilho: CICLO ESTRATÉGICO (Calendário de Risco).")

            # ETAPA 1: Camada Prata (Transformação e H3)
            logger.info("Etapa 1/3: Iniciando processamento da Camada Prata.")
            prata = ProcessamentoPrata()
            prata.executar_todos_os_anos()

            # ETAPA 2: IA (Treinamento Evolutivo)
            logger.info("Etapa 2/3: Iniciando treinamento dos modelos de Machine Learning.")
            treinador = TreinadorEvolutivo()
            if not treinador.treinar_modelo_mestre():
                raise RuntimeError("Falha crítica no treinamento da IA.")

            # ETAPA 3: Camada Ouro (Inferência e BigQuery)
            logger.info("Etapa 3/3: Gerando predições e sincronizando com BigQuery.")
            ouro = CamadaOuroSafeDriver()
            if not ouro.executar_predicao_atual():
                raise RuntimeError("Falha na sincronização final com o BigQuery.")

            # Finalização
            tempo_total = str(datetime.now() - self.inicio).split(".")[0]
            logger.info(f"✅ SafeDriver Autobot finalizado com sucesso em {tempo_total}.")
            
            self.comunicador.relatar_sucesso(
                datetime.now().year, 
                tempo_total, 
                "Pipeline completo sincronizado com sucesso no BigQuery."
            )

        except Exception as e:
            err_msg = f"Erro sistémico no Orquestrador: {str(e)}"
            logger.error(err_msg)
            self.comunicador.relatar_erro("SafeDriver Maestro", err_msg)
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Maestro - Orquestrador SafeDriver")
    parser.add_argument('--force', action='store_true', help="Força a execução total do pipeline.")
    args = parser.parse_args()

    SafeDriverMaestro().run(force=args.force)
