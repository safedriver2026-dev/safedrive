import sys
import os
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import logging
import argparse
from datetime import datetime
from autobot.ingestao_bronze import IngestaoBronze  # <-- NOVA IMPORTAÇÃO
from autobot.processamento_prata import ProcessamentoPrata
from autobot.treinador_ia import TreinadorEvolutivo
from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
from autobot.calendario_estrategico import CalendarioEstrategico
from autobot.comunicador import ComunicadorSafeDriver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

class SafeDriverMaestro:
    def __init__(self):
        self.inicio = datetime.now()
        self.cal = CalendarioEstrategico()
        self.comunicador = ComunicadorSafeDriver()

    def _dados_essenciais_ausentes(self):
        ano_atual = datetime.now().year
        path_prata = f"safedriver/datalake/prata/ssp_consolidada_{ano_atual}.parquet"
        path_modelo = "safedriver/modelos_ml/latest_cat_motorista.pkl"

        try:
            s3 = boto3.client(
                's3',
                endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
                aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
                aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
                config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
            )
            bucket = os.getenv("R2_BUCKET_NAME", "").strip()

            s3.head_object(Bucket=bucket, Key=path_prata)
            s3.head_object(Bucket=bucket, Key=path_modelo)
            return False
            
        except ClientError as e:
            if 'NoSuchKey' in str(e) or '404' in str(e):
                return True
            else:
                return True 

    def run(self, force=False):
        try:
            logger.info("SafeDriver Maestro: Iniciando rotina de verificação e orquestração.")
            
            # ETAPA 0: Extração Direta da Fonte (SSP)
            # A Ingestão valida o CDC. Se houver dados novos, ela atualiza o R2 e retorna True.
            bronze = IngestaoBronze()
            teve_atualizacao_dados = bronze.executar_ingestao_continua()
            
            # Validação de Regras de Negócio e Integridade
            deve_executar_calendario = self.cal.deve_rodar_hoje()
            dados_ausentes = self._dados_essenciais_ausentes()
            
            # O Pipeline só corre completo se:
            # 1. Houver novos dados extraídos (CDC = True) OU
            # 2. Faltarem dados na infraestrutura OU
            # 3. For uma data de alto risco (Calendário Estratégico)
            deve_executar = teve_atualizacao_dados or deve_executar_calendario or dados_ausentes

            if not deve_executar and not force:
                logger.info("Operação suspensa: Nenhuma atualização na SSP, fora de ciclo estratégico e Datalake íntegro.")
                return

            if teve_atualizacao_dados:
                logger.info("Gatilho acionado: Novos dados detetados na SSP. Reprocessamento em cadeia iniciado.")
            elif dados_ausentes:
                logger.info("Gatilho acionado: Execução mandatória para recuperação de dados em falta.")
            else:
                logger.info("Gatilho acionado: Ciclo estratégico identificado via calendário de risco.")

            # ETAPA 1: Execução da Camada Prata
            logger.info("Etapa 1/3: Iniciando consolidação e cálculo de Deltas (Camada Prata).")
            prata = ProcessamentoPrata()
            prata.executar_todos_os_anos()

            # ETAPA 2: Treinamento de Machine Learning
            logger.info("Etapa 2/3: Iniciando ciclo de treinamento e versionamento de IA.")
            treinador = TreinadorEvolutivo()
            if not treinador.treinar_modelo_mestre():
                raise RuntimeError("Falha no Treinamento da IA. Pipeline interrompido.")

            # ETAPA 3: Inferência Ouro
            logger.info("Etapa 3/3: Processando inferência dinâmica e sincronização com BigQuery.")
            ouro = CamadaOuroSafeDriver()
            if not ouro.executar_predicao_atual():
                raise RuntimeError("Falha na geração da Predição de Risco na Camada Ouro.")

            tempo_total = str(datetime.now() - self.inicio).split(".")[0]
            logger.info(f"Processo finalizado com êxito em {tempo_total}. BigQuery sincronizado.")
            
            self.comunicador.relatar_sucesso(
                datetime.now().year, 
                tempo_total, 
                "Pipeline concluído: SSP (Bronze) -> Prata -> IA -> BigQuery."
            )

        except Exception as e:
            err_msg = f"Erro sistémico no Orquestrador: {str(e)}"
            logger.error(err_msg)
            self.comunicador.relatar_erro("SafeDriver Maestro", err_msg)
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeDriver - Orquestrador de Pipeline de Dados e ML")
    parser.add_argument('--force', action='store_true', help="Força a execução do pipeline ignorando validações.")
    args = parser.parse_args()

    maestro = SafeDriverMaestro()
    maestro.run(force=args.force)
