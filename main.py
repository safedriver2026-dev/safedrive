import sys
import os
import boto3
import logging
import argparse
from datetime import datetime
from autobot.processamento_prata import ProcessamentoPrata
from autobot.treinador_ia import TreinadorEvolutivo
from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
from autobot.calendario_estrategico import CalendarioEstrategico
from autobot.comunicador import ComunicadorSafeDriver

# Configuração de logging padrão corporativo
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
        """
        Verifica a integridade do Datalake. 
        Retorna True se os dados do ano atual ou os modelos de produção estiverem ausentes.
        """
        ano_atual = datetime.now().year
        path_prata = f"safedriver/datalake/prata/ssp_consolidada_{ano_atual}.parquet"
        path_modelo = "modelos_ml/latest_cat_motorista.pkl"

        try:
            s3 = boto3.client(
                's3',
                endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
                aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
                aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
            )
            bucket = os.getenv("R2_BUCKET_NAME", "").strip()

            # Tenta acessar os metadados dos arquivos críticos
            s3.head_object(Bucket=bucket, Key=path_prata)
            s3.head_object(Bucket=bucket, Key=path_modelo)
            return False  # Os dados essenciais existem
            
        except Exception as e:
            logger.warning("Verificação de integridade: Dados ou modelos essenciais não localizados no storage.")
            return True   # Faltam dados, a execução é obrigatória

    def run(self, force=False):
        """
        Orquestra a execução do pipeline de dados e inferência de Machine Learning.
        Parâmetros:
            force (bool): Ignora validações de calendário e força o processamento.
        """
        try:
            logger.info("SafeDriver Maestro: Iniciando rotina de verificação e orquestração.")
            
            # 1. Validação de Regras de Negócio e Integridade
            deve_executar_calendario = self.cal.deve_rodar_hoje()
            dados_ausentes = self._dados_essenciais_ausentes()
            
            deve_executar = deve_executar_calendario or dados_ausentes

            if not deve_executar and not force:
                logger.info("Operação suspensa: Fora do ciclo estratégico e Datalake atualizado.")
                return

            if dados_ausentes:
                logger.info("Gatilho acionado: Execução mandatória para recuperação de dados faltantes.")
            else:
                logger.info("Gatilho acionado: Ciclo estratégico identificado via calendário.")

            # 2. Execução da Camada Prata
            logger.info("Etapa 1/3: Iniciando consolidação e cálculo de Deltas (Camada Prata).")
            prata = ProcessamentoPrata()
            prata.executar_todos_os_anos()

            # 3. Execução do Treinamento de Machine Learning
            logger.info("Etapa 2/3: Iniciando ciclo de treinamento e versionamento de IA.")
            treinador = TreinadorEvolutivo()
            if not treinador.treinar_modelo_mestre():
                raise RuntimeError("Falha no Treinamento da IA. Pipeline interrompido.")

            # 4. Execução da Camada Ouro (Inferência)
            logger.info("Etapa 3/3: Processando inferência dinâmica e sincronização com BigQuery.")
            ouro = CamadaOuroSafeDriver()
            if not ouro.executar_predicao_atual():
                raise RuntimeError("Falha na geração da Predição de Risco na Camada Ouro.")

            # 5. Conclusão e Registro
            tempo_total = str(datetime.now() - self.inicio).split(".")[0]
            logger.info(f"Processo finalizado com êxito. Tempo de execução: {tempo_total}. BigQuery atualizado.")
            
            self.comunicador.relatar_sucesso(
                datetime.now().year, 
                tempo_total, 
                "Pipeline concluído: Camada Prata, Modelos ML e Inferência Ouro sincronizados."
            )

        except Exception as e:
            err_msg = f"Erro sistêmico no Orquestrador: {str(e)}"
            logger.error(err_msg)
            self.comunicador.relatar_erro("SafeDriver Maestro", err_msg)
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeDriver - Orquestrador de Pipeline de Dados e ML")
    parser.add_argument('--force', action='store_true', help="Força a execução do pipeline ignorando verificações.")
    args = parser.parse_args()

    maestro = SafeDriverMaestro()
    maestro.run(force=args.force)
