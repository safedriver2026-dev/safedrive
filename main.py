import sys
import logging
import argparse
from datetime import datetime
from autobot.processamento_prata import ProcessamentoPrata
from autobot.treinador_ia import TreinadorEvolutivo
from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
from autobot.calendario_estrategico import CalendarioEstrategico
from autobot.comunicador import ComunicadorSafeDriver

# Configuração de logging profissional
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

    def run(self, force=False):
        """
        Orquestra o pipeline. 
        Se 'force' for True, ignora o calendário e executa tudo.
        """
        try:
            logger.info("🤖 SafeDriver Autobot: Iniciando verificação de rotina...")
            
            # 1. Verificação de Gatilho Estratégico
            deve_executar = self.cal.deve_rodar_hoje()
            
            if not deve_executar and not force:
                logger.info("💤 Fora de ciclo estratégico. Encerrando operação sem custos.")
                return

            logger.info("🚀 GATILHO ATIVADO: Iniciando ciclo completo de Predição de Risco.")

            # 2. Camada Prata (Consolidação Histórica e Deltas)
            # Varre todos os anos de 2022 até hoje para garantir tendências precisas
            logger.info("🔄 Passo 1/3: Sincronizando Camada Prata e Deltas...")
            prata = ProcessamentoPrata()
            prata.executar_todos_os_anos()

            # 3. Treinador IA (Evolução de Modelos e MLOps)
            # Injeta Meta-Features de erro e aprende com os Deltas gerados na Prata
            logger.info("🧠 Passo 2/3: Iniciando Treinamento Evolutivo (Ensemble)...")
            treinador = TreinadorEvolutivo()
            if not treinador.treinar_modelo_mestre():
                raise Exception("Falha crítica no Treinamento da IA.")

            # 4. Camada Ouro (Predição Dinâmica com Pesos de Contexto)
            # Aplica os multiplicadores de Semana de Pagamento e Feriados no Score final
            logger.info("🏆 Passo 3/3: Gerando Predição de Risco Atualizada...")
            ouro = CamadaOuroSafeDriver()
            if not ouro.executar_predicao_atual():
                raise Exception("Falha na geração da Predição na Camada Ouro.")

            # 5. Finalização e Relatório
            tempo_total = str(datetime.now() - self.inicio).split(".")[0]
            msg_sucesso = f"Pipeline finalizado em {tempo_total}. Predição estratégica disponível no BigQuery."
            logger.info(f"✅ {msg_sucesso}")
            
            self.comunicador.relatar_sucesso(
                datetime.now().year, 
                tempo_total, 
                "Ciclo Estratégico (Deltas + IA + Calendário)"
            )

        except Exception as e:
            err_msg = f"Erro no Maestro: {str(e)}"
            logger.error(f"💥 {err_msg}")
            self.comunicador.relatar_erro("Main Maestro", err_msg)
            sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeDriver Autobot - Orquestrador de IA")
    parser.add_argument('--force', action='store_true', help="Força a execução ignorando o calendário")
    args = parser.parse_args()

    maestro = SafeDriverMaestro()
    maestro.run(force=args.force)
