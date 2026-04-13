import sys
import os
import time
import logging
import traceback
from datetime import datetime

# =================================================================
# 🛡️ ESCUDO DE PATH (Anti-ModuleNotFoundError)
# Garante que o Python encontre a pasta 'autobot' no ambiente do GitHub
# =================================================================
diretorio_raiz = os.path.dirname(os.path.abspath(__file__))
if diretorio_raiz not in sys.path:
    sys.path.insert(0, diretorio_raiz)

# =================================================================
# 📥 IMPORTS CORRIGIDOS (Conforme sua hierarquia real)
# =================================================================
try:
    # Ajustado para os nomes exatos dos seus arquivos
    from autobot.processamento_prata import ProcessamentoPrata
    from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
    from autobot.comunicador import ComunicadorSafeDriver
    from autobot.treinador_ia import TreinadorEvolutivo 
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    print(f"🔍 Verifique se os nomes dos arquivos em /autobot estão corretos.")
    sys.exit(1)

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class OrquestradorSafeDriver:
    def __init__(self):
        self.comunicador = ComunicadorSafeDriver()
        self.ano_inicial = 2022
        self.ano_atual = datetime.now().year

    def executar_fluxo(self):
        tempo_inicio = time.time()
        anos_para_processar = list(range(self.ano_inicial, self.ano_atual + 1))
        
        logger.info(f"🚀 SAFEDRIVER: Iniciando processamento escalável ({self.ano_inicial} a {self.ano_atual})")

        try:
            # 1. Fase de Treinamento (Opcional: Pode ser movido para fora do loop se desejar)
            logger.info("🧠 Atualizando cérebro da IA com base histórica...")
            treinador = TreinadorEvolutivo()
            treinador.treinar_modelo_mestre()

            # 2. Loop de Processamento por Ano
            for ano in anos_para_processar:
                logger.info(f"--- 🛠️ TRABALHANDO NO ANO: {ano} ---")
                
                # Camada Prata
                prata = ProcessamentoPrata()
                prata.executar_prata(ano)
                
                # Camada Ouro (IA + BigQuery)
                ouro = CamadaOuroSafeDriver()
                ouro.processar_ouro(ano)
                
                logger.info(f"✅ Ano {ano} processado com sucesso.")

            # 3. Notificação de Sucesso
            tempo_total = time.strftime("%M min %S seg", time.gmtime(time.time() - tempo_inicio))
            self.comunicador.relatar_sucesso(
                ano_ref=f"{self.ano_inicial}-{self.ano_atual}",
                tempo_execucao=tempo_total,
                total_linhas="Pipeline Multi-Ano Concluído"
            )
            logger.info(f"🏁 PIPELINE FINALIZADO EM {tempo_total}!")

        except Exception as e:
            logger.error(f"💥 FALHA NO PIPELINE: {e}")
            erro_full = traceback.format_exc()
            self.comunicador.relatar_erro("Orquestrador / Main", erro_full)
            raise e

if __name__ == "__main__":
    safedriver = OrquestradorSafeDriver()
    safedriver.executar_fluxo()
