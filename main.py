import sys
import os
import time
import logging
import traceback
from datetime import datetime

# =================================================================
# 🛡️ ESCUDO DE AMBIENTE (Anti-ModuleNotFoundError)
# Força o Python a enxergar a pasta 'autobot' no GitHub Actions
# =================================================================
diretorio_raiz = os.path.dirname(os.path.abspath(__file__))
if diretorio_raiz not in sys.path:
    sys.path.insert(0, diretorio_raiz)

# =================================================================
# 📥 IMPORTS (Nomes exatos da sua hierarquia)
# =================================================================
try:
    from autobot.ingestao_raw import IngestaoRaw
    from autobot.processamento_prata import ProcessamentoPrata
    from autobot.treinador_ia import TreinadorEvolutivo
    from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
    from autobot.comunicador import ComunicadorSafeDriver
except ImportError as e:
    print(f"❌ Erro crítico de hierarquia: {e}")
    sys.exit(1)

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class OrquestradorMestre:
    def __init__(self):
        self.comunicador = ComunicadorSafeDriver()
        self.ano_inicial = 2022
        self.ano_atual = datetime.now().year
        self.anos = list(range(self.ano_inicial, self.ano_atual + 1))

    def executar_fluxo_completo(self):
        """
        Execução em Cascata:
        1. RAW (Bronze) -> Busca na fonte se não existir.
        2. PRATA (Silver) -> Refina e aplica H3.
        3. TREINO (IA) -> Atualiza o modelo com os novos dados.
        4. OURO (Gold) -> Sincroniza Score com BigQuery.
        """
        tempo_inicio = time.time()
        logger.info(f"🚀 INICIANDO PIPELINE SAFEDRIVER ({self.ano_inicial}-{self.ano_atual})")

        try:
            # --- FASE 1: INGESTÃO E REFINO (LOOP DE ANOS) ---
            for ano in self.anos:
                logger.info(f"--- 🛠️ PROCESSANDO ANO: {ano} ---")
                
                # Ingestão Raw (Bronze)
                logger.info(f"🥉 [BRONZE] Verificando dados brutos...")
                raw = IngestaoRaw()
                raw.executar_ingestao(ano) 
                
                # Processamento Prata (Silver)
                logger.info(f"🥈 [PRATA] Transformando e Geocodificando...")
                prata = ProcessamentoPrata()
                prata.executar_prata(ano)

            # --- FASE 2: INTELIGÊNCIA (TREINO) ---
            # O treino acontece após o loop de anos para garantir base histórica completa
            logger.info("🧠 [TREINO] Atualizando modelos de IA com base acumulada...")
            treinador = TreinadorEvolutivo()
            treino_ok = treinador.treinar_modelo_mestre()

            # --- FASE 3: SINCRONIZAÇÃO (OURO) ---
            if treino_ok:
                for ano in self.anos:
                    logger.info(f"🥇 [OURO] Gerando Scores e subindo para BigQuery: {ano}")
                    ouro = CamadaOuroSafeDriver()
                    ouro.processar_ouro(ano)
            else:
                logger.warning("⚠️ Treinamento não gerou novos modelos. Pulando sincronização Ouro.")

            # --- SUCESSO ---
            duracao = time.strftime("%M min %S seg", time.gmtime(time.time() - tempo_inicio))
            self.comunicador.relatar_sucesso(
                ano_ref=f"{self.ano_inicial}-{self.ano_atual}",
                tempo_execucao=duracao,
                total_linhas="Pipeline Completo Finalizado"
            )
            logger.info(f"🏁 PIPELINE CONCLUÍDO COM SUCESSO EM {duracao}")

        except Exception as e:
            logger.error(f"💥 FALHA NO ORQUESTRADOR: {e}")
            erro_trace = traceback.format_exc()
            self.comunicador.relatar_erro("Main / Orquestrador", erro_trace)
            raise e

if __name__ == "__main__":
    autobot = OrquestradorMestre()
    autobot.executar_fluxo_completo()
