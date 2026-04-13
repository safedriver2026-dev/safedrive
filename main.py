import sys
import os
import time
import logging
import traceback
from datetime import datetime

# =================================================================
# 🛡️ ESCUDO ANTI-ERRO DE IMPORTAÇÃO (ModuleNotFoundError)
# Este bloco garante que o Python encontre a pasta 'autobot' no GitHub
# =================================================================
diretorio_raiz = os.path.dirname(os.path.abspath(__file__))
if diretorio_raiz not in sys.path:
    sys.path.insert(0, diretorio_raiz)

# Agora os imports funcionam com segurança total
try:
    from autobot.camada_prata import ProcessamentoPrata
    from autobot.camada_ouro import CamadaOuroSafeDriver
    from autobot.comunicador import ComunicadorSafeDriver
except ImportError as e:
    print(f"❌ Erro crítico de importação: {e}")
    sys.exit(1)

# Configuração de Logs para Visualização no GitHub Actions
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class OrquestradorSafeDriver:
    def __init__(self):
        self.comunicador = ComunicadorSafeDriver()
        self.ano_inicial = 2022
        self.ano_atual = datetime.now().year

    def executar_pipeline_completo(self):
        """
        Orquestra o fluxo de dados de forma escalável:
        1. Identifica a janela temporal (2022 até hoje)
        2. Refina os dados na Prata (H3 + Geocodificação)
        3. Gera inteligência na Ouro (IA + Backtesting + BigQuery)
        4. Notifica o time via Discord
        """
        tempo_inicio_geral = time.time()
        # Gera lista: [2022, 2023, 2024...]
        anos_para_processar = list(range(self.ano_inicial, self.ano_atual + 1))
        
        logger.info(f"🚀 SAFEDRIVER AUTOBOT: Iniciando processamento de {len(anos_para_processar)} anos.")

        try:
            for ano in anos_para_processar:
                logger.info(f"--- 🛠️ TRABALHANDO NO ANO: {ano} ---")
                
                # --- CAMADA PRATA ---
                logger.info(f"🥈 [PRATA] Refinando dados de {ano}...")
                prata = ProcessamentoPrata()
                prata.executar_prata(ano)
                
                # --- CAMADA OURO ---
                logger.info(f"🥇 [OURO] Aplicando IA e subindo para o BigQuery {ano}...")
                ouro = CamadaOuroSafeDriver()
                ouro.processar_ouro(ano)
                
                logger.info(f"✅ Ano {ano} finalizado com sucesso.")

            # --- FINALIZAÇÃO ---
            tempo_total = time.strftime("%M min %S seg", time.gmtime(time.time() - tempo_inicio_geral))
            
            # Notificação de Sucesso para o canal DISCORD_SUCESSO
            self.comunicador.relatar_sucesso(
                ano_ref=f"{self.ano_inicial}-{self.ano_atual}",
                tempo_execucao=tempo_total,
                total_linhas="Processamento Multi-Ano Concluído"
            )
            
            logger.info(f"🏁 PIPELINE FINALIZADO EM {tempo_total}!")

        except Exception as e:
            logger.error(f"💥 FALHA CRÍTICA NO PIPELINE: {e}")
            
            # Captura o erro técnico completo para o DISCORD_ERRO
            erro_traceback = traceback.format_exc()
            self.comunicador.relatar_erro("Orquestrador / Main Loop", erro_traceback)
            
            # Força o erro para o GitHub Actions marcar o job como falho
            raise e

if __name__ == "__main__":
    # Instancia e roda o orquestrador
    safedriver = OrquestradorSafeDriver()
    safedriver.executar_pipeline_completo()
