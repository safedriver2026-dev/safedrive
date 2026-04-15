import os
import logging
import argparse
from datetime import datetime
from autobot.ingestao_bronze import IngestaoBronze
from autobot.processamento_prata import ProcessamentoPrata
from autobot.treinador_ia import TreinadorEvolutivo
from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver

# Configuração de Logs Profissional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

def orquestrar_pipeline(force_reprocess=False):
    logger.info("🛡️ SafeDriver Maestro: Iniciando orquestração do pipeline de dados.")
    start_time = datetime.now()

    try:
        # --- ETAPA 1: BRONZE (Ingestão e Conversão Trusted) ---
        bronze = IngestaoBronze()
        novos_dados = bronze.executar_ingestao_continua()

        # --- ETAPA 2: PRATA (Consolidação e Geocodificação H3) ---
        # Se houve novos dados ou se o usuário forçou o reprocessamento
        if novos_dados or force_reprocess:
            logger.info("🚀 Gatilho Ativado: Iniciando Processamento Prata.")
            prata = ProcessamentoPrata()
            relatorio_prata = prata.executar_todos_os_anos(force=force_reprocess)
            logger.info(f"✅ Prata concluída. Anos processados: {len(relatorio_prata)}")
            
            # --- ETAPA 3: IA (Treinamento de Modelos) ---
            logger.info("🧠 IA: Iniciando treinamento dos modelos Tweedie.")
            treinador = TreinadorEvolutivo()
            sucesso_ia = treinador.treinar_modelo_mestre()
            
            if sucesso_ia:
                metricas = treinador.obter_metricas_finais()
                logger.info(f"📈 IA Treinada com Sucesso. Métricas: {metricas}")
            
            # --- ETAPA 4: OURO (BigQuery e SHAP) ---
            logger.info("🏆 OURO: Sincronizando Data Warehouse e calculando Explicabilidade.")
            ouro = CamadaOuroSafeDriver()
            resultado_ouro = ouro.executar_predicao_atual()
            
            if resultado_ouro:
                logger.info("✨ Pipeline SafeDriver finalizado com sucesso absoluto.")
        else:
            logger.info("😴 Decisão: Sem dados novos e sem comando de força. Pipeline em repouso.")

    except Exception as e:
        logger.error(f"💥 FALHA NO ORQUESTRADOR: {e}", exc_info=True)
        raise e

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"⏱️ Duração total do ciclo: {duration}")

if __name__ == "__main__":
    # Suporte para a flag --force vinda do GitHub Actions
    parser = argparse.ArgumentParser(description="Maestro do Pipeline SafeDriver")
    parser.add_argument('--force', action='store_true', help='Forçar reprocessamento total')
    args = parser.parse_args()

    orquestrar_pipeline(force_reprocess=args.force)
