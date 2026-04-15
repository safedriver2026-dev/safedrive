import os
import logging
import argparse
import sys
from datetime import datetime

# Importação dos módulos do diretório autobot
try:
    from autobot.ingestao_bronze import IngestaoBronze
    from autobot.processamento_prata import ProcessamentoPrata
    from autobot.treinador_ia import TreinadorEvolutivo
    from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
except ImportError as e:
    print(f"Erro ao importar módulos do diretório 'autobot': {e}")
    sys.exit(1)

# Configuração de Logs Profissional
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def orquestrar_pipeline(force_reprocess=False):
    """
    Coordena o fluxo de dados do SafeDriver: Bronze -> Prata -> IA -> Ouro.
    """
    logger.info(f"🛡️ SafeDriver Maestro: Iniciando orquestração RÁPIDA (Dev Mode).")
    start_time = datetime.now()

    try:
        # --- ETAPA 1: BRONZE (DESLIGADA PARA TESTES) ---
        logger.info("⚠️ DEV MODE: Ingestão da Camada Bronze IGNORADA para poupar 16 minutos.")
        # bronze = IngestaoBronze()
        # novos_dados_bronze = bronze.executar_ingestao_continua(force=force_reprocess)
        
        # Mágica para enganar o orquestrador e forçar ele a descer o código:
        novos_dados_bronze = True 

        # --- ETAPA 2: PRATA (Geolocalização H3 e Enriquecimento) ---
        if novos_dados_bronze or force_reprocess:
            logger.info("🚀 Gatilho Ativado: Iniciando Processamento Prata (Fórmula 1).")
            prata = ProcessamentoPrata()
            
            # Forçamos a Prata com True para ela rodar a nossa nova engine Lazy/Streaming
            relatorio_prata = prata.executar_todos_os_anos(force=True)
            
            if not relatorio_prata and not force_reprocess:
                logger.warning("⚠️ Prata concluída sem gerar novos registros.")
                # return (Comentado para não travar os testes)

            logger.info(f"✅ Prata finalizada.")
            
            # --- ETAPA 3: IA (Modelagem Preditiva Tweedie) ---
            logger.info("🧠 IA: Iniciando treinamento evolutivo dos modelos CatBoost e LightGBM.")
            
            # Forçamos o dev_mode=True para treinar super rápido só com 2026
            treinador = TreinadorEvolutivo(dev_mode=True)
            sucesso_ia = treinador.treinar_modelo_mestre()
            
            if sucesso_ia:
                metricas = treinador.obter_metricas_finais()
                logger.info(f"📈 IA Treinada com Sucesso. Métricas de Performance: {metricas}")
            else:
                logger.error("❌ Falha no treinamento da IA. Verifique os logs do TreinadorEvolutivo.")
                return
            
            # --- ETAPA 4: OURO (BigQuery, SHAP e Materialização) ---
            logger.info("🏆 OURO: Sincronizando Data Warehouse e processando Explicabilidade (XAI).")
            
            # Forçamos o dev_mode=True aqui também para não explodir o SHAP
            ouro = CamadaOuroSafeDriver(dev_mode=True)
            resultado_ouro = ouro.executar_predicao_atual()
            
            if resultado_ouro:
                logger.info("✨ Pipeline SafeDriver finalizado com sucesso absoluto. Dados prontos para o Looker Studio.")
            else:
                logger.error("❌ Falha na sincronização da Camada Ouro.")
        
        else:
            logger.info("😴 Decisão: Sem atualizações detectadas na SSP-SP e modo force desativado. Pipeline em repouso.")

    except Exception as e:
        logger.error(f"💥 FALHA CRÍTICA NO ORQUESTRADOR: {e}", exc_info=True)
        sys.exit(1)

    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"⏱️ Duração total do ciclo de dados: {duration}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orquestrador Central do SafeDriver Autobot")
    parser.add_argument(
        '--force', 
        action='store_true', 
        default=False,
        help='Ignora trackers de estado e força o reprocessamento de todas as camadas (Backfill)'
    )
    
    args = parser.parse_args()
    orquestrar_pipeline(force_reprocess=args.force)
