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
    from autobot.comunicador import ComunicadorSafeDriver # 🔌 NOVO: Importando o Comunicador
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
    
    # 🔌 NOVO: Instanciando o comunicador
    comunicador = ComunicadorSafeDriver()

    try:
        # --- ETAPA 1: BRONZE (DESLIGADA PARA TESTES) ---
        logger.info("⚠️ DEV MODE: Ingestão da Camada Bronze IGNORADA para poupar 16 minutos.")
        novos_dados_bronze = True 

        # --- ETAPA 2: PRATA (Geolocalização H3 e Enriquecimento) ---
        if novos_dados_bronze or force_reprocess:
            logger.info("🚀 Gatilho Ativado: Iniciando Processamento Prata (Fórmula 1).")
            prata = ProcessamentoPrata()
            
            relatorio_prata = prata.executar_todos_os_anos(force=True)
            
            if not relatorio_prata and not force_reprocess:
                logger.warning("⚠️ Prata concluída sem gerar novos registros.")
                # return

            logger.info(f"✅ Prata finalizada.")
            
            # --- ETAPA 3: IA (Modelagem Preditiva Tweedie) ---
            logger.info("🧠 IA: Iniciando treinamento evolutivo dos modelos CatBoost e LightGBM.")
            
            treinador = TreinadorEvolutivo(dev_mode=True)
            sucesso_ia = treinador.treinar_modelo_mestre()
            
            if sucesso_ia:
                metricas = treinador.obter_metricas_finais()
                logger.info(f"📈 IA Treinada com Sucesso. Métricas: {metricas}")
            else:
                erro_msg = "Falha no treinamento da IA. Verifique os logs do TreinadorEvolutivo."
                logger.error(f"❌ {erro_msg}")
                comunicador.relatar_erro_critico("Camada IA (Treinador)", erro_msg) # 🚨 ALERTA DE ERRO
                return
            
            # --- ETAPA 4: OURO (BigQuery, SHAP e Materialização) ---
            logger.info("🏆 OURO: Sincronizando Data Warehouse e processando Explicabilidade (XAI).")
            
            ouro = CamadaOuroSafeDriver(dev_mode=True)
            resultado_ouro = ouro.executar_predicao_atual()
            
            if resultado_ouro:
                end_time = datetime.now()
                duration = end_time - start_time
                logger.info("✨ Pipeline SafeDriver finalizado com sucesso absoluto.")
                
                # 🎉 ALERTA DE SUCESSO: O pipeline chegou ao fim sem quebrar!
                comunicador.enviar_webhook({
                    "embeds": [{
                        "title": "✅ Pipeline SafeDriver Concluído!",
                        "description": "O Data Warehouse foi atualizado com sucesso. Os modelos foram treinados e o Looker Studio já tem acesso aos dados frescos.",
                        "color": 3066993, # Verde
                        "fields": [
                            {"name": "Camadas Processadas", "value": "Prata, Modelos IA, Ouro (BigQuery)", "inline": False},
                            {"name": "Tempo de Execução", "value": f"`{duration}`", "inline": True}
                        ]
                    }]
                })
                
            else:
                erro_msg = "Falha na sincronização da Camada Ouro com o BigQuery."
                logger.error(f"❌ {erro_msg}")
                comunicador.relatar_erro_critico("Camada Ouro", erro_msg) # 🚨 ALERTA DE ERRO
        
        else:
            logger.info("😴 Decisão: Sem atualizações detectadas. Pipeline em repouso.")

    except Exception as e:
        logger.error(f"💥 FALHA CRÍTICA NO ORQUESTRADOR: {e}", exc_info=True)
        # 🚨 ALERTA DE ERRO CRÍTICO (Quebra inesperada no código)
        comunicador.relatar_erro_critico("Maestro (main.py)", str(e))
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
        help='Ignora trackers de estado e força o reprocessamento de todas as camadas'
    )
    
    args = parser.parse_args()
    orquestrar_pipeline(force_reprocess=args.force)
