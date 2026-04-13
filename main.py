import sys
import argparse
import logging
from datetime import datetime
from autobot.ingestao_raw import IngestaoRaw
from autobot.processamento_prata import ProcessamentoPrata
from autobot.treinador_ia import TreinadorEvolutivo
from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
from autobot.comunicador import ComunicadorSafeDriver

# Configuração de Log Profissional
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def executar_pipeline():
    parser = argparse.ArgumentParser(description="Orquestrador SafeDriver Autobot v3.0 - MLOps Edition")
    parser.add_argument("--weekly", action="store_true", help="Ciclo de Engenharia: Bronze e Prata")
    parser.add_argument("--daily", action="store_true", help="Ciclo de Inteligência: Treino Evolutivo e Ouro")
    args = parser.parse_args()

    comunicador = ComunicadorSafeDriver()
    inicio_geral = datetime.now()
    ano_atual = inicio_geral.year

    try:
        # --- CICLO SEMANAL: ENGENHARIA DE DADOS ---
        if args.weekly:
            logger.info("🚀 [WEEKLY] Iniciando extração e higienização (Bronze -> Prata)")
            raw = IngestaoRaw()
            prata = ProcessamentoPrata()
            
            for ano in range(2022, ano_atual + 1):
                # A Prata só roda se a Bronze garantir o dado bruto
                if raw.executar_ingestao(ano):
                    prata.executar_prata(ano)
                else:
                    logger.warning(f"Aviso: Bronze falhou para {ano}. Prata abortada para este ano.")
            
            tempo = str(datetime.now() - inicio_geral).split(".")[0]
            comunicador.relatar_sucesso(ano_atual, tempo, "Carga semanal finalizada. Dados prontos no R2.")

        # --- CICLO DIÁRIO: IA E ANALYTICS ---
        if args.daily:
            logger.info("🧠 [DAILY] Iniciando ciclo de MLOps (Treino -> Ouro)")
            treinador = TreinadorEvolutivo()
            ouro = CamadaOuroSafeDriver()

            # 1. Treinador: Gera modelos versionados e meta-features (Memória)
            if treinador.treinar_modelo_mestre():
                logger.info("IA: Novos modelos versionados exportados com sucesso.")
                
                # 2. Ouro: Aplica os modelos (Cold Start ou Evolutivo) e sobe para o BigQuery
                for ano in range(2022, ano_atual + 1):
                    ouro.processar_ouro(ano)

                tempo = str(datetime.now() - inicio_geral).split(".")[0]
                comunicador.relatar_sucesso(
                    ano_atual, 
                    tempo, 
                    "Ensemble 80/20 aplicado com Meta-Features. BigQuery atualizado."
                )
            else:
                logger.error("IA: Falha no treinamento evolutivo. Abortando atualização da Ouro.")
                comunicador.relatar_erro("Treinador IA", "Falha ao gerar modelos evolutivos.")

        if not args.weekly and not args.daily:
            logger.warning("Nenhum argumento detectado. Use --weekly ou --daily para iniciar.")

    except Exception as e:
        logger.error(f"💥 FALHA CRÍTICA: {e}")
        # Notifica o canal DISCORD_ERRO imediatamente
        comunicador.relatar_erro("Orquestrador Geral", str(e))
        sys.exit(1)

if __name__ == "__main__":
    executar_pipeline()
