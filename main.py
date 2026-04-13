import sys
import argparse
import logging
from datetime import datetime
from autobot.ingestao_raw import IngestaoRaw
from autobot.processamento_prata import ProcessamentoPrata
from autobot.treinador_ia import TreinadorEvolutivo
from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
from autobot.comunicador import ComunicadorSafeDriver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def executar_pipeline():
    parser = argparse.ArgumentParser(description="Orquestrador do Pipeline SafeDriver")
    parser.add_argument("--weekly", action="store_true", help="Executa o ciclo de Extracao e Transformacao (Bronze/Prata)")
    parser.add_argument("--daily", action="store_true", help="Executa o ciclo de Inteligencia Artificial (Treino/Ouro)")
    args = parser.parse_args()

    comunicador = ComunicadorSafeDriver()
    inicio = datetime.now()
    ano_atual = inicio.year

    try:
        if args.weekly:
            logger.info("=== Iniciando Ciclo Semanal: Carga de Dados (Bronze -> Prata) ===")
            raw = IngestaoRaw()
            prata = ProcessamentoPrata()
            
            for ano in range(2022, ano_atual + 1):
                # O pipeline so tenta processar a Prata se a Bronze for bem-sucedida (ou se ja existir)
                if raw.executar_ingestao(ano):
                    prata.executar_prata(ano)
            
            comunicador.relatar_sucesso(ano_atual, str(datetime.now() - inicio), "Ingestao (Bronze) e Limpeza (Prata) concluidas.")

        if args.daily:
            logger.info("=== Iniciando Ciclo Diario: Machine Learning e Analytics (Treino -> Ouro) ===")
            treinador = TreinadorEvolutivo()
            ouro = CamadaOuroSafeDriver()

            # O modelo so avanca para a Ouro (BigQuery) se o retreino for bem-sucedido
            if treinador.treinar_modelo_mestre():
                logger.info("IA: Modelos retreinados com sucesso. Iniciando sincronizacao Delta com BigQuery.")
                for ano in range(2022, ano_atual + 1):
                    ouro.processar_ouro(ano)

                comunicador.relatar_sucesso(ano_atual, str(datetime.now() - inicio), "Sincronizacao Ouro (BigQuery) e Modelos IA atualizados.")
            else:
                logger.warning("Treinamento cancelado ou sem dados novos. A Camada Ouro foi ignorada para proteger o BigQuery.")

        if not args.weekly and not args.daily:
            logger.warning("Nenhum argumento fornecido. Usa --weekly ou --daily.")

    except Exception as e:
        logger.error(f"Falha critica no orquestrador: {e}")
        comunicador.relatar_erro("Orquestrador Geral (main.py)", str(e))
        sys.exit(1)

if __name__ == "__main__":
    executar_pipeline()
