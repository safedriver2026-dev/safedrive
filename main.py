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
    parser = argparse.ArgumentParser()
    parser.add_argument("--weekly", action="store_true")
    parser.add_argument("--daily", action="store_true")
    args = parser.parse_args()

    comunicador = ComunicadorSafeDriver()
    inicio = datetime.now()
    ano_atual = inicio.year

    try:
        if args.weekly:
            logger.info("Iniciando Ciclo Semanal: Bronze e Prata")
            raw = IngestaoRaw()
            prata = ProcessamentoPrata()
            
            for ano in range(2022, ano_atual + 1):
                if raw.executar_ingestao(ano):
                    prata.executar_prata(ano)
            
            comunicador.relatar_sucesso(ano_atual, str(datetime.now() - inicio), "Ingestao e Prata concluidas")

        if args.daily:
            logger.info("Iniciando Ciclo Diario: Treinamento e Ouro")
            treinador = TreinadorEvolutivo()
            ouro = CamadaOuroSafeDriver()

            if treinador.treinar_modelo_mestre():
                for ano in range(2022, ano_atual + 1):
                    ouro.processar_ouro(ano)

                comunicador.relatar_sucesso(ano_atual, str(datetime.now() - inicio), "Modelos e BigQuery atualizados")
            else:
                logger.warning("Treinamento cancelado ou sem dados. Ouro ignorada.")

    except Exception as e:
        logger.error(f"Falha critica no pipeline: {e}")
        comunicador.relatar_erro("Orquestrador Geral", str(e))
        sys.exit(1)

if __name__ == "__main__":
    executar_pipeline()
