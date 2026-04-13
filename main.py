import sys
import argparse
import logging
from datetime import datetime
from autobot.ingestao_raw import IngestaoRaw
from autobot.processamento_prata import ProcessamentoPrata
from autobot.treinador_ia import TreinadorEvolutivo
from autobot.ia_sincronizacao_ouro import CamadaOuroSafeDriver
from autobot.comunicador import ComunicadorSafeDriver

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

def executar_pipeline():
    parser = argparse.ArgumentParser(description="Orquestrador SafeDriver Autobot")
    parser.add_argument("--weekly", action="store_true", help="Executa Bronze (Ingestão) e Prata (Limpeza)")
    parser.add_argument("--daily", action="store_true", help="Executa Treino IA e Ouro (BigQuery)")
    args = parser.parse_args()

    comunicador = ComunicadorSafeDriver()
    inicio_geral = datetime.now()
    ano_atual = inicio_geral.year

    try:
        # --- CICLO SEMANAL: DADOS BRUTOS ---
        if args.weekly:
            logger.info("🚀 Iniciando Ciclo Semanal: Bronze e Prata")
            raw = IngestaoRaw()
            prata = ProcessamentoPrata()
            
            for ano in range(2022, ano_atual + 1):
                # Só avança para a Prata se a Ingestão (Bronze) der OK ou o arquivo já existir
                if raw.executar_ingestao(ano):
                    prata.executar_prata(ano)
                else:
                    logger.warning(f"Aviso: Pulando processamento Prata para o ano {ano} devido a falha na Bronze.")
            
            tempo_total = str(datetime.now() - inicio_geral).split(".")[0]
            comunicador.relatar_sucesso(ano_atual, tempo_total, "Dados extraídos da SSP e higienizados na Prata (R2).")

        # --- CICLO DIÁRIO: INTELIGÊNCIA E SCORE ---
        if args.daily:
            logger.info("🧠 Iniciando Ciclo Diário: Treino e Ouro")
            treinador = TreinadorEvolutivo()
            ouro = CamadaOuroSafeDriver()

            # O pipeline só sobe para o BigQuery se o retreino dos modelos for bem-sucedido
            if treinador.treinar_modelo_mestre():
                logger.info("Modelos atualizados. Sincronizando com BigQuery...")
                for ano in range(2022, ano_atual + 1):
                    ouro.processar_ouro(ano)

                tempo_total = str(datetime.now() - inicio_geral).split(".")[0]
                comunicador.relatar_sucesso(ano_atual, tempo_total, "Modelos IA retreinados e BigQuery atualizado via Merge Delta.")
            else:
                logger.warning("Treinamento não gerou novos modelos. Camada Ouro ignorada para segurança.")

        if not args.weekly and not args.daily:
            logger.warning("Nenhum modo de execução selecionado. Use --weekly ou --daily.")

    except Exception as e:
        logger.error(f"Falha Crítica no Pipeline: {e}")
        # Envia o erro direto para o seu canal DISCORD_ERRO
        comunicador.relatar_erro("Orquestrador (main.py)", str(e))
        sys.exit(1)

if __name__ == "__main__":
    executar_pipeline()
