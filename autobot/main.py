import time
import logging
import traceback
from datetime import datetime
from autobot.camada_prata import ProcessamentoPrata
from autobot.camada_ouro import CamadaOuroSafeDriver
from autobot.comunicador import ComunicadorSafeDriver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class OrquestradorMestre:
    def __init__(self):
        self.comunicador = ComunicadorSafeDriver()
        self.ano_inicial = 2022
        self.ano_atual = datetime.now().year

    def executar_fluxo_escalavel(self):
        start_time = time.time()
        # Gera a lista de 2022 até o ano atual (ex: [2022, 2023, 2024])
        anos_para_processar = list(range(self.ano_inicial, self.ano_atual + 1))
        
        logging.info(f"🚀 Iniciando Processamento Escalável: {anos_para_processar}")

        try:
            for ano in anos_para_processar:
                logging.info(f"--- 🛠️ Processando Safedriver para o Ano: {ano} ---")
                
                # 1. Camada Prata: Refina os dados históricos
                prata = ProcessamentoPrata()
                prata.executar_prata(ano)
                
                # 2. Camada Ouro: Aplica a IA e gera o Score
                # Aqui a Ouro compara Predição vs Realidade se o ano já passou (Backtesting)
                ouro = CamadaOuroSafeDriver()
                ouro.processar_ouro(ano)
                
            # 3. Sucesso Geral
            tempo_total = time.strftime("%M min %S seg", time.gmtime(time.time() - start_time))
            self.comunicador.relatar_sucesso(
                ano_ref=f"{self.ano_inicial}-{self.ano_atual}", 
                tempo_execucao=tempo_total, 
                total_linhas="Escala Multi-Ano Concluída"
            )
            
        except Exception as e:
            logging.error(f"❌ Erro fatal na escala: {e}")
            self.comunicador.relatar_erro("Orquestrador Multi-Ano", traceback.format_exc())
            raise e

if __name__ == "__main__":
    # Comando de execução
    orquestrador = OrquestradorMestre()
    orquestrador.executar_fluxo_escalavel()
