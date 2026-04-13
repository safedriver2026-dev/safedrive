import time
import logging
import traceback
from autobot.camada_prata import ProcessamentoPrata
from autobot.camada_ouro import CamadaOuroSafeDriver
from autobot.comunicador import ComunicadorSafeDriver

# Configuração de logs para visualização no GitHub Actions
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def rodar_pipeline_completo(ano=2024):
    inicio = time.time()
    comunicador = ComunicadorSafeDriver()
    
    try:
        logging.info("🔥 Iniciando Pipeline SafeDriver...")
        
        # Etapa 1: Refinação Prata
        prata = ProcessamentoPrata()
        prata.executar_prata(ano)
        
        # Etapa 2: Inteligência Ouro
        ouro = CamadaOuroSafeDriver()
        ouro.processar_ouro(ano)
        
        # Finalização e Notificação de Sucesso
        tempo_total = time.strftime("%M min %S seg", time.gmtime(time.time() - inicio))
        comunicador.relatar_sucesso(ano, tempo_total, 1500000) # Exemplo de volumetria
        
        logging.info("🏁 Pipeline concluído com sucesso!")
        
    except Exception as e:
        logging.error(f"❌ Falha crítica no pipeline: {e}")
        # Envia o erro técnico para o canal DISCORD_ERRO
        comunicador.relatar_erro("Orquestrador / Main", traceback.format_exc())
        raise e

if __name__ == "__main__":
    rodar_pipeline_completo(2024)
