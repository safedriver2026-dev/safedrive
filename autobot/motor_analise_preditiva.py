import sys
import traceback
import importlib
from comunicador import RoboComunicador

bronze = importlib.import_module("0_ingestao_raw")
prata = importlib.import_module("1_processamento_prata")
ouro = importlib.import_module("2_ia_sincronizacao_ouro")

def executar_pipeline_mestre():
    comunicador = RoboComunicador()

    try:
        print("🚀 A iniciar o Pipeline SafeDriver Autobot...")

        print("\n--- PASSO 1: Camada Bronze (Ingestão) ---")
        bronze.executar_ingestao(comunicador)

        print("\n--- PASSO 2: Camada Prata (Fusão Geográfica) ---")
        prata.executar_silver(comunicador)

        print("\n--- PASSO 3: Camada Ouro (Inteligência e Delta Sync) ---")
        ouro.executar_ia_ouro(comunicador)

        print("\n✅ Pipeline finalizado com sucesso. O mapa de risco está atualizado.")

    except Exception as e:
        erro_completo = traceback.format_exc()
        print("\n❌ Falha crítica detetada no pipeline!")
        comunicador.enviar_alerta_tecnico("Orquestrador Mestre (Pipeline Principal)", erro_completo)

        sys.exit(1)

if __name__ == "__main__":
    executar_pipeline_mestre()
