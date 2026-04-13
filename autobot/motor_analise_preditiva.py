import sys
import traceback
import importlib
from comunicador import RoboComunicador

bronze = importlib.import_module("0_ingestao_raw")
prata = importlib.import_module("1_processamento_prata")
ouro = importlib.import_module("2_ia_sincronizacao_ouro")

def executar_pipeline_bronze_prata():
    comunicador = RoboComunicador()

    try:
        print("🚀 A iniciar o Pipeline SafeDriver Autobot (Bronze e Prata)...")

        print("\n--- PASSO 1: Camada Bronze (Ingestão) ---")
        houve_atualizacao_bronze = bronze.executar_ingestao(comunicador)

        if houve_atualizacao_bronze:
            print("\n--- PASSO 2: Camada Prata (Fusão Geográfica) ---")
            prata.executar_silver(comunicador)
            print("\n✅ Pipeline Bronze/Prata finalizado com sucesso.")
        else:
            print("\n✅ Camada Bronze não atualizada. Pulando Camada Prata.")

    except Exception as e:
        erro_completo = traceback.format_exc()
        print("\n❌ Falha crítica detetada no pipeline Bronze/Prata!")
        comunicador.enviar_alerta_tecnico("Orquestrador Mestre (Pipeline Bronze/Prata)", erro_completo)
        sys.exit(1)

def executar_pipeline_ouro():
    comunicador = RoboComunicador()

    try:
        print("🚀 A iniciar o Pipeline SafeDriver Autobot (Ouro - Atualização Diária)...")

        print("\n--- PASSO 3: Camada Ouro (Inteligência e Delta Sync) ---")
        ouro.executar_ia_ouro(comunicador)

        print("\n✅ Pipeline Ouro finalizado com sucesso. O mapa de risco está atualizado.")

    except Exception as e:
        erro_completo = traceback.format_exc()
        print("\n❌ Falha crítica detetada no pipeline Ouro!")
        comunicador.enviar_alerta_tecnico("Orquestrador Mestre (Pipeline Ouro)", erro_completo)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "executar_pipeline_bronze_prata":
            executar_pipeline_bronze_prata()
        elif sys.argv[1] == "executar_pipeline_ouro":
            executar_pipeline_ouro()
        else:
            print(f"Argumento desconhecido: {sys.argv[1]}")
            sys.exit(1)
    else:
        print("Nenhum argumento fornecido. Use 'executar_pipeline_bronze_prata' ou 'executar_pipeline_ouro'.")
        sys.exit(1)
