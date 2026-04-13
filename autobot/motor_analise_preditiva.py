# Conteúdo hipotético de autobot/motor_analise_preditiva.py
import os
import sys
from datetime import datetime
from comunicador import RoboComunicador
import autobot.0_ingestao_raw as ingestao_raw
import autobot.1_processamento_prata as processamento_prata
# import autobot.2_processamento_ouro as processamento_ouro # Para o pipeline ouro

def executar_pipeline_bronze_prata():
    robo = RoboComunicador(
        webhook_sucesso=os.environ.get("DISCORD_SUCESSO"),
        webhook_erro=os.environ.get("DISCORD_ERRO")
    )

    try:
        robo.enviar_relatorio_operacional("🚀 A iniciar o Pipeline SafeDriver Autobot (Bronze e Prata)...")

        print("--- PASSO 1: Camada Bronze (Ingestão) ---")
        # A função executar_ingestao retorna True se a ingestão ocorreu, False se foi pulada
        ingestao_bem_sucedida = ingestao_raw.executar_ingestao(robo)

        if ingestao_bem_sucedida:
            print("--- PASSO 2: Camada Prata (Processamento) ---")
            processamento_prata.executar_processamento(robo)
        else:
            print("✅ Camada Bronze não atualizada. Pulando Camada Prata.")
            # A mensagem de "pulando Camada Prata" já está sendo enviada pelo 0_ingestao_raw.py
            # então não precisamos enviar outro relatório aqui, apenas o print.

        robo.enviar_relatorio_operacional("✅ Pipeline SafeDriver Autobot (Bronze e Prata) concluído com sucesso!")

    except Exception as e:
        robo.enviar_alerta_tecnico("Pipeline Bronze e Prata", str(e))
        print(f"❌ Erro crítico no Pipeline Bronze e Prata: {e}")
        sys.exit(1)

def executar_pipeline_ouro():
    robo = RoboComunicador(
        webhook_sucesso=os.environ.get("DISCORD_SUCESSO"),
        webhook_erro=os.environ.get("DISCORD_ERRO")
    )

    try:
        robo.enviar_relatorio_operacional("🚀 A iniciar o Pipeline SafeDriver Autobot (Ouro)...")



        robo.enviar_relatorio_operacional("✅ Pipeline SafeDriver Autobot (Ouro) concluído com sucesso!")

    except Exception as e:
        robo.enviar_alerta_tecnico("Pipeline Ouro", str(e))
        print(f"❌ Erro crítico no Pipeline Ouro: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "executar_pipeline_bronze_prata":
            executar_pipeline_bronze_prata()
        elif sys.argv[1] == "executar_pipeline_ouro":
            executar_pipeline_ouro()
    else:
        print("Uso: python motor_analise_preditiva.py [executar_pipeline_bronze_prata|executar_pipeline_ouro]")
