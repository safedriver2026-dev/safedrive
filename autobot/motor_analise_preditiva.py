import os
import sys
from datetime import datetime
from .comunicador import RoboComunicador
import autobot.ingestao_raw as ingestao_raw
import autobot.processamento_prata as processamento_prata
import autobot.ia_sincronizacao_ouro as ia_sincronizacao_ouro

def executar_pipeline_bronze_prata_multi_ano():
    robo = RoboComunicador(
        webhook_sucesso=os.environ.get("DISCORD_SUCESSO"),
        webhook_erro=os.environ.get("DISCORD_ERRO")
    )

    anos_para_processar = list(range(2022, datetime.now().year + 1))

    for ano_processar in anos_para_processar:
        try:
            robo.enviar_relatorio_operacional(f"🚀 Iniciando Pipeline SafeDriver Autobot (Bronze e Prata) para o ano {ano_processar}...")

            ingestao_bem_sucedida = ingestao_raw.executar_ingestao(robo, ano=ano_processar)

            if ingestao_bem_sucedida:
                processamento_prata.executar_processamento(robo, ano=ano_processar)
            else:
                robo.enviar_relatorio_operacional(f"✅ Camada Bronze para o ano {ano_processar} não atualizada. Pulando Camada Prata.")

            robo.enviar_relatorio_operacional(f"✅ Pipeline SafeDriver Autobot (Bronze e Prata) para o ano {ano_processar} concluído com sucesso!")

        except Exception as e:
            robo.enviar_alerta_tecnico(f"Pipeline Bronze e Prata (Ano {ano_processar})", str(e))
            sys.exit(1) # Em produção, um erro crítico deve parar o pipeline

def executar_pipeline_ouro_multi_ano():
    robo = RoboComunicador(
        webhook_sucesso=os.environ.get("DISCORD_SUCESSO"),
        webhook_erro=os.environ.get("DISCORD_ERRO")
    )

    anos_para_processar = list(range(2022, datetime.now().year + 1))

    try:
        robo.enviar_relatorio_operacional("🚀 Iniciando Pipeline SafeDriver Autobot (Ouro) para múltiplos anos...")

        ia_sincronizacao_ouro.executar_sincronizacao_ouro(robo, anos=anos_para_processar)

        robo.enviar_relatorio_operacional("✅ Pipeline SafeDriver Autobot (Ouro) concluído com sucesso para múltiplos anos!")

    except Exception as e:
        robo.enviar_alerta_tecnico("Pipeline Ouro (Múltiplos Anos)", str(e))
        sys.exit(1) # Em produção, um erro crítico deve parar o pipeline

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "executar_pipeline_bronze_prata":
            executar_pipeline_bronze_prata_multi_ano()
        elif sys.argv[1] == "executar_pipeline_ouro":
            executar_pipeline_ouro_multi_ano()
    else:
        print("Uso: python motor_analise_preditiva.py [executar_pipeline_bronze_prata|executar_pipeline_ouro]")
        sys.exit(1)
