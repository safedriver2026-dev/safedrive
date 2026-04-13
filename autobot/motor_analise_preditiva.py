import os
from datetime import datetime
from .ingestao_bruta import IngestaoBruta
from .processamento_prata import ProcessamentoPrata
from .sincronizacao_ouro import SincronizacaoOuro
from .comunicador import RoboComunicador

class MotorAnalisePreditiva:
    def __init__(self):
        self.robo = RoboComunicador()
        self.ano_inicial = 2022
        self.ano_atual = datetime.now().year

    def executar_fluxo_completo(self):
        try:
            self.robo.enviar_relatorio_operacional("🚀 Iniciando Maestro SafeDriver")

            for ano in range(self.ano_inicial, self.ano_atual + 1):
                
                bruta = IngestaoBruta(self.robo)
                sucesso_bruta = bruta.processar_ano(ano)
                
                if not sucesso_bruta:
                    continue

                prata = ProcessamentoPrata(self.robo)
                sucesso_prata = prata.executar(ano)
                
                if not sucesso_prata:
                    continue

                ouro = SincronizacaoOuro(self.robo)
                ouro.executar(ano)

            self.robo.enviar_relatorio_operacional("🏁 Ciclo SafeDriver Finalizado com Sucesso")

        except Exception as e:
            import traceback
            self.robo.enviar_alerta_tecnico("Falha Critica no Maestro", traceback.format_exc())

if __name__ == "__main__":
    maestro = MotorAnalisePreditiva()
    maestro.executar_fluxo_completo()
