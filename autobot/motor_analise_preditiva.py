import os
import sys
import boto3
from datetime import datetime


from autobot.ingestao_bruta import IngestaoBruta
from autobot.processamento_prata import ProcessamentoPrata
from autobot.sincronizacao_ouro import SincronizacaoOuro
from autobot.comunicador import RoboComunicador

class MotorSafeDriver:
    def __init__(self):
        self.robo = RoboComunicador()
        self.s3 = boto3.client('s3',
                                endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                                aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                                aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))
        self.bucket = os.environ.get("R2_BUCKET_NAME")
        self.ano_atual = datetime.now().year

    def arquivo_existe_no_r2(self, caminho):
        """Verifica se o arquivo já foi processado para garantir o Delta Sync na Bronze/Prata"""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=caminho)
            return True
        except:
            return False

    def fluxo_delta(self, modo="predicao"):
        self.robo.enviar_relatorio_operacional(f"🚀 Iniciando SafeDriver - Modo: {modo}")

        for ano in range(2022, self.ano_atual + 1):
            caminho_bruta = f"datalake/bruta/ssp_{ano}_bronze.parquet"
            caminho_prata = f"datalake/prata/ssp_{ano}_prata.parquet"

            # DELTA SYNC: Só roda a ingestão pesada se o arquivo não existir ou se for o ano atual
            if modo == "full" or ano == self.ano_atual:
                if not self.arquivo_existe_no_r2(caminho_bruta):
                    ingestor = IngestaoBruta(self.robo)
                    ingestor.processar_ano(ano)
                
                if not self.arquivo_existe_no_r2(caminho_prata):
                    processador = ProcessamentoPrata(self.robo)
                    processador.executar(ano)

            # A camada OURO roda sempre (Diário), mas com lógica de arredondamento interna
            sincronizador = SincronizacaoOuro(self.robo)
            sincronizador.executar(ano)

        self.robo.enviar_relatorio_operacional("🏁 Ciclo Delta Sync Finalizado")

if __name__ == "__main__":
    # Garante que o diretório atual está no path para execução via módulo
    sys.path.append(os.getcwd())
    
    modo_selecionado = sys.argv[1] if len(sys.argv) > 1 else "predicao"
    maestro = MotorSafeDriver()
    maestro.fluxo_delta(modo=modo_selecionado)
