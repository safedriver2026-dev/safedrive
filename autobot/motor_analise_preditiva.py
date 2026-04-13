import os
import sys
from datetime import datetime
import boto3

# Importações Absolutas para garantir funcionamento no GitHub Actions
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

    def verificar_processamento_existente(self, caminho):
        """Verifica se o arquivo já existe no R2 para evitar re-processamento (Delta Sync)"""
        try:
            self.s3.head_object(Bucket=self.bucket, Key=caminho)
            return True
        except:
            return False

    def rodar_ciclo(self, modo="diario"):
        self.robo.enviar_relatorio_operacional(f"🤖 SafeDriver iniciado | Modo: {modo.upper()}")
        
        # DEFINIÇÃO DE ESCOPO: Full (2022 até hoje) ou Diário (apenas ano atual)
        anos_processamento = range(2022, self.ano_atual + 1) if modo == "full" else [self.ano_atual]

        for ano in anos_processamento:
            caminho_bruta = f"datalake/bruta/ssp_{ano}_bronze.parquet"
            caminho_prata = f"datalake/prata/ssp_{ano}_prata.parquet"

            # --- CAMADA BRUTA E PRATA (Processamento Pesado) ---
            # Só roda se for modo 'full' OU se o arquivo ainda não existir (Delta Sync)
            if modo == "full" or not self.verificar_processamento_existente(caminho_prata):
                self.robo.enviar_relatorio_operacional(f"📦 Sincronizando Base Histórica: {ano}")
                
                # Ingestão (Bronze)
                if not self.verificar_processamento_existente(caminho_bruta) or ano == self.ano_atual:
                    bruta = IngestaoBruta(self.robo)
                    bruta.processar_ano(ano)

                # Refino (Prata - Aqui entra o peso 7.5 e as variáveis residenciais)
                prata = ProcessamentoPrata(self.robo)
                prata.executar(ano)

            # --- CAMADA OURO (Predição Diária) ---
            # Roda sempre para garantir que a predição considere o dia/hora atual
            self.robo.enviar_relatorio_operacional(f"🧠 Atualizando Predição de Risco: {ano}")
            ouro = SincronizacaoOuro(self.robo)
            ouro.executar(ano)

        self.robo.enviar_relatorio_operacional(f"🏁 Ciclo {modo} finalizado com sucesso.")

if __name__ == "__main__":
    # Adiciona o diretório atual ao sys.path para evitar erros de módulo
    sys.path.append(os.getcwd())
    
    # Captura o modo via argumento: python -m autobot.motor_analise_preditiva full
    modo_exec = sys.argv[1] if len(sys.argv) > 1 else "diario"
    
    maestro = MotorSafeDriver()
    maestro.rodar_ciclo(modo=modo_exec)
