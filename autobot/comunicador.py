import os
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ComunicadorSafeDriver:
    def __init__(self):
        # Lendo as variaveis EXATAS que voce configurou no GitHub
        self.webhook_sucesso = os.getenv("DISCORD_SUCESSO")
        self.webhook_erro = os.getenv("DISCORD_ERRO")

    def relatar_sucesso(self, ano, tempo, msg):
        if not self.webhook_sucesso:
            logger.warning("COMUNICADOR: Webhook DISCORD_SUCESSO nao encontrado nas variaveis de ambiente.")
            return
        
        payload = {
            "embeds": [{
                "title": "✅ SafeDriver - Pipeline concluído!",
                "color": 65280, # Verde
                "fields": [
                    {"name": "Ano Processado", "value": str(ano), "inline": True},
                    {"name": "Tempo de Execução", "value": str(tempo), "inline": True},
                    {"name": "Status", "value": msg}
                ],
                "footer": {"text": "Autobot SafeDriver v3.0"}
            }]
        }
        
        try:
            requests.post(self.webhook_sucesso, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Erro ao enviar notificacao de sucesso: {e}")

    def relatar_erro(self, modulo, erro):
        if not self.webhook_erro:
            logger.warning("COMUNICADOR: Webhook DISCORD_ERRO nao encontrado nas variaveis de ambiente.")
            return

        payload = {
            "embeds": [{
                "title": "🚨 SafeDriver - FALHA NO PIPELINE",
                "color": 16711680, # Vermelho
                "fields": [
                    {"name": "Módulo", "value": modulo, "inline": True},
                    {"name": "Erro Detalhado", "value": f"```{erro}```"}
                ],
                "footer": {"text": "Atenção necessária imediatamente!"}
            }]
        }
        
        try:
            requests.post(self.webhook_erro, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Erro ao enviar notificacao de erro: {e}")

if __name__ == "__main__":
    # Teste rapido
    com = ComunicadorSafeDriver()
    print(f"Webhook Sucesso Configurado: {bool(com.webhook_sucesso)}")
    print(f"Webhook Erro Configurado: {bool(com.webhook_erro)}")
