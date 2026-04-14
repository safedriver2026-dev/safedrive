import os
import requests
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ComunicadorSafeDriver:
    def __init__(self):
        self.webhook_sucesso = os.getenv("DISCORD_SUCESSO")
        self.webhook_erro = os.getenv("DISCORD_ERRO")
        self.COR_ERRO = 15158332

        # O novo rodapé padrão definido em uma variável global para toda a classe
        self.rodape_padrao = "SafeDriver Autobot • Sistema Autônomo de Segurança Preditiva"

    def enviar_webhook(self, payload):
        if not self.webhook_sucesso:
            logger.warning("COMUNICADOR: DISCORD_SUCESSO ausente.")
            return False

        if isinstance(payload, dict):
            payload["username"] = "Autobot SafeDriver"

            for embed in payload.get("embeds", []):
                if "timestamp" not in embed:
                    embed["timestamp"] = datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()

                # Injeta e sobrescreve qualquer rodapé antigo pelo nosso novo padrão
                embed["footer"] = {"text": self.rodape_padrao}

        return self._disparar(self.webhook_sucesso, payload)

    def relatar_erro_critico(self, modulo, erro):
        if not self.webhook_erro:
            logger.error(f"ERRO CRITICO EM {modulo}: {erro}")
            return False

        erro_msg = str(erro)[:1000]
        payload = {
            "username": "Autobot SafeDriver",
            "embeds": [{
                "title": "🚨 SafeDriver - FALHA NO PIPELINE",
                "color": self.COR_ERRO,
                "fields": [
                    {"name": "Módulo", "value": f"`{modulo}`", "inline": True},
                    {"name": "Status", "value": "Interrompido", "inline": True},
                    {"name": "Diagnóstico", "value": erro_msg} # AQUI ESTAVA O ERRO!
                ],
                "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
                "footer": {"text": self.rodape_padrao}
            }]
        }
        return self._disparar(self.webhook_erro, payload)

    def _disparar(self, webhook_url, payload):
        try:
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"Webhook enviado com sucesso para {webhook_url}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao enviar webhook para {webhook_url}: {e}")
            return False
