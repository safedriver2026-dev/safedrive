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
        self.avatar = "https://cdn-icons-png.flaticon.com/512/6062/6062646.png"

    def enviar_webhook(self, payload):
        if not self.webhook_sucesso:
            logger.warning("COMUNICADOR: DISCORD_SUCESSO ausente.")
            return False

        if isinstance(payload, dict):
            payload["username"] = "Autobot SafeDriver"
            payload["avatar_url"] = self.avatar

            for embed in payload.get("embeds", []):
                if "timestamp" not in embed:
                    embed["timestamp"] = datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()

        return self._disparar(self.webhook_sucesso, payload)

    def relatar_erro_critico(self, modulo, erro):
        if not self.webhook_erro:
            logger.error(f"ERRO CRITICO EM {modulo}: {erro}")
            return False

        erro_msg = str(erro)[:1000]
        payload = {
            "username": "Autobot SafeDriver",
            "avatar_url": self.avatar,
            "embeds": [{
                "title": "🚨 SafeDriver - FALHA NO PIPELINE",
                "color": self.COR_ERRO,
                "fields": [
                    {"name": "Módulo", "value": f"`{modulo}`", "inline": True},
                    {"name": "Status", "value": "Interrompido", "inline": True},
                    {"name": "Diagnóstico", "value": f"```{erro_msg}```", "inline": False} # Corrigido aqui!
                ]
            }]
        }
        return self._disparar(self.webhook_erro, payload)

    def _disparar(self, webhook_url, payload):
        try:
            response = requests.post(webhook_url, json=payload, timeout=15)
            response.raise_for_status()
            logger.info(f"Webhook enviado com sucesso para {webhook_url}.")
            return True
        except requests.exceptions.Timeout:
            logger.error(f"Timeout ao enviar webhook para {webhook_url}. O Discord pode estar lento ou indisponível.")
            return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Erro ao enviar webhook para {webhook_url}: {e}")
            return False
