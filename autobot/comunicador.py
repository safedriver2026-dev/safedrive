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
        self.COR_ERRO = 15158332 # Um vermelho vibrante para erros! 🚨

        # O novo rodapé padrão definido em uma variável global para toda a classe
        self.rodape_padrao = "SafeDriver Autobot • Sistema Autônomo de Segurança Preditiva"

    def _disparar(self, url, payload):
        """Método interno para disparar o webhook."""
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status() # Levanta um erro para status de resposta ruins (4xx ou 5xx)
            logger.info(f"Webhook enviado com sucesso para {url}!")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Falha ao enviar webhook para {url}: {e}")
            return False

    def enviar_webhook(self, payload):
        if not self.webhook_sucesso:
            logger.warning("COMUNICADOR: Variável de ambiente 'DISCORD_SUCESSO' ausente. Não foi possível enviar o webhook de sucesso.")
            return False

        if isinstance(payload, dict):
            payload.setdefault("username", "Autobot SafeDriver") # Define o username se não estiver presente

            for embed in payload.get("embeds", []):
                embed.setdefault("timestamp", datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat())
                # Injeta e sobrescreve qualquer rodapé antigo pelo nosso novo padrão
                embed["footer"] = {"text": self.rodape_padrao}
        else:
            logger.error("Payload para enviar_webhook deve ser um dicionário.")
            return False

        return self._disparar(self.webhook_sucesso, payload)

    def relatar_erro_critico(self, modulo, erro):
        if not self.webhook_erro:
            logger.error(f"ERRO CRITICO EM {modulo}: Variável de ambiente 'DISCORD_ERRO' ausente. Não foi possível enviar o webhook de erro.")
            return False

        # Prevenção contra erro vazio (Discord recusa "value" vazio em embeds)
        erro_str = str(erro).strip() if str(erro).strip() else "Erro desconhecido (Traceback vazio)."

        # Formatação em Markdown para ficar bonito no canal do Discord
        # Ajuste na f-string para garantir que o conteúdo seja válido
        embed_erro = {
            "title": f"🚨 Erro Crítico no Módulo: {modulo}",
            "description": "Ocorreu um problema inesperado que requer atenção imediata!",
            "color": self.COR_ERRO,
            "fields": [
                {
                    "name": "Detalhes do Erro",
                    "value": f"```\n{erro_str}\n```", # Usando blocos de código para formatar o erro
                    "inline": False
                }
            ],
            "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
            "footer": {"text": self.rodape_padrao} # Garante o rodapé padrão também nos erros
        }

        payload_erro = {
            "username": "Autobot SafeDriver - Alerta de Erro",
            "embeds": [embed_erro]
        }

        return self._disparar(self.webhook_erro, payload_erro)
