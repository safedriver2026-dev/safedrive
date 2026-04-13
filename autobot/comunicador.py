import requests
import os
import logging
import platform
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ComunicadorSafeDriver:
    def __init__(self):
        self.webhook_sucesso = os.getenv("DISCORD_WEBHOOK_SUCESSO", "").strip()
        self.webhook_erro = os.getenv("DISCORD_WEBHOOK_ERRO", "").strip()
        self.env = "GitHub Actions" if os.getenv("GITHUB_ACTIONS") else "Local Environment"

    def relatar_sucesso(self, ano_ref, tempo_execucao, total_linhas):
        url = self.webhook_sucesso if self.webhook_sucesso else self.webhook_erro
        if not url:
            return

        payload = {
            "username": "SafeDriver Operations",
            "embeds": [{
                "title": "RELATORIO EXECUTIVO - PIPELINE CONCLUIDO",
                "color": 3066993,
                "author": {
                    "name": "Data Intelligence Unit",
                    "icon_url": "https://cdn-icons-png.flaticon.com/512/2103/2103633.png"
                },
                "fields": [
                    {"name": "PROJETO", "value": "SafeDriver Autobot", "inline": True},
                    {"name": "AMBIENTE", "value": self.env, "inline": True},
                    {"name": "ANO BASE", "value": str(ano_ref), "inline": True},
                    {"name": "PERFORMANCE", "value": f"Tempo total: {tempo_execucao}", "inline": False},
                    {"name": "VOLUMETRIA", "value": f"Registros processados: {total_linhas}", "inline": False},
                    {"name": "INFRAESTRUTURA", "value": f"OS: {platform.system()} | Python: {platform.python_version()}", "inline": False}
                ],
                "footer": {
                    "text": f"ID Execucao: {os.getenv('GITHUB_RUN_ID', 'N/A')} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
            }]
        }

        self._enviar(url, payload)

    def relatar_erro(self, etapa, erro):
        url = self.webhook_erro if self.webhook_erro else self.webhook_sucesso
        if not url:
            return

        payload = {
            "username": "SafeDriver Critical Alerts",
            "embeds": [{
                "title": "ALERTA OPERACIONAL - FALHA NO SISTEMA",
                "color": 15158332,
                "fields": [
                    {"name": "ETAPA CRITICA", "value": etapa, "inline": True},
                    {"name": "SISTEMA", "value": "SafeDriver Autobot", "inline": True},
                    {"name": "AMBIENTE", "value": self.env, "inline": True},
                    {"name": "DIAGNOSTICO TECNICO", "value": f"CODE_ERROR: {str(erro)[:800]}", "inline": False},
                    {"name": "ACAO RECOMENDADA", "value": "Verificar logs do GitHub Actions e integridade da fonte SSP.", "inline": False}
                ],
                "footer": {
                    "text": f"Ocorrencia registrada em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                }
            }]
        }

        self._enviar(url, payload)

    def _enviar(self, url, payload):
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Falha na comunicacao externa: {e}")
