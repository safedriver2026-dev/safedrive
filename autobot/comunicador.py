import requests
import os
import json
from datetime import datetime

class RoboComunicador:
    def __init__(self):
        self.webhook_sucesso = os.environ.get("DISCORD_SUCESSO")
        self.webhook_erro = os.environ.get("DISCORD_ERRO")

    def enviar_relatorio_operacional(self, resumo, metricas, auditoria_ia=None):
        if not self.webhook_sucesso: return
        descricao = f"**Status Operacional:**\n{resumo}"
        if auditoria_ia:
            descricao += f"\n\n**🧠 Auditoria de Transparência (XAI):**\n*{auditoria_ia}*"

        payload = {
            "embeds": [{
                "title": "🤖 CENTRAL DE INTELIGÊNCIA | SafeDriver Autobot",
                "color": 3066993,
                "description": descricao,
                "fields": [{"name": k, "value": str(v), "inline": True} for k, v in metricas.items()],
                "footer": {"text": "Processamento Automatizado | BigQuery Delta Sync"},
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        try: requests.post(self.webhook_sucesso, json=payload, timeout=10)
        except: pass

    def enviar_alerta_tecnico(self, modulo, stack):
        if not self.webhook_erro: return
        payload = {
            "embeds": [{
                "title": "⚠️ FALHA CRÍTICA NO PIPELINE",
                "color": 15158332,
                "description": f"Erro detectado no módulo: **{modulo}**",
                "fields": [{"name": "Rastreio do Sistema", "value": f"```python\n{stack[:1000]}...\n```"}],
                "footer": {"text": "Processamento Automatizado | BigQuery Delta Sync"},
                "timestamp": datetime.utcnow().isoformat()
            }]
        }
        try: requests.post(self.webhook_erro, json=payload, timeout=10)
        except: pass
