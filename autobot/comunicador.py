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
        
       
        self.COR_SUCESSO = 3066993 
        self.COR_ERRO = 15158332    

    def relatar_conclusao_rica(self, contexto, metricas_executivas, metricas_operacionais):
        """
        Gera o painel de Sucesso Diário com layout de colunas (Inline).
        """
        if not self.webhook_sucesso:
            logger.warning("COMUNICADOR: Webhook DISCORD_SUCESSO não encontrado nas variáveis de ambiente.")
            return

        fields = []

        # --- Bloco Executivo ---
        fields.append({
            "name": "📊 VISÃO EXECUTIVA", 
            "value": "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
            "inline": False
        })
        for chave, valor in metricas_executivas.items():
            fields.append({"name": chave, "value": str(valor), "inline": True})

        # --- Bloco Operacional ---
        fields.append({
            "name": "⚙️ VISÃO OPERACIONAL", 
            "value": "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
            "inline": False
        })
        for chave, valor in metricas_operacionais.items():
            fields.append({"name": chave, "value": str(valor), "inline": True})

        # Montagem do Payload do Discord
        payload = {
            "username": "Autobot SafeDriver",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/6062/6062646.png",
            "embeds": [{
                "title": "✅ SafeDriver - Relatório de Consolidação Diária",
                "description": f"**Gatilho da Execução:** {contexto.get('gatilho', 'N/A')}\n**Tempo Total:** {contexto.get('tempo_total', 'N/A')}",
                "color": self.COR_SUCESSO,
                "fields": fields,
                "footer": {
                    "text": "Autobot SafeDriver - Data Engineering Team"
                },
                "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()
            }]
        }
        
        self._enviar_payload(self.webhook_sucesso, payload)

    def relatar_alerta_critico(self, modulo, status, detalhe_erro):
        """
        Gera o painel de Alerta Crítico com bloco de código.
        """
        if not self.webhook_erro:
            logger.warning("COMUNICADOR: Webhook DISCORD_ERRO não encontrado nas variáveis de ambiente.")
            return

        payload = {
            "username": "Autobot SafeDriver",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/6062/6062646.png",
            "embeds": [{
                "title": "🚨 SafeDriver - FALHA CRÍTICA NO PIPELINE",
                "color": self.COR_ERRO,
                "fields": [
                    {"name": "Módulo de Falha", "value": modulo, "inline": True},
                    {"name": "Status", "value": status, "inline": True},
                    {
                        "name": "Detalhes do Diagnóstico", 
                        "value": f"```log\n{detalhe_erro}\n```", 
                        "inline": False
                    }
                ],
                "footer": {
                    "text": "Autobot SafeDriver - Data Engineering Team"
                },
                "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat()
            }]
        }
        
        self._enviar_payload(self.webhook_erro, payload)

    def _enviar_payload(self, webhook_url, payload):
        """Método interno blindado para gerenciar os disparos e falhas da API."""
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            logger.error(f"COMUNICADOR: Discord recusou a mensagem. Status Code: {e.response.status_code}")
        except Exception as e:
            logger.error(f"COMUNICADOR: Falha de conexão ao tentar enviar mensagem: {e}")
