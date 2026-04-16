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
        self.COR_SUCESSO = 3066993  # Verde
        self.COR_ALERTA = 16776960 # Amarelo
        self.COR_ERRO = 15158332    # Vermelho
        self.rodape_padrao = "SafeDriver Autobot • Monitoramento de Ciclo de Dados"
        self.fuso_br = ZoneInfo("America/Sao_Paulo")

    def _obter_agora_br(self):
        """Retorna o objeto datetime atualizado para o horário de Brasília."""
        return datetime.now(self.fuso_br)

    def _disparar(self, url, payload):
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Falha ao enviar webhook para {url}: {e}")
            return False

    def enviar_relatorio_executivo(self, stats):
        if not self.webhook_sucesso:
            logger.warning("Webhook de sucesso não configurado.")
            return False

        agora = self._obter_agora_br()
        
        # Pega as métricas de forma robusta, independentemente de onde o main.py empacotar
        higiene_stats = stats.get('hygiene', {})
        taxa = higiene_stats.get('taxa_recuperacao', stats.get('taxa_recuperacao', 100))
        cura_grade = higiene_stats.get('recuperado_grade', stats.get('recuperado_grade', 0))
        linhas_ouro = higiene_stats.get('linhas_ouro', stats.get('linhas_ouro', "N/A")) # <--- Busca os dados da Ouro
        
        cor = self.COR_SUCESSO if taxa > 80 else self.COR_ALERTA

        em_repouso = all(v == "⏭️ (Cache)" for v in stats.get('status_camadas', {}).values())
        titulo = "😴 Ciclo SafeDriver: Sem Alterações" if em_repouso else "🛡️ Ciclo SafeDriver: Atualização Concluída"

        data_formatada = agora.strftime("%d/%m/%Y %H:%M:%S")
        descricao_relatorio = f"Status do ecossistema SafeDriver atualizado em **{data_formatada}** (Horário de Brasília)."

        embed = {
            "title": titulo,
            "description": descricao_relatorio,
            "color": cor,
            "fields": [
                {
                    "name": "⛓️ Status do Pipeline",
                    "value": (
                        f"**Bronze:** {stats.get('status_camadas', {}).get('bronze', 'N/A')}\n"
                        f"**Prata:** {stats.get('status_camadas', {}).get('prata', 'N/A')}\n"
                        f"**IA (Treino):** {stats.get('status_camadas', {}).get('ia', 'N/A')}\n"
                        f"**Ouro (DW):** {stats.get('status_camadas', {}).get('ouro', 'N/A')}"
                    ),
                    "inline": False
                },
                {
                    "name": "🧹 Higiene (Prata)",
                    "value": f"Taxa: `{taxa}%` ✅",
                    "inline": True
                },
                {
                    "name": "🛠️ Cura Geográfica", 
                    "value": f"`{cura_grade}` hexágonos",
                    "inline": True
                },
                {
                    "name": "🏆 Destino Ouro", # <--- CAMPO NOVO ADICIONADO AQUI
                    "value": f"`{linhas_ouro}` predições",
                    "inline": True
                },
                {
                    "name": "🧠 IA MAE",
                    "value": f"`{stats.get('metrics_ia', {}).get('mae', 'N/A')}`",
                    "inline": True
                },
                {
                    "name": "⏱️ Duração",
                    "value": f"`{stats.get('duracao', '0s')}`",
                    "inline": True
                }
            ],
            "timestamp": agora.isoformat(),
            "footer": {"text": self.rodape_padrao}
        }

        payload = {"username": "SafeDriver Maestro", "embeds": [embed]}
        return self._disparar(self.webhook_sucesso, payload)

    def relatar_erro_critico(self, modulo, erro):
        if not self.webhook_erro:
            return False

        agora = self._obter_agora_br()

        embed_erro = {
            "title": f"🚨 Falha Crítica: {modulo}",
            "description": f"Erro no módulo **{modulo}** às {agora.strftime('%H:%M:%S')}.\n\n```python\n{erro}\n```",
            "color": self.COR_ERRO,
            "timestamp": agora.isoformat(),
            "footer": {"text": self.rodape_padrao}
        }

        payload = {"username": "SafeDriver Alerta", "embeds": [embed_erro]}
        return self._disparar(self.webhook_erro, payload)
