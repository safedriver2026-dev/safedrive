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

    def _disparar(self, url, payload):
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Falha ao enviar webhook para {url}: {e}")
            return False
        except Exception as e:
            logger.error(f"Erro inesperado ao enviar webhook para {url}: {e}")
            return False

    def enviar_relatorio_executivo(self, stats):
        """
        Gera um relatório detalhado com métricas de Engenharia e IA.
        stats deve conter: status_camadas (dict), metrics_ia (dict), hygiene (dict), duracao (str)
        """
        if not self.webhook_sucesso:
            logger.warning("Webhook de sucesso não configurado. Relatório executivo não será enviado.")
            return False

        # Determina a cor baseada na taxa de recuperação (Higiene)
        taxa = stats.get('hygiene', {}).get('taxa_recuperacao', 100)
        cor = self.COR_SUCESSO if taxa > 80 else self.COR_ALERTA

        # Se o pipeline estiver em repouso, mudamos o título
        em_repouso = all(v == "⏭️ (Cache)" for v in stats.get('status_camadas', {}).values())
        titulo = "😴 Ciclo SafeDriver: Sem Alterações" if em_repouso else "🛡️ Ciclo SafeDriver: Atualização Concluída"

        # Data e hora atual para a descrição
        agora_sp = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d/%m/%Y %H:%M:%S")
        descricao_relatorio = f"Status atual do ecossistema de dados SafeDriver, gerado em {agora_sp}."

        embed = {
            "title": titulo,
            "description": descricao_relatorio,
            "color": cor,
            "fields": [
                {
                    "name": "⛓️ Status do Pipeline",
                    "value": (
                        f"**Bronze:** {stats['status_camadas'].get('bronze', 'N/A')}\n"
                        f"**Prata:** {stats['status_camadas'].get('prata', 'N/A')}\n"
                        f"**IA (Treino):** {stats['status_camadas'].get('ia', 'N/A')}\n"
                        f"**Ouro (DW):** {stats['status_camadas'].get('ouro', 'N/A')}"
                    ),
                    "inline": False
                },
                {
                    "name": "🧹 Higiene de Dados (Prata)",
                    "value": (
                        f"Entrada: `{stats['hygiene'].get('linhas_in', 0)}` pts\n"
                        f"Recuperados: `{stats['hygiene'].get('linhas_out', 0)}` pts\n"
                        f"Taxa: `{taxa}%` ✅"
                    ),
                    "inline": True
                },
                {
                    "name": "🧠 Performance IA",
                    "value": (
                        f"Erro (MAE): `{stats['metrics_ia'].get('mae', 'N/A')}`\n"
                        f"Algoritmo: `CatBoost Tweedie`"
                    ),
                    "inline": True
                },
                {
                    "name": "🏆 Destino Ouro (BigQuery)",
                    "value": (
                        f"Total Predições: `{stats['hygiene'].get('linhas_ouro', 'N/A')}`\n"
                        f"Dataset: `safedriver_gold`"
                    ),
                    "inline": False
                },
                {
                    "name": "⏱️ Tempo de Execução",
                    "value": f"`{stats.get('duracao', '0s')}`",
                    "inline": True
                }
            ],
            "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
            "footer": {"text": self.rodape_padrao}
        }

        payload = {"username": "SafeDriver Maestro", "embeds": [embed]}
        return self._disparar(self.webhook_sucesso, payload)

    def relatar_erro_critico(self, modulo, erro):
        """
        Envia uma notificação de erro crítico para o Discord.
        """
        if not self.webhook_erro:
            logger.warning("Webhook de erro não configurado. Erro crítico não será enviado.")
            return False

        agora_sp = datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%d/%m/%Y %H:%M:%S")

        embed_erro = {
            "title": f"🚨 Falha Crítica: {modulo}",
            "description": f"Ocorreu um erro inesperado no módulo **{modulo}**.\n\n**Detalhes do Erro:**\n```\n{erro}\n```\n\nPor favor, verifique os logs para mais informações.",
            "color": self.COR_ERRO,
            "timestamp": datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),
            "footer": {"text": self.rodape_padrao}
        }

        payload = {"username": "SafeDriver Alerta", "embeds": [embed_erro]}
        return self._disparar(self.webhook_erro, payload)
