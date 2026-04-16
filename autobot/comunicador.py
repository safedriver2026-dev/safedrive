import os
import requests
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ComunicadorSafeDriver:
    def __init__(self):
        self.webhook_sucesso = os.getenv("DISCORD_SUCESSO")
        self.webhook_erro = os.getenv("DISCORD_ERRO")
        self.COR_SUCESSO = 3066993  # Verde
        self.COR_ALERTA = 16776960 # Amarelo
        self.COR_ERRO = 15158332    # Vermelho
        self.rodape_padrao = "SafeDriver Autobot • Inteligência de Segurança Pública"
        self.fuso_br = ZoneInfo("America/Sao_Paulo")

    def _obter_agora_br(self):
        """Obtém a data e hora atual no fuso horário de São Paulo."""
        return datetime.now(self.fuso_br)

    def _formatar_numero(self, valor):
        """
        Formata um número inteiro para o padrão brasileiro (ex: 1.000.000)
        para melhor legibilidade em relatórios executivos.
        """
        try:
            # Converte para int primeiro para garantir que não há casas decimais indesejadas
            # e depois formata com separador de milhares.
            return f"{int(valor):,}".replace(",", ".")
        except (ValueError, TypeError):
            # Retorna o valor original se não for possível formatar (ex: se for string)
            return valor

    def _disparar(self, url, payload):
        """
        Envia o payload para a URL do webhook.
        Retorna True em caso de sucesso, False em caso de falha.
        """
        if not url:
            logger.warning("URL do webhook não fornecida. Impossível disparar.")
            return False
        try:
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()  # Levanta um HTTPError para códigos de status ruins (4xx ou 5xx)
            logger.info(f"Webhook enviado com sucesso para {url[:30]}...")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Falha ao enviar webhook para {url[:30]}...: {e}")
            return False

    def enviar_relatorio_executivo(self, stats):
        """
        Envia um relatório executivo para o Discord com base nas estatísticas fornecidas.
        Inclui métricas de higiene, severidade e integridade do pipeline.
        """
        if not self.webhook_sucesso:
            logger.warning("Webhook de sucesso não configurado. Não foi possível enviar o relatório executivo.")
            return False

        agora = self._obter_agora_br()
        higiene_stats = stats.get('hygiene', {})

        # Métricas de higiene e volume
        taxa = higiene_stats.get('taxa_recuperacao', 100)
        cura_grade = self._formatar_numero(higiene_stats.get('recuperado_grade', 0))
        linhas_ouro = self._formatar_numero(higiene_stats.get('linhas_ouro', 0))

        # Métrica de Severidade (se presente nas stats)
        severidade_total = self._formatar_numero(stats.get('severidade_total', 0))

        # Define a cor do embed com base na taxa de recuperação
        cor = self.COR_SUCESSO if taxa > 90 else self.COR_ALERTA

        # Verifica o status das camadas para determinar se o pipeline está em repouso (cache)
        status_camadas = stats.get('status_camadas', {})
        em_repouso = all("Cache" in str(v) for v in status_camadas.values())

        # Título dinâmico para o relatório
        titulo = "💤 SafeDriver: Mantido em Cache" if em_repouso else "🛡️ SafeDriver: Snapshot de Risco Atualizado"

        # Campos padrão do embed
        fields = [
            {
                "name": "⛓️ Integridade do Pipeline",
                "value": (
                    f"**Bronze:** {status_camadas.get('bronze', 'N/A')}\n"
                    f"**Prata:** {status_camadas.get('prata', 'N/A')}\n"
                    f"**IA (Train):** {status_camadas.get('ia', 'N/A')}\n"
                    f"**Ouro (DW):** {status_camadas.get('ouro', 'N/A')}"
                ),
                "inline": False
            },
            {
                "name": "🧹 Higiene Crítica",
                "value": f"`{taxa}%` ✅",
                "inline": True
            },
            {
                "name": "🗺️ Malha Ativa", 
                "value": f"`{cura_grade}` hex",
                "inline": True
            },
            {
                "name": "🏆 Volume de Predição",
                "value": f"`{linhas_ouro}` cenários",
                "inline": True
            },
            {
                "name": "🧠 Precisão (MAE)",
                "value": f"`{stats.get('metrics_ia', {}).get('mae', 'N/A')}`",
                "inline": True
            },
            {
                "name": "⏱️ Tempo de Ciclo",
                "value": f"`{stats.get('duracao', '0s')}`",
                "inline": True
            }
        ]

        # Adiciona a métrica de severidade se o pipeline não estiver em repouso e houver dados de severidade
        if not em_repouso and severidade_total != "0":
             fields.insert(4, { # Insere na posição 4 para ficar antes da Precisão (MAE)
                "name": "🔥 Severidade Total",
                "value": f"`{severidade_total}` pts",
                "inline": True
            })

        embed = {
            "title": titulo,
            "description": f"Relatório de inteligência gerado em **{agora.strftime('%d/%m/%Y %H:%M:%S')}**.",
            "color": cor,
            "fields": fields,
            "timestamp": agora.isoformat(),
            "footer": {"text": self.rodape_padrao}
        }

        payload = {"username": "SafeDriver Maestro", "embeds": [embed]}
        return self._disparar(self.webhook_sucesso, payload)

    def relatar_erro_critico(self, modulo, erro):
        """
        Envia um relatório de erro crítico para o Discord.
        Limita o tamanho da mensagem de erro para evitar falhas no webhook.
        """
        if not self.webhook_erro:
            logger.warning("Webhook de erro não configurado. Não foi possível enviar o relatório de erro.")
            return False

        agora = self._obter_agora_br()

        # Limita o erro a 1800 caracteres para evitar problemas com o limite do Discord
        erro_formatado = str(erro)[:1800] 
        if len(str(erro)) > 1800:
            erro_formatado += "..." # Adiciona reticências se o erro foi truncado

        embed_erro = {
            "title": f"🚨 Falha Crítica: {modulo}",
            "description": f"Ocorreu uma interrupção no módulo **{modulo}**.\n\n```\n{erro_formatado}\n```",
            "color": self.COR_ERRO,
            "timestamp": agora.isoformat(),
            "footer": {"text": self.rodape_padrao}
        }

        payload = {"username": "SafeDriver Alerta", "embeds": [embed_erro]}
        return self._disparar(self.webhook_erro, payload)
