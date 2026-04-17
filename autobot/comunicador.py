import os
import requests
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfiguracaoComunicador:
    WEBHOOK_SUCESSO = os.getenv("DISCORD_SUCESSO")
    WEBHOOK_ERRO = os.getenv("DISCORD_ERRO")
    COR_SUCESSO = 3066993
    COR_ALERTA = 16776960
    COR_ERRO = 15158332
    RODAPE_PADRAO = "SafeDriver Autobot • Inteligencia de Seguranca Publica"
    FUSO_HORARIO = "America/Sao_Paulo"
    NOME_USUARIO_SUCESSO = "SafeDriver Maestro"
    NOME_USUARIO_ERRO = "SafeDriver Alerta"

class ComunicadorDiscord:
    def __init__(self):
        self.configuracao = ConfiguracaoComunicador()
        self.fuso = ZoneInfo(self.configuracao.FUSO_HORARIO)

    def _obter_horario_atual(self) -> datetime:
        return datetime.now(self.fuso)

    def _formatar_milhares(self, valor) -> str:
        try:
            return f"{int(valor):,}".replace(",", ".")
        except (ValueError, TypeError):
            return str(valor)

    def _enviar_webhook(self, url: str, carga_util: dict) -> bool:
        if not url:
            logger.warning("URL do webhook ausente. Envio cancelado.")
            return False
        try:
            resposta = requests.post(url, json=carga_util, timeout=10)
            resposta.raise_for_status()
            return True
        except requests.exceptions.RequestException as erro:
            logger.error(f"Falha na comunicacao com webhook: {erro}")
            return False

    def enviar_relatorio_operacional(self, estatisticas: dict) -> bool:
        if not self.configuracao.WEBHOOK_SUCESSO:
            return False

        agora = self._obter_horario_atual()
        metricas_higiene = estatisticas.get('hygiene', {})
        
        taxa_recuperacao = metricas_higiene.get('taxa_recuperacao', 100)
        malha_processada = self._formatar_milhares(metricas_higiene.get('recuperado_grade', 0))
        cenarios_gerados = self._formatar_milhares(metricas_higiene.get('linhas_ouro', 0))
        
        severidade_total = self._formatar_milhares(estatisticas.get('severidade_total', 0))
        
        cor_destaque = self.configuracao.COR_SUCESSO if taxa_recuperacao > 90 else self.configuracao.COR_ALERTA
        
        status_camadas = estatisticas.get('status_camadas', {})
        operacao_em_cache = all("Cache" in str(valor) for valor in status_camadas.values())
        
        titulo_relatorio = "💤 SafeDriver: Operacao em Cache" if operacao_em_cache else "🛡️ SafeDriver: Atualizacao de Risco Concluida"

        campos_relatorio = [
            {
                "name": "⛓️ Integridade do Fluxo",
                "value": (
                    f"**Bronze:** {status_camadas.get('bronze', 'N/A')}\n"
                    f"**Prata:** {status_camadas.get('prata', 'N/A')}\n"
                    f"**Treinamento IA:** {status_camadas.get('ia', 'N/A')}\n"
                    f"**Sincronizacao Ouro:** {status_camadas.get('ouro', 'N/A')}"
                ),
                "inline": False
            },
            {
                "name": "🧹 Higiene de Dados",
                "value": f"`{taxa_recuperacao}%` ✅",
                "inline": True
            },
            {
                "name": "🗺️ Cobertura Espacial", 
                "value": f"`{malha_processada}` hex",
                "inline": True
            },
            {
                "name": "🏆 Volume Analitico",
                "value": f"`{cenarios_gerados}` cenarios",
                "inline": True
            },
            {
                "name": "🧠 Precisao do Modelo (MAE)",
                "value": f"`{estatisticas.get('metrics_ia', {}).get('mae', 'N/A')}`",
                "inline": True
            },
            {
                "name": "⏱️ Tempo de Processamento",
                "value": f"`{estatisticas.get('duracao', '0s')}`",
                "inline": True
            }
        ]

        if not operacao_em_cache and str(severidade_total) != "0":
             campos_relatorio.insert(4, {
                "name": "🔥 Severidade Acumulada",
                "value": f"`{severidade_total}` pts",
                "inline": True
            })

        estrutura_mensagem = {
            "title": titulo_relatorio,
            "description": f"Relatorio gerado em **{agora.strftime('%d/%m/%Y %H:%M:%S')}**.",
            "color": cor_destaque,
            "fields": campos_relatorio,
            "timestamp": agora.isoformat(),
            "footer": {"text": self.configuracao.RODAPE_PADRAO}
        }

        carga_util = {"username": self.configuracao.NOME_USUARIO_SUCESSO, "embeds": [estrutura_mensagem]}
        return self._enviar_webhook(self.configuracao.WEBHOOK_SUCESSO, carga_util)

    def reportar_falha_critica(self, modulo_origem: str, detalhes_erro: str) -> bool:
        if not self.configuracao.WEBHOOK_ERRO:
            return False

        agora = self._obter_horario_atual()
        erro_truncado = str(detalhes_erro)[:1800] 
        
        if len(str(detalhes_erro)) > 1800:
            erro_truncado += "..."

        estrutura_mensagem = {
            "title": f"🚨 Falha de Operacao: {modulo_origem}",
            "description": f"Interrupcao detectada no modulo **{modulo_origem}**.\n\n```\n{erro_truncado}\n```",
            "color": self.configuracao.COR_ERRO,
            "timestamp": agora.isoformat(),
            "footer": {"text": self.configuracao.RODAPE_PADRAO}
        }

        carga_util = {"username": self.configuracao.NOME_USUARIO_ERRO, "embeds": [estrutura_mensagem]}
        return self._enviar_webhook(self.configuracao.WEBHOOK_ERRO, carga_util)
