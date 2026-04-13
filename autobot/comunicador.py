import os
import requests
import logging
from datetime import datetime

# Configuração de Log para Produção
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComunicadorSafeDriver:
    def __init__(self):
        # Busca os Webhooks do Discord nas variáveis de ambiente (GitHub Secrets)
        self.webhook_sucesso = os.getenv("DISCORD_SUCESSO")
        self.webhook_erro = os.getenv("DISCORD_ERRO")

        # Modo Fail-Secure: Se esquecer de colocar a chave, o código avisa mas não quebra o pipeline
        if not self.webhook_sucesso or not self.webhook_erro:
            logger.warning("⚠️ Webhooks do Discord não encontrados (DISCORD_SUCESSO / DISCORD_ERRO). Modo silencioso ativado.")
            self.ativo = False
        else:
            self.ativo = True

    def _enviar_discord(self, webhook_url, embed_payload):
        if not self.ativo:
            return

        # O Discord exige que o payload seja enviado dentro de uma lista chamada "embeds"
        payload = {
            "embeds": [embed_payload]
        }

        try:
            # Timeout de 10s evita que o GitHub Actions fique travado consumindo minutos
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("📱 Notificação Embed enviada com sucesso para o Discord.")
        except Exception as e:
            logger.error(f"Falha ao comunicar com a API do Discord: {e}")

    def relatar_sucesso(self, ano_ref, tempo_execucao, total_linhas):
        """Envia um card verde formatado para o canal de Sucesso."""
        embed = {
            "title": "🟢 SafeDriver Pipeline: SUCESSO",
            "color": 5763719, # Código decimal para a cor Verde
            "description": "A esteira de processamento finalizou todas as etapas sem erros.",
            "fields": [
                {"name": "📅 Referência", "value": str(ano_ref), "inline": True},
                {"name": "⏱️ Tempo de Execução", "value": str(tempo_execucao), "inline": True},
                {"name": "📊 Registros Prontos", "value": f"{total_linhas:,}".replace(",", "."), "inline": False},
                {"name": "Status das Camadas", "value": "✅ Bronze (Ingestão RAW)\n✅ Prata (Limpeza e Geo H3)\n✅ Ouro (IA e BigQuery)", "inline": False}
            ],
            "footer": {
                "text": f"Atualizado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
            }
        }
        self._enviar_discord(self.webhook_sucesso, embed)

    def relatar_erro(self, camada, erro_traceback):
        """Envia um card vermelho com o log do erro para o canal de Erros Críticos."""

        # Isolando a formatação do erro para evitar bugs de copy-paste
        # A correção está aqui: a f-string foi ajustada para não ter quebra de linha inesperada.
        erro_formatado = f"Ocorreu um erro crítico na camada **{camada}** do pipeline SafeDriver.\n\n```python\n{erro_traceback}\n```"

        embed = {
            "title": "🔴 SafeDriver Pipeline: ERRO CRÍTICO",
            "color": 15548997, # Código decimal para a cor Vermelha
            "description": erro_formatado,
            "fields": [
                {"name": "Camada do Erro", "value": camada, "inline": True},
                {"name": "Data/Hora do Erro", "value": datetime.now().strftime('%d/%m/%Y %H:%M:%S'), "inline": True}
            ],
            "footer": {
                "text": "Por favor, verifique os logs para mais detalhes."
            }
        }
        self._enviar_discord(self.webhook_erro, embed)
