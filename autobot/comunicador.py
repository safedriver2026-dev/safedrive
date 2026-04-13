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

        # Modo Fail-Secure: Se esquecer de colocar a chave, não quebra o pipeline
        if not self.webhook_sucesso or not self.webhook_erro:
            logger.warning("⚠️ Webhooks do Discord não encontrados (DISCORD_SUCESSO / DISCORD_ERRO). Modo silencioso ativado.")
            self.ativo = False
        else:
            self.ativo = True

    def _enviar_discord(self, webhook_url, embed_payload):
        if not self.ativo:
            return

        payload = {
            "embeds": [embed_payload]
        }

        try:
            # Timeout de 10s evita que o GitHub Actions fique travado se o Discord cair
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("📱 Notificação Embed enviada com sucesso para o Discord.")
        except Exception as e:
            logger.error(f"Falha ao comunicar com a API do Discord: {e}")

    def relatar_sucesso(self, ano_ref, tempo_execucao, total_linhas):
        """Envia um card verde para o canal de Sucesso."""
        embed = {
            "title": "🟢 SafeDriver Pipeline: SUCESSO",
            "color": 5763719, # Verde (Cor decimal para Discord Embeds)
            "description": "A esteira de processamento finalizou todas as etapas sem erros.",
            "fields": [
                {"name": "📅 Referência", "value": str(ano_ref), "inline": True},
                {"name": "⏱️ Tempo de Execução", "value": str(tempo_execucao), "inline": True},
                {"name": "📊 Registros Prontos", "value": f"{total_linhas:,}", "inline": False},
                {"name": "Status das Camadas", "value": "✅ Bronze (Ingestão RAW)\n✅ Prata (Limpeza e Geo H3)\n✅ Ouro (IA e BigQuery)", "inline": False}
            ],
            "footer": {
                "text": f"Atualizado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
            }
        }
        self._enviar_discord(self.webhook_sucesso, embed)

    def relatar_erro(self, camada, erro_traceback):
        """Envia um card vermelho para o canal de Erros Críticos."""
        embed = {
            "title": "🔴 CRÍTICO: Falha no SafeDriver",
            "color": 15548997, # Vermelho (Cor decimal para Discord Embeds)
            "description": "O pipeline foi abortado preventivamente para não corromper o Data Lake. Verifique os logs no GitHub Actions.",
            "fields": [
                {"name": "⚠️ Camada Afetada", "value": camada.upper(), "inline": False},
                {"name": "🚨 Traceback do Erro", "value": f"```python\n{str(erro_traceback)[:1000]}...\n```", "inline": False} # Limita o traceback para não exceder o limite do Discord
            ],
            "footer": {
                "text": f"Ocorrido em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
            }
        }
        self._enviar_discord(self.webhook_erro, embed)

# Exemplo de uso (apenas para testar o arquivo isoladamente)
if __name__ == "__main__":
    # Para testar, defina estas variáveis de ambiente no seu terminal:
    # export DISCORD_SUCESSO="SUA_WEBHOOK_URL_DE_SUCESSO"
    # export DISCORD_ERRO="SUA_WEBHOOK_URL_DE_ERRO"
    comunicador = ComunicadorSafeDriver()

    # Teste de sucesso
    comunicador.relatar_sucesso(2024, "02m 45s", 1543020)

    # Teste de erro
    try:
        1 / 0
    except Exception as e:
        import traceback
        comunicador.relatar_erro("Camada de Teste", traceback.format_exc())
