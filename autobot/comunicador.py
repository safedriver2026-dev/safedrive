import os
import requests
import logging
from datetime import datetime

# Configuração de Log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComunicadorSafeDriver:
    def __init__(self):
        # As credenciais DEVEM vir do GitHub Secrets (.env localmente)
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.bot_token or not self.chat_id:
            logger.warning("⚠️ Credenciais do Telegram não encontradas. O modo silencioso foi ativado.")
            self.ativo = False
        else:
            self.ativo = True
            self.url_base = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"

    def _enviar_mensagem(self, texto):
        if not self.ativo:
            return

        payload = {
            "chat_id": self.chat_id,
            "text": texto,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }

        try:
            response = requests.post(self.url_base, json=payload, timeout=10)
            response.raise_for_status()
            logger.info("📱 Notificação enviada com sucesso para o Telegram.")
        except Exception as e:
            logger.error(f"Falha ao enviar notificação do Telegram: {e}")

    def relatar_sucesso(self, ano_ref, tempo_execucao, total_linhas):
        data_hora = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        
        mensagem = (
            f"🟢 <b>SafeDriver Pipeline: SUCESSO</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📅 <b>Referência:</b> {ano_ref}\n"
            f"⏱️ <b>Tempo de Execução:</b> {tempo_execucao}\n"
            f"📊 <b>Registos Processados:</b> {total_linhas:,}\n\n"
            f"✅ <b>Bronze:</b> Dados Brutos Ingeridos\n"
            f"✅ <b>Prata:</b> Geocodificação e H3 Concluídos\n"
            f"✅ <b>Ouro:</b> Inferência IA e BigQuery Atualizados\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"<i>Atualizado em: {data_hora}</i>"
        )
        self._enviar_mensagem(mensagem)

    def relatar_erro(self, camada, erro_traceback):
        mensagem = (
            f"🔴 <b>CRÍTICO: Falha no SafeDriver</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ <b>Camada Afetada:</b> {camada.upper()}\n"
            f"🚨 <b>Erro:</b> <code>{str(erro_traceback)[:200]}...</code>\n\n"
            f"<i>O pipeline foi interrompido para evitar corrupção no Data Lake. Verifique os logs no GitHub Actions.</i>"
        )
        self._enviar_mensagem(mensagem)

# Exemplo de teste rápido (se rodar o arquivo isolado)
if __name__ == "__main__":
    # Para testar, exporte as variáveis no seu terminal antes:
    # export TELEGRAM_BOT_TOKEN="seu_token"
    # export TELEGRAM_CHAT_ID="seu_chat_id"
    comunicador = ComunicadorSafeDriver()
    comunicador.relatar_sucesso(2024, "02m 45s", 1543020)
