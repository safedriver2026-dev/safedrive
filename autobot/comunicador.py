import requests
import json
import os

class RoboComunicador:
    def __init__(self, webhook_sucesso=None, webhook_erro=None):
        self.webhook_sucesso = webhook_sucesso
        self.webhook_erro = webhook_erro

    def enviar_mensagem(self, webhook_url, mensagem, cor=None):
        if not webhook_url:
            print(f"Aviso: URL do webhook não configurada. Mensagem não enviada: {mensagem}")
            return

        headers = {'Content-Type': 'application/json'}
        payload = {
            "embeds": [
                {
                    "description": mensagem,
                    "color": cor if cor else 0x00ff00 # Verde padrão
                }
            ]
        }
        try:
            response = requests.post(webhook_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Erro ao enviar mensagem para o Discord: {e}")

    def enviar_relatorio_operacional(self, titulo, detalhes=None):
        mensagem = f"**{titulo}**"
        if detalhes:
            for chave, valor in detalhes.items():
                mensagem += f"\n- **{chave}**: {valor}"
        self.enviar_mensagem(self.webhook_sucesso, mensagem, cor=0x00ff00)

    def enviar_alerta_tecnico(self, titulo, erro_detalhes):
        mensagem = f"🚨 **ALERTA TÉCNICO: {titulo}** 🚨\n```\n{erro_detalhes}\n```"
        self.enviar_mensagem(self.webhook_erro, mensagem, cor=0xff0000)
