class Telemetria:
    def __init__(self):
        self.sucesso = os.environ.get("DISCORD_SUCESSO", "").strip(' "\'')
        self.erro = os.environ.get("DISCORD_ERRO", "").strip(' "\'')

    def _enviar_webhook(self, url, payload, tipo):
        if not url or not url.startswith("https://discord.com/api/webhooks/"):
            print(f"⚠️ Webhook {tipo} inválido ou ausente.", file=sys.stderr)
            return

        try:
            resp = requests.post(url, json=payload, timeout=10)

            if resp.status_code == 204:
                print(f"✅ Webhook {tipo} enviado com sucesso.")
            else:
                print(f"❌ REJEIÇÃO DO DISCORD ({tipo}): {resp.status_code} - {resp.text}", file=sys.stderr)

        except Exception as e:
            print(f"❌ FALHA DE CONEXÃO ({tipo}): {e}", file=sys.stderr)

    def notificar_sucesso(self, titulo, tempo_execucao, registros, media_risco, status_s3):
        payload = {
            "embeds": [{
                "title": f"🟢 {titulo}",
                "description": "**Relatório Executivo SafeDriver**\nPipeline executado com sucesso.",
                "color": 3066993,
                "fields": [
                    {"name": "📊 Volumetria", "value": f"{registros:,} ocorrências", "inline": True},
                    {"name": "⚠️ Risco Médio", "value": f"{media_risco:.2f}", "inline": True},
                    {"name": "⏱ Tempo", "value": f"{tempo_execucao:.1f}s", "inline": True},
                    {"name": "☁️ Backup R2", "value": status_s3, "inline": False}
                ],
                "footer": {"text": f"SafeDriver • {datetime.now().strftime('%d/%m/%Y %H:%M')}"}
            }]
        }

        self._enviar_webhook(self.sucesso, payload, "SUCESSO")

    def notificar_info(self, titulo, corpo):
        payload = {
            "embeds": [{
                "title": f"🔵 {titulo}",
                "description": corpo,
                "color": 3447003,
                "footer": {"text": f"SafeDriver • {datetime.now().strftime('%d/%m/%Y %H:%M')}"}
            }]
        }

        self._enviar_webhook(self.sucesso, payload, "INFO")

    def notificar_erro(self, titulo, erro_msg):
        stacktrace = traceback.format_exc()

        if not stacktrace:
            stacktrace = str(erro_msg)

        if len(stacktrace) > 1000:
            stacktrace = stacktrace[:1000] + "..."

        payload = {
            "embeds": [{
                "title": f"🔴 {titulo}",
                "description": "**Falha Crítica no Pipeline MLOps**",
                "color": 15158332,
                "fields": [
                    {
                        "name": "Detalhes Técnicos",
                        "value": f"```{stacktrace}```"
                    }
                ],
                "footer": {"text": f"SafeDriver • {datetime.now().strftime('%d/%m/%Y %H:%M')}"}
            }]
        }

        self._enviar_webhook(self.erro, payload, "ERRO")
