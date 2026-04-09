import sys, os, requests, traceback, hashlib, gc, warnings, re, time, json
from pathlib import Path
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3, polars as pl, pandas as pd, numpy as np
import h3, holidays, boto3, shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

class Telemetria:
    def __init__(self):
        self.sucesso = os.environ.get("DISCORD_SUCESSO", "").strip(' "\'')
        self.erro = os.environ.get("DISCORD_ERRO", "").strip(' "\'')

    def notificar_sucesso(self, titulo, tempo_execucao, registros, media_risco, status_s3):
        if not self.sucesso or not self.sucesso.startswith("https://discord"):
            print("⚠️ Webhook de sucesso ausente ou inválido.", file=sys.stderr)
            return

        payload = {
            "embeds": [{
                "title": f"🟢 {titulo}",
                "description": "**Relatório Executivo Diário**\nO pipeline concluiu a sincronização e modelagem com sucesso.",
                "color": 3066993,
                "fields": [
                    {"name": "📊 Volumetria (Camada Prata)", "value": f"{registros:,} registros", "inline": True},
                    {"name": "⚠️ Risco Médio Global", "value": f"{media_risco:.2f}", "inline": True},
                    {"name": "⏱️ Tempo de Execução", "value": f"{tempo_execucao:.1f}s", "inline": True},
                    {"name": "☁️ Backup Cloudflare R2", "value": status_s3, "inline": False}
                ],
                "footer": {"text": f"SafeDriver AI • {datetime.now().strftime('%d/%m/%Y %H:%M')}"}
            }]
        }

        try:
            requests.post(self.sucesso, json=payload, timeout=10)
        except Exception as e:
            print(f"Erro ao enviar webhook: {e}", file=sys.stderr)

    def notificar_erro(self, titulo, erro_msg):
        if not self.erro or not self.erro.startswith("https://discord"):
            print("⚠️ Webhook de erro ausente ou inválido.", file=sys.stderr)
            return

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
                        "name": "Stacktrace Resumido",
                        "value": f"```{stacktrace}```"
                    }
                ],
                "footer": {"text": f"SafeDriver AI • {datetime.now().strftime('%d/%m/%Y %H:%M')}"}
            }]
        }

        try:
            requests.post(self.erro, json=payload, timeout=10)
        except Exception as e:
            print(f"Erro ao enviar webhook de erro: {e}", file=sys.stderr)
