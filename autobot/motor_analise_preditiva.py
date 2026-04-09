import sys, os, requests, traceback, hashlib, gc, warnings, re, time
from pathlib import Path
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import polars as pl
import pandas as pd
import numpy as np
import h3, holidays, boto3
from catboost import CatBoostRegressor

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

class NotificadorDiscord:
    def __init__(self):
        self.url_sucesso = os.environ.get("DISCORD_SUCESSO", "").strip()
        self.url_erro = os.environ.get("DISCORD_ERRO", "").strip()

    def enviar(self, webhook_url, titulo, mensagem, cor):
        if not webhook_url: return
        payload = {
            "embeds": [{
                "title": titulo,
                "description": mensagem,
                "color": cor,
                "footer": {"text": f"SafeDriver AI • {datetime.now().strftime('%d/%m/%Y %H:%M')}"}
            }]
        }
        try:
            requests.post(webhook_url, json=payload, timeout=10)
        except: pass

    def relatar_sucesso(self, total_linhas, tempo_segundos, anos_proc):
        msg = (
            f"**Status:** 🟢 Operação Concluída\n"
            f"**Anos Processados:** {anos_proc}\n"
            f"**Volume de Dados:** {total_linhas:,} registros limpos\n"
            f"**Tempo de Processamento:** {tempo_segundos:.1f} segundos\n"
            f"**Destino:** Data Lake atualizado (Cloudflare R2)."
        )
        self.enviar(self.url_sucesso, "📊 Relatório Executivo - Pipeline", msg, 3066993) # Verde

    def relatar_erro(self, erro_msg):
        msg = f"**Status:** 🔴 Falha Crítica\n**Detalhes:**\n
http://googleusercontent.com/immersive_entry_chip/0

### O que acontece agora?
* **Se der certo:** Ele vai pegar o número exato de linhas que a IA mastigou (usando `df_prata.height`), o tempo exato em segundos e os anos que foram compilados, montando um card Verde no seu servidor.
* **Se der pau:** Qualquer `Exception` (seja de conexão, Polars ou do CatBoost) cai no bloco `try...except` na última linha, formata a pilha de erros e lança um card Vermelho, permitindo que você veja onde quebrou pelo celular sem precisar abrir o GitHub Actions.

Mande um `git push`. O pipeline não está mais "mudo".
