import io, os, json, math, shutil, hashlib, unicodedata, warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pygeohash as gh
import requests, firebase_admin
import xgboost as xgb
from firebase_admin import credentials, firestore
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from autobot.config import *

warnings.filterwarnings("ignore")

class MotorSafeDriver:
    def __init__(self, persistencia=True):
        self.inicio = datetime.now()
        self.ref = pd.Timestamp(self.inicio.date())
        self.janela = self.ref - pd.Timedelta(days=730)
        self.lock = 'datalake/metadata/baseline.lock'
        self.db = self._conectar() if persistencia else None
        self.session = self._criar_sessao_resiliente()
        self.auditoria = {
            "modo": "INCREMENTAL",
            "volumes": {"raw": 0, "trusted": 0, "refined_e": 0, "refined_m": 0},
            "validacao": {"treino": 0, "teste": 0, "mae": 0.0, "rmse": 0.0},
            "malha": {"Motorista": 0, "Motociclista": 0, "Pedestre": 0, "Ciclista": 0},
            "sincronizacao": {"avaliados": 0, "atualizados": 0, "removidos": 0}
        }
        
        if not os.path.exists(self.lock):
            self.auditoria["modo"] = "HARD_RESET"
            if os.path.exists('datalake'): shutil.rmtree('datalake', ignore_errors=True)
        
        for d in ['raw', 'trusted', 'refined', 'metadata']:
            os.makedirs(f'datalake/{d}', exist_ok=True)

    def _criar_sessao_resiliente(self):
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        s.mount('https://', HTTPAdapter(max_retries=retries))
        return s

    def _conectar(self):
        c = os.environ.get('FIREBASE_JSON')
        try:
            if c and not firebase_admin._apps:
                cred = credentials.Certificate(json.loads(c))
                firebase_admin.initialize_app(cred)
            return firestore.client()
        except: return None

    def _limpar(self, t):
        if pd.isna(t): return ""
        n = unicodedata.normalize('NFKD', str(t))
        return "".join([c for c in n if not unicodedata.combining(c)]).upper().strip()

    def _normalizar(self, n):
        l = self._limpar(n).replace(" ", "_")
        for k, v in DICIONARIO_SEMANTICO.items():
            if l in v: return k
        return l

    def _corrigir_ponto_decimal(self, v, is_lat=True):
        try:
            val = float(str(v).replace(',', '.'))
            if val == 0: return np.nan
            while abs(val) > 180: val /= 10.0
            if val > 0: val = -val
            if is_lat and (-25.5 <= val <= -19.5): return val
            if not is_lat and (-53.5 <= val <= -44.0): return val
            return np.nan
        except: return np.nan

    def _classificar_crime(self, x):
        if pd.isna(x) or str(x).strip() == "": return np.nan
        limpo = self._limpar(x)
        for k in CATALOGO_CRIMES.keys():
            if k != "OUTROS" and k in limpo: return k
        return "OUTROS"

    def _notificar(self, s=True, e=None):
        u = os.environ.get('DISCORD_SUCESSO') if s else os.environ.get('DISCORD_ERRO')
        if not u: return
        m = {
            "embeds": [{
                "title": f"🛡️ SafeDriver Engine: {self.auditoria['modo']}",
                "color": 3066993 if s else 15158332,
                "fields": [
                    {"name": "Janela", "value": f"{self.janela.strftime('%Y-%m-%d')} a {self.ref.strftime('%Y-%m-%d')}", "inline": False},
                    {"name": "Camadas", "value": f"RAW: {self.auditoria['volumes']['raw']:,}\nTRUSTED: {self.auditoria['volumes']['trusted']:,}\nREFINED: {self.auditoria['volumes']['refined_e']:,}", "inline": True},
                    {"name": "Performance", "value": f"MAE: {self.auditoria['validacao']['mae']:.4f}\nRMSE: {self.auditoria['validacao']['rmse']:.4f}", "inline": True},
                    {"name": "Cloud Sync", "value": f"Atualizados: {self.auditoria['sincronizacao']['atualizados']:,}", "inline": True}
                ],
                "footer": {"text": f"Duração: {(datetime.now() - self.inicio).total_seconds():.1f}s"}
            }]
        }
        if not s: m["embeds"][0]["fields"].append({"name": "Erro", "value": f"
http://googleusercontent.com/immersive_entry_chip/0
