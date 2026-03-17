import io, os, json, math, shutil, unicodedata, warnings
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
        self.ano_vigente = self.inicio.year
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
            "sincronizacao": {"avaliados": 0, "atualizados": 0, "removidos": 0},
            "novos_dados": False
        }
        
        if not os.path.exists(self.lock):
            self.auditoria["modo"] = "HARD_RESET"
            if os.path.exists('datalake'): shutil.rmtree('datalake', ignore_errors=True)
        
        for d in ['raw', 'trusted', 'refined', 'metadata']:
            os.makedirs(f'datalake/{d}', exist_ok=True)

    def _criar_sessao_resiliente(self):
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=2, status_forcelist=[403, 429, 500, 502, 503, 504], allowed_methods=["HEAD", "GET", "OPTIONS"])
        s.mount('http://', HTTPAdapter(max_retries=retries))
        s.mount('https://', HTTPAdapter(max_retries=retries))
        s.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
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

    def _verificar_necessidade_download(self, url, ano):
        path = f"datalake/metadata/tamanho_{ano}.json"
        try:
            head = self.session.head(url, timeout=30, allow_redirects=True)
            tamanho = int(head.headers.get('Content-Length', 0))
            if os.path.exists(path):
                with open(path, 'r') as f:
                    if json.load(f).get('tamanho') == tamanho: return False, tamanho
            return True, tamanho
        except: return True, 0

    def _notificar(self, s=True, e=None):
        u = os.environ.get('DISCORD_SUCESSO') if s else os.environ.get('DISCORD_ERRO')
        if not u: return
        desc = "Novos arquivos detectados." if self.auditoria['novos_dados'] else "Re-treinando IA com cache."
        m = {
            "embeds": [{
                "title": f"SafeDriver Engine: {self.auditoria['modo']}",
                "description": desc if s else "Falha Critica",
                "color": 3066993 if s else 15158332,
                "fields": [
                    {"name": "Camadas", "value": f"RAW: {self.auditoria['volumes']['raw']:,}\nTRUSTED: {self.auditoria['volumes']['trusted']:,}\nREFINED: {self.auditoria['volumes']['refined_e']:,}", "inline": True},
                    {"name": "Performance", "value": f"MAE: {self.auditoria['validacao']['mae']:.4f}\nRMSE: {self.auditoria['validacao']['rmse']:.4f}", "inline": True},
                    {"name": "Cloud Sync", "value": f"Atualizados: {self.auditoria['sincronizacao']['atualizados']:,}", "inline": True}
                ],
                "footer": {"text": f"Duracao: {(datetime.now() - self.inicio).total_seconds():.1f}s"}
            }]
        }
        if not s: m["embeds"][0]["fields"].append({"name": "Erro", "value": f"{str(e)[:1000]}"})
        self.session.post(u, json=m)

    def _extrair_arquivo(self, url, ano, caminho_raw, tamanho):
        r = self.session.get(url, timeout=120)
        if r.status_code != 200: raise ConnectionError(f"Erro HTTP {r.status_code}")
        pre = pd.read_excel(io.BytesIO(r.content), nrows=50, header=None)
        lh = next((i for i, row in pre.iterrows() if any(t in [self._limpar(str(c)) for c in row.values] for t in ['NUM_BO', 'LATITUDE', 'NATUREZA_APURADA'])), None)
        if lh is None: lh = 0
        df = pd.read_excel(io.BytesIO(r.content), skiprows=lh, dtype=str)
        df.columns = [self._normalizar(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()].copy()
        df.to_parquet(caminho_raw, index=False)
        with open(f"datalake/metadata/tamanho_{ano}.json", 'w') as f: json.dump({'tamanho': tamanho}, f)
        self.auditoria['novos_dados'] = True
        return df

    def _qualificar(self, raw):
        df = raw.copy()
        for c in ESQUEMA_CANONICO.keys():
            if c not in df.columns: df[c] = np.nan
        df['LATITUDE'] = df['LATITUDE'].apply(lambda x: self._corrigir_ponto_decimal(x, True))
        df['LONGITUDE'] = df['LONGITUDE'].apply(lambda x: self._corrigir_ponto_decimal(x, False))
        df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'DATA_OCORRENCIA_BO']).copy()
        df['DATA_OCORRENCIA_BO'] = df['DATA_OCORRENCIA_BO'].dt.normalize()
        df_t = df[df['DATA_OCORRENCIA_BO'].between(self.janela, self.ref)].copy()
        self.auditoria["volumes"]["trusted"] += len(df_t)
        df_t['CRIME_DETECTADO'] = df_t['NATUREZA_APURADA'].apply(self._classificar_crime)
        df_r = df_t.dropna(subset=['CRIME_DETECTADO']).copy()
        self.auditoria["volumes"]["refined_e"] += len(df_r)
        return df_t, df_r

    def _modelar(self, df_eventos):
        if df_eventos.empty: return pd.DataFrame(), pd.DataFrame()
        df = df_eventos.copy()
        df['pf'] = df.apply(lambda x: [p for p, k in PALAVRAS_CHAVE_PERFIL.items() if any(z in str(x).upper() for z in k)] or ['Motorista'], axis=1)
        df = df.explode('pf')
        df['gh'] = [gh.encode(la, lo, precision=7) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]
        df['hr'] = df['HORA_OCORRENCIA_BO'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)
        df['pd'] = df['hr'].apply(lambda h: 'Noite' if h >= 18 or h < 6 else 'Dia')
        df['rc'] = (self.ref - df['DATA_OCORRENCIA_BO']).dt.days
        df['yw'] = np.exp(-df['rc'] / 180) * df['CRIME_DETECTADO'].apply(lambda x: CATALOGO_CRIMES.get(x, {}).get('peso', 1.0))
        cl = DBSCAN(eps=0.005, min_samples=10).fit(df[['LATITUDE', 'LONGITUDE']])
        df['cl'] = cl.labels_
        pnl = df.groupby(['gh', 'pf', 'pd', 'DATA_OCORRENCIA_BO'], as_index=False).agg(y=('yw', 'sum'), v=('NUM_BO', 'count'), la=('LATITUDE', 'mean'), lo=('LONGITUDE', 'mean'), c=('cl', 'max'))
        self.auditoria["volumes"]["refined_m"] = len(pnl)
        pnl['target'] = pnl.groupby(['gh', 'pf', 'pd'])['y'].transform(lambda x: x.shift(-7).rolling(7, min_periods=1).sum())
        st = pnl.groupby('DATA_OCORRENCIA_BO')['v'].sum().reset_index().rename(columns={'DATA_OCORRENCIA_BO': 'ds', 'v': 'y'})
        if len(st) >= 2:
            pp = Prophet().fit(st)
            pre = pp.predict(pp.make_future_dataframe(periods=14))[['ds', 'yhat']]
            pnl = pnl.merge(pre, left_on='DATA_OCORRENCIA_BO', right_on='ds', how='left')
            pnl['ft'] = pnl['yhat'] / max(pnl['yhat'].mean(), 1.0)
        else: pnl['ft'] = 1.0
        return pnl, df

    def _finalizar(self, pnl):
        if pnl.empty: return pd.DataFrame()
        val = pnl.dropna(subset=['target']).copy()
        if len(val) >= 10:
            tr, te = train_test_split(val, test_size=0.2, shuffle=False)
            self.auditoria["validacao"]["treino"], self.auditoria["validacao"]["teste"] = len(tr), len(te)
            fs = ['la', 'lo', 'c', 'ft']
            md = xgb.XGBRegressor(n_estimators=100).fit(tr[fs], tr['target'])
            preds = md.predict(te[fs])
            self.auditoria['validacao']['mae'] = float(mean_absolute_error(te['target'], preds))
            self.auditoria['validacao']['rmse'] = float(math.sqrt(mean_squared_error(te['target'], preds)))
            grid = pnl.sort_values('DATA_OCORRENCIA_BO').groupby(['gh', 'pf', 'pd']).tail(1).copy()
            grid['rs'] = np.clip(md.predict(grid[fs]), 0, None)
        else:
            grid = pnl.sort_values('DATA_OCORRENCIA_BO').groupby(['gh', 'pf', 'pd']).tail(1).copy()
            grid['rs'] = grid['y']
            
        scaler_penalty = MinMaxScaler(feature_range=(1.0, 5.0))
        grid['penalty'] = scaler_penalty.fit_transform(grid[['rs']]).round(2)
        
        scaler_score = MinMaxScaler(feature_range=(0, 100))
        grid['score'] = scaler_score.fit_transform(grid[['rs']]).round(1)
        
        counts = grid['pf'].value_counts().to_dict()
        for p in self.auditoria["malha"]: self.auditoria["malha"][p] = counts.get(p, 0)
        
        if self.db:
            col = self.db.collection('niveis_risco')
            vs, l, o = set(), self.db.batch(), 0
            ats = {d.id: d.to_dict().get('penalty') for d in col.stream()}
            for _, r in grid.iterrows():
                did = f"{r['gh']}_{r['pf']}_{r['pd']}"
                vs.add(did); pnlty = float(r['penalty'])
                if did not in ats or ats[did] != pnlty or self.auditoria["modo"] == "HARD_RESET":
                    l.set(col.document(did), {'penalty': pnlty, 'score': float(r['score']), 'geohash': r['gh'], 'prefix': r['gh'][:4], 'perfil': r['pf'], 'periodo': r['pd'], 'ts': firestore.SERVER_TIMESTAMP}, merge=True)
                    o += 1; self.auditoria["sincronizacao"]["atualizados"] += 1
                if o >= 450: l.commit(); l, o = self.db.batch(), 0
            for rid in ats.keys():
                if rid not in vs:
                    l.delete(col.document(rid)); self.auditoria["sincronizacao"]["removidos"] += 1
                    o += 1
                    if o >= 450: l.commit(); l, o = self.db.batch(), 0
            if o > 0: l.commit()
        return grid

    def rodar(self):
        try:
            m = pd.DataFrame()
            anos = range(2022, self.ano_vigente + 1) if self.auditoria["modo"] == "HARD_RESET" else [self.ano_vigente]
            for a in anos:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{a}.xlsx"
                caminho_raw = f'datalake/raw/ssp_{a}.parquet'
                baixar, tamanho = self._verificar_necessidade_download(url, a)
                if baixar or not os.path.exists(caminho_raw):
                    df = self._extrair_arquivo(url, a, caminho_raw, tamanho)
                else:
                    df = pd.read_parquet(caminho_raw)
                
                self.auditoria["volumes"]["raw"] += len(df)
                t, d = self._qualificar(df)
                if not d.empty: m = pd.concat([m, d], ignore_index=True)
                
            if not m.empty:
                p, b = self._modelar(m)
                self._finalizar(p)
                
            with open(self.lock, 'w') as f: f.write(str(datetime.now()))
            self._notificar(True)
        except Exception as e: self._notificar(False, e); raise

if __name__ == "__main__":
    MotorSafeDriver().rodar()
