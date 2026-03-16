import io, os, json, math, shutil, hashlib, logging, unicodedata, warnings, difflib
from datetime import datetime
import numpy as np
import pandas as pd
import pygeohash as gh
import requests, firebase_admin
import xgboost as xgb
from firebase_admin import credentials, firestore
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from autobot.config import *

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
        if c and not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(c))
            firebase_admin.initialize_app(cred)
        return firestore.client()

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
            limit = 90 if is_lat else 180
            if abs(val) > limit:
                val /= 1_000_000
            if abs(val) > limit:
                return np.nan
            return val
        except:
            return np.nan

    def _notificar(self, s=True, e=None):
        u = os.environ.get('DISCORD_SUCESSO') if s else os.environ.get('DISCORD_ERRO')
        if not u: return
        pipeline_desc = "Relatorio Semanal SafeDriver\nPipeline executado com sucesso."
        m = {
            "embeds": [{
                "title": f"🛡️ SafeDriver Engine: {self.auditoria['modo']}",
                "description": pipeline_desc,
                "color": 3066993 if s else 15158332,
                "fields": [
                    {"name": "Janela", "value": f"{self.janela.strftime('%Y-%m-%d')} a {self.ref.strftime('%Y-%m-%d')}", "inline": False},
                    {"name": "Camadas", "value": f"RAW: {self.auditoria['volumes']['raw']:,}\nTRUSTED: {self.auditoria['volumes']['trusted']:,}\nREFINED: {self.auditoria['volumes']['refined_e']:,}", "inline": True},
                    {"name": "Performance", "value": f"MAE: {self.auditoria['validacao']['mae']:.4f}\nRMSE: {self.auditoria['validacao']['rmse']:.4f}", "inline": True},
                    {"name": "Malha", "value": "\n".join([f"{k}: {v:,}" for k, v in self.auditoria['malha'].items()]), "inline": True},
                    {"name": "Cloud", "value": f"Atualizados: {self.auditoria['sincronizacao']['atualizados']:,}", "inline": True}
                ],
                "footer": {"text": f"Duração: {(datetime.now() - self.inicio).total_seconds():.1f}s"}
            }]
        }
        if not s: m["embeds"][0]["fields"].append({"name": "Erro", "value": f"```{str(e)}```"})
        self.session.post(u, json=m)

    def _extrair(self, b):
        xl = pd.ExcelFile(io.BytesIO(b))
        ds = []
        for s in xl.sheet_names:
            if any(x in s.upper() for x in ["CAMPOS", "METADADOS"]): continue
            am = xl.parse(s, nrows=50, header=None)
            lh = 0
            for i, r in am.iterrows():
                ts = [self._limpar(str(v)) for v in r.values]
                if any(k in ts for k in ['NUM_BO', 'NATUREZA_APURADA', 'LATITUDE', 'RUBRICA']):
                    lh = i
                    break
            df = xl.parse(s, skiprows=lh, dtype=str)
            df.columns = [self._normalizar(c) for c in df.columns]
            df = df.loc[:, ~df.columns.duplicated()].copy()
            ds.append(df)
        return pd.concat(ds, ignore_index=True) if ds else pd.DataFrame()

    def _qualificar(self, raw, a):
        df = raw.copy()
        for c in ESQUEMA_CANONICO.keys():
            if c not in df.columns: df[c] = np.nan
        
        df['LATITUDE'] = df['LATITUDE'].apply(lambda x: self._corrigir_ponto_decimal(x, True))
        df['LONGITUDE'] = df['LONGITUDE'].apply(lambda x: self._corrigir_ponto_decimal(x, False))
        
        df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'DATA_OCORRENCIA_BO']).copy()
        df['DATA_OCORRENCIA_BO'] = df['DATA_OCORRENCIA_BO'].dt.normalize()
        m_espacial = (df['DATA_OCORRENCIA_BO'].between(self.janela, self.ref)) & \
                     (df['LATITUDE'].between(LIMITES_GEOGRAFICOS['lat'][0], LIMITES_GEOGRAFICOS['lat'][1])) & \
                     (df['LONGITUDE'].between(LIMITES_GEOGRAFICOS['lon'][0], LIMITES_GEOGRAFICOS['lon'][1]))
        df_t = df[m_espacial].copy()
        self.auditoria["volumes"]["trusted"] += len(df_t)
        df_t['CRIME_DETECTADO'] = df_t['NATUREZA_APURADA'].apply(
            lambda x: next((k for k in CATALOGO_CRIMES.keys() if k in self._limpar(x)), None)
        )
        df_r = df_t.dropna(subset=['CRIME_DETECTADO']).copy()
        self.auditoria["volumes"]["refined_e"] = len(df_r)
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

    def _finalizar(self, pnl, df_base):
        if df_base.empty or pnl.empty: return
        val = pnl.dropna(subset=['target']).copy()
        if len(val) >= 2:
            tr, te = train_test_split(val, test_size=0.2, shuffle=False)
            self.auditoria["validacao"]["treino"], self.auditoria["validacao"]["teste"] = len(tr), len(te)
            fs = ['la', 'lo', 'c', 'ft']
            md = xgb.XGBRegressor(n_estimators=100).fit(tr[fs], tr['target'])
            self.auditoria['validacao']['mae'] = mean_absolute_error(te['target'], md.predict(te[fs]))
            self.auditoria['validacao']['rmse'] = math.sqrt(mean_squared_error(te['target'], md.predict(te[fs])))
            grid = pnl.sort_values('DATA_OCORRENCIA_BO').groupby(['gh', 'pf', 'pd']).tail(1).copy()
            grid['rs'] = np.clip(md.predict(grid[fs]), 0, None)
        else:
            grid = pnl.sort_values('DATA_OCORRENCIA_BO').groupby(['gh', 'pf', 'pd']).tail(1).copy()
            grid['rs'] = grid['y']
            
        scaler = MinMaxScaler(feature_range=(0, 100))
        grid['score'] = scaler.fit_transform(grid[['rs']]).round(1)
        grid['penalty'] = (1.0 + (grid['score'] / 50.0)).round(2)
        
        counts = grid['pf'].value_counts().to_dict()
        for p in self.auditoria["malha"]: self.auditoria["malha"][p] = counts.get(p, 0)
        
        bi_path = 'datalake/refined/power_bi_visualizacao.csv'
        df_base.merge(grid[['gh', 'pf', 'pd', 'score', 'penalty']], on=['gh', 'pf', 'pd'], how='left').to_csv(bi_path, index=False, sep=';', encoding='utf-8-sig')
        
        if self.db:
            col = self.db.collection('niveis_risco')
            vs, l, o = set(), self.db.batch(), 0
            ats = {d.id: d.to_dict().get('score') for d in col.stream()}
            for _, r in grid.iterrows():
                did = f"{r['gh']}_{r['pf']}_{r['pd']}"
                vs.add(did); sc = float(r['score'])
                if did not in ats or ats[did] != sc:
                    l.set(col.document(did), {'score': sc, 'penalty': float(r['penalty']), 'geohash': r['gh'], 'prefix': r['gh'][:4], 'perfil': r['pf'], 'periodo': r['pd'], 'ts': firestore.SERVER_TIMESTAMP}, merge=True)
                    o += 1; self.auditoria["sincronizacao"]["atualizados"] += 1
                self.auditoria["sincronizacao"]["avaliados"] += 1
                if o >= 450: l.commit(); l, o = self.db.batch(), 0
            for rid in ats.keys():
                if rid not in vs:
                    l.delete(col.document(rid)); o += 1; self.auditoria["sincronizacao"]["removidos"] += 1
                    if o >= 450: l.commit(); l, o = self.db.batch(), 0
            if o > 0: l.commit()

    def rodar(self):
        try:
            m = pd.DataFrame()
            anos = range(2022, self.inicio.year + 1) if self.auditoria["modo"] == "HARD_RESET" else [self.inicio.year]
            for a in anos:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{a}.xlsx"
                r = self.session.get(url, timeout=600)
                if r.status_code == 200:
                    df = self._extrair(r.content)
                    self.auditoria["volumes"]["raw"] += len(df)
                    df.to_parquet(f'datalake/raw/ssp_{a}.parquet', index=False)
                    t, d = self._qualificar(df, a)
                    if not d.empty: m = pd.concat([m, d], ignore_index=True)
            if not m.empty:
                p, b = self._modelar(m)
                self._finalizar(p, b)
            with open(self.lock, 'w') as f: f.write(str(datetime.now()))
            self._notificar(True)
        except Exception as e: self._notificar(False, e); raise

if __name__ == "__main__":
    MotorSafeDriver().rodar()
