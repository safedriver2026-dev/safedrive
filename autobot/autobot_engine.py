import pandas as pd
import numpy as np
import os, io, requests, json, unicodedata, gc, re, logging
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import h3
from prophet import Prophet
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler("datalake/logs_operacionais.log"), logging.StreamHandler()])

class SafeDriverEngine:
    def __init__(self, persistence=True):
        self.id = "SD-MEDALLION-BOT"
        self.persistence = persistence
        self.db = self._connect() if persistence else None
        self.session = self._session()
        self.stats = {"bronze": 0, "silver": 0, "gold": 0, "metrics": {}, "cloud": {"total": 0, "delta": 0}}
        for layer in ['bronze_raw', 'silver_trusted', 'gold_refined']: os.makedirs(f'datalake/{layer}', exist_ok=True)

    def _connect(self):
        token = os.environ.get('FIREBASE_JSON')
        if token and not firebase_admin._apps:
            try:
                cred = credentials.Certificate(json.loads(token))
                firebase_admin.initialize_app(cred)
                return firestore.client()
            except: return None
        return None

    def _session(self):
        s = requests.Session()
        r = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=r))
        return s

    def _sanitize(self, t):
        if pd.isna(t) or not isinstance(t, str): return ""
        return "".join([c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c)]).upper().strip()

    def _get_weight(self, row):
        txt = " ".join([str(v) for v in row.values if pd.api.types.is_scalar(v) and pd.notnull(v)])
        t = self._sanitize(txt)
        if any(w in t for w in ["LATROCINIO", "SEQUESTRO"]): return 10.0
        if "ROUBO" in t and any(w in t for w in ["VEICULO", "CARRO", "MOTO", "AUTO"]): return 8.5
        if "ROUBO" in t and "CARGA" in t: return 8.0
        if "ROUBO" in t and any(w in t for w in ["CELULAR", "PEDESTRE", "TRANSEUNTE"]): return 7.5
        if "FURTO" in t and any(w in t for w in ["VEICULO", "CARRO", "MOTO"]): return 4.0
        return 3.0 if "FURTO" in t else 1.0

    def _generate_gold(self, df):
        df['h3_id'] = df.apply(lambda r: h3.latlng_to_cell(float(r['LATITUDE']), float(r['LONGITUDE']), 10), axis=1)
        df['peso'] = df.apply(self._get_weight, axis=1)
        
        profiles = {"Pedestre": ["CELULAR", "ONIBUS", "PEDESTRE"], "Motorista": ["VEICULO", "CARGA", "CARRO"], "Ciclista": ["BICI"], "Motociclista": ["MOTO"]}
        def classify(r):
            t = self._sanitize(" ".join([str(v) for v in r.values if pd.api.types.is_scalar(v)]))
            m = [p for p, words in profiles.items() if any(w in t for w in words)]
            return m if m else ["Geral"]
        
        df['perfil'] = df.apply(classify, axis=1)
        df = df.explode('perfil')
        
        def get_shift(h):
            try:
                h = int(str(h).split(':')[0])
                return 'Madrugada' if 0<=h<6 else 'Manha' if 6<=h<12 else 'Tarde' if 12<=h<18 else 'Noite'
            except: return 'Noite'
        df['turno'] = df['HORA_OCORRENCIA_BO'].apply(get_shift)
        
        le = LabelEncoder()
        df['t_c'] = le.fit_transform(df['turno'])
        X, y = df[['LATITUDE', 'LONGITUDE', 't_c']], df['peso']
        
        if len(X) >= 10:
            xt, xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
            mdl = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05).fit(xt, yt)
            yp = mdl.predict(xv)
            mae, r2 = mean_absolute_error(yv, yp), r2_score(yv, yp)
        else: mae, r2 = 0.0, 1.0
        
        self.stats['metrics'] = {"mae": round(mae, 4), "r2": round(r2, 4), "acc": round(max(0.0, 100.0 - ((mae/10)*100)), 2)}
        
        res = df.groupby(['h3_id', 'perfil', 'turno']).agg({'peso': ['mean', 'count'], 'LATITUDE': 'mean', 'LONGITUDE': 'mean'}).reset_index()
        res.columns = ['h3', 'perfil', 'turno', 'peso_m', 'freq', 'lat', 'lon']
        res['pt'] = MinMaxScaler(feature_range=(0.5, 10.0)).fit_transform(res[['peso_m']]).round(1) if len(res)>1 else 5.0
        res['pn'] = (1 + (res['pt'] * 0.15)).round(2)

        dim_h3 = res[['h3', 'lat', 'lon']].drop_duplicates()
        dim_p = pd.DataFrame({'id_p': range(len(res['perfil'].unique())), 'nome_p': res['perfil'].unique()})
        dim_t = pd.DataFrame({'id_t': range(len(res['turno'].unique())), 'nome_t': res['turno'].unique()})
        
        fato = res.merge(dim_p, left_on='perfil', right_on='nome_p').merge(dim_t, left_on='turno', right_on='nome_t')
        fato = fato[['h3', 'id_p', 'id_t', 'pt', 'pn', 'freq']]
        fato['dt_atu'] = datetime.now()
        
        dim_h3.to_csv('datalake/gold_refined/dim_geometria.csv', index=False)
        dim_p.to_csv('datalake/gold_refined/dim_perfil.csv', index=False)
        dim_t.to_csv('datalake/gold_refined/dim_periodo.csv', index=False)
        fato.to_csv('datalake/gold_refined/fato_risco.csv', index=False)
        
        return res

    def _sync(self, df):
        if not self.db or df.empty: return
        ref = 'datalake/silver_trusted/last_run.parquet'
        delta = df.copy()
        if os.path.exists(ref):
            try:
                h = pd.read_parquet(ref)
                df['hash'] = df['h3'] + df['perfil'] + df['turno'] + df['pt'].astype(str)
                h['hash'] = h['h3'] + h['perfil'] + h['turno'] + h['pt'].astype(str)
                delta = df[~df['hash'].isin(h['hash'])].copy()
                df.drop(columns=['hash'], inplace=True)
            except: pass
            
        b = self.db.batch()
        c, s = 0, 0
        for _, r in delta.iterrows():
            b.set(self.db.collection('malha_risco').document(f"{r['perfil'].lower()}_{r['h3']}"), {
                "h3": r['h3'], "perfil": r['perfil'],
                f"scores.{r['turno']}": {"pt": r['pt'], "pn": r['pn'], "ts": firestore.SERVER_TIMESTAMP}
            }, merge=True)
            c += 1; s += 1
            if c >= 450: b.commit(); b = self.db.batch(); c = 0
        if c > 0: b.commit()
        self.stats['cloud'] = {"total": len(df), "delta": s}

    def start(self):
        try:
            m = pd.DataFrame()
            meta_p, meta = 'datalake/bronze_raw/meta.json', {}
            if os.path.exists(meta_p):
                with open(meta_p, 'r') as f: meta = json.load(f)
            
            for a in range(2023, datetime.now().year + 1):
                p_b = f'datalake/bronze_raw/ssp_{a}.parquet'
                u = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{a}.xlsx"
                try:
                    h = self.session.head(u, timeout=20)
                    sz = int(h.headers.get('Content-Length', 0))
                    if os.path.exists(p_b) and meta.get(str(a)) == sz: d = pd.read_parquet(p_b)
                    else:
                        r = self.session.get(u, timeout=300)
                        d = pd.read_excel(io.BytesIO(r.content), dtype=str)
                        d.columns = [self._sanitize(c) for c in d.columns]
                        d.to_parquet(p_b, index=False); meta[str(a)] = sz
                except: d = pd.read_parquet(p_b) if os.path.exists(p_b) else pd.DataFrame()
                if not d.empty:
                    self.stats['bronze'] += len(d)
                    m = pd.concat([m, d])

            if not m.empty:
                silv = m[m['LATITUDE'].notna()].copy()
                self.stats['silver'] = len(silv)
                gold = self._generate_gold(silv)
                if not gold.empty:
                    self.stats['gold'] = len(gold)
                    self._sync(gold)
                    gold.to_parquet('datalake/silver_trusted/last_run.parquet', index=False)
                    with open(meta_p, 'w') as f: json.dump(meta, f)
                    self._report(True)
        except Exception as e:
            self._report(False, str(e))

    def _report(self, ok, err=None):
        url = os.environ.get('DISCORD_SUCESSO')
        if not url: return
        if ok:
            t, d = self.stats['cloud']['total'], self.stats['cloud']['delta']
            eco = ((t - d) / t * 100) if t > 0 else 0
            msg = {"embeds": [{"title": "⚙️ SISTEMA SAFE-DRIVER: RELATÓRIO DE CICLO", "color": 3066993, "fields": [
                {"name": "📈 MODELAGEM", "value": f"Acurácia: {self.stats['metrics']['acc']}% | R²: {self.stats['metrics']['r2']}", "inline": False},
                {"name": "🏗️ MEDALLION", "value": f"Bronze: {self.stats['bronze']} | Silver: {self.stats['silver']} | Gold: {self.stats['gold']}", "inline": True},
                {"name": "💰 ECONOMIA", "value": f"Escritas: {d} | Economia: **{eco:.1f}%**", "inline": True}]}]}
        else:
            msg = {"embeds": [{"title": "🚨 SISTEMA SAFE-DRIVER: ERRO CRÍTICO", "color": 15158332, "description": f"Erro: `{err}`"}]}
        requests.post(url, json=msg)

if __name__ == "__main__":
    SafeDriverEngine().start()
