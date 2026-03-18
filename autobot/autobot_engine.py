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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler("datalake/audit_log.log"), logging.StreamHandler()])

class AutobotSafeDriver:
    def __init__(self, persistencia=True):
        self.id = "AUTOBOT-SD-OMEGA"
        self.persistencia = persistencia
        self.db = self._init_db() if persistencia else None
        self.session = self._init_session()
        self.audit = {
            "camadas": {"bruta": 0, "confiavel": 0, "refinada": 0},
            "stats": {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "acc": 0.0},
            "nuvem": {"total": 0, "delta": 0}
        }
        for p in ['bruto', 'confiavel', 'refinado']: os.makedirs(f'datalake/{p}', exist_ok=True)

    def _init_db(self):
        cfg = os.environ.get('FIREBASE_JSON')
        if cfg and not firebase_admin._apps:
            try:
                cred = credentials.Certificate(json.loads(cfg))
                firebase_admin.initialize_app(cred)
                return firestore.client()
            except: return None
        return None

    def _init_session(self):
        s = requests.Session()
        r = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=r))
        return s

    def _clean(self, t):
        if pd.isna(t) or not isinstance(t, str): return ""
        return "".join([c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c)]).upper().strip()

    def _weight(self, row):
        txt = " ".join([str(v) for v in row.values if pd.api.types.is_scalar(v) and pd.notnull(v)])
        t = self._clean(txt)
        if any(w in t for w in ["LATROCINIO", "SEQUESTRO"]): return 10.0
        if "ROUBO" in t and any(w in t for w in ["VEICULO", "CARRO", "MOTO", "AUTO"]): return 8.5
        if "ROUBO" in t and "CARGA" in t: return 8.0
        if "ROUBO" in t and any(w in t for w in ["CELULAR", "PEDESTRE"]): return 7.5
        if "FURTO" in t and any(w in t for w in ["VEICULO", "CARRO", "MOTO"]): return 4.0
        if "ROUBO" in t: return 6.0
        if "FURTO" in t: return 3.0
        return 1.0

    def _process(self, df):
        limit = 50 if self.persistencia else 2
        if len(df) < limit: return pd.DataFrame()
        df['h3'] = df.apply(lambda r: h3.latlng_to_cell(float(r['LATITUDE']), float(r['LONGITUDE']), 10), axis=1)
        df['peso'] = df.apply(self._weight, axis=1)
        
        map_p = {"Pedestre": ["CELULAR", "ONIBUS", "PEDESTRE"], "Motorista": ["VEICULO", "CARGA", "CARRO"], "Ciclista": ["BICI"], "Motociclista": ["MOTO"]}
        def get_p(r):
            t = self._clean(" ".join([str(v) for v in r.values if pd.api.types.is_scalar(v)]))
            f = [p for p, words in map_p.items() if any(w in t for w in words)]
            return f if f else ["Geral"]
        
        df['perfis'] = df.apply(get_p, axis=1)
        df = df.explode('perfis')
        
        def get_t(h):
            try:
                h = int(str(h).split(':')[0])
                return 'Madrugada' if 0<=h<6 else 'Manha' if 6<=h<12 else 'Tarde' if 12<=h<18 else 'Noite'
            except: return 'Noite'
        df['turno'] = df['HORA_OCORRENCIA_BO'].apply(get_t)
        
        enc = LabelEncoder()
        df['t_c'] = enc.fit_transform(df['turno'])
        X, y = df[['LATITUDE', 'LONGITUDE', 't_c']], df['peso']
        
        if len(X) >= 10:
            xt, xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
            mdl = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05).fit(xt, yt)
            yp = mdl.predict(xv)
            mae, rmse, r2 = mean_absolute_error(yv, yp), np.sqrt(mean_squared_error(yv, yp)), r2_score(yv, yp)
        else: mae, rmse, r2 = 0.0, 0.0, 1.0
        
        self.audit['stats'] = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4), "acc": round(max(0.0, 100.0 - ((mae/10)*100)), 2)}
        
        res = df.groupby(['h3', 'perfis', 'turno']).agg({'peso': ['mean', 'count']}).reset_index()
        res.columns = ['h3', 'perfil', 'turno', 'peso_m', 'freq']
        res['pt'] = MinMaxScaler(feature_range=(0.5, 10.0)).fit_transform(res[['peso_m']]).round(1) if len(res)>1 else 5.0
        res['pn'] = (1 + (res['pt'] * 0.15)).round(2)
        return res

    def _sync(self, df):
        if not self.db or df.empty: return
        path_old = 'datalake/refinado/malha_final.parquet'
        df_delta = df.copy()
        if os.path.exists(path_old):
            try:
                old = pd.read_parquet(path_old)
                df['sig'] = df['h3']+df['perfil']+df['turno']+df['pt'].astype(str)
                old['sig'] = old['h3']+old['perfil']+old['turno']+old['pt'].astype(str)
                df_delta = df[~df['sig'].isin(old['sig'])].copy()
                df.drop(columns=['sig'], inplace=True)
            except: pass
        
        batch = self.db.batch()
        col = self.db.collection('malha_seguranca')
        c, s = 0, 0
        for _, r in df_delta.iterrows():
            # Estrutura otimizada: Perfil_H3 como documento, Scores como Map interno
            batch.set(col.document(f"{r['perfil'].lower()}_{r['h3']}"), {
                "id_h3": r['h3'], 
                "perfil": r['perfil'],
                f"scores.{r['turno']}": {"pt": r['pt'], "pn": r['pn'], "ts": firestore.SERVER_TIMESTAMP}
            }, merge=True)
            c += 1; s += 1
            if c >= 450: batch.commit(); batch = self.db.batch(); c = 0
        if c > 0: batch.commit()
        self.audit['nuvem'] = {"total": len(df), "delta": s}

    def run(self):
        try:
            m = pd.DataFrame()
            meta_p, meta = 'datalake/bruto/metadata.json', {}
            if os.path.exists(meta_p):
                with open(meta_p, 'r') as f: meta = json.load(f)
            
            for a in range(2024, 2027):
                p = f'datalake/bruto/ssp_{a}.parquet'
                u = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{a}.xlsx"
                try:
                    h = self.session.head(u, timeout=20)
                    size = int(h.headers.get('Content-Length', 0))
                    if os.path.exists(p) and meta.get(str(a)) == size: d = pd.read_parquet(p)
                    else:
                        r = self.session.get(u, timeout=300)
                        d = pd.read_excel(io.BytesIO(r.content), dtype=str)
                        d.columns = [self._clean(c) for c in d.columns]
                        d.to_parquet(p, index=False); meta[str(a)] = size
                except: d = pd.read_parquet(p) if os.path.exists(p) else pd.DataFrame()
                if not d.empty:
                    self.audit['camadas']['bruta'] += len(d)
                    m = pd.concat([m, d])
            
            if not m.empty:
                res = self._process(m)
                if not res.empty:
                    self.audit['camadas']['refinada'] = len(res)
                    self._sync(res)
                    res.to_parquet('datalake/refinado/malha_final.parquet', index=False)
                    # Exportação para BI com Timestamp
                    res['dt_proc'] = datetime.now()
                    res.to_csv('datalake/refinado/bi_export.csv', index=False)
                    with open(meta_p, 'w') as f: json.dump(meta, f)
                    self._report(True)
        except Exception as e:
            self._report(False, str(e))

    def _report(self, ok, err=None):
        url = os.environ.get('DISCORD_SUCESSO')
        if not url: return
        if ok:
            t, d = self.audit['nuvem']['total'], self.audit['nuvem']['delta']
            p = ((t-d)/t*100) if t>0 else 0
            body = {"embeds": [{"title": f"🤖 {self.id} | CICLO CONCLUÍDO", "color": 3066993, "fields": [
                {"name": "📊 MÉTRICAS IA", "value": f"Precisão: **{self.audit['stats']['acc']}%**\n$R^2$: **{self.audit['stats']['r2']}**\n$MAE$: **{self.audit['stats']['mae']}**", "inline": False},
                {"name": "⚙️ PROCESSAMENTO", "value": f"Input: {self.audit['camadas']['bruta']}\nOutput: {self.audit['camadas']['refinada']}", "inline": True},
                {"name": "☁️ CLOUD SYNC", "value": f"Delta: {d}\nEconomia: **{p:.1f}%**", "inline": True}]}]}
        else:
            body = {"embeds": [{"title": f"🚨 {self.id} | ERRO DE SISTEMA", "color": 15158332, "description": f"Stacktrace: `{err}`"}]}
        requests.post(url, json=body)

if __name__ == "__main__":
    AutobotSafeDriver().run()
