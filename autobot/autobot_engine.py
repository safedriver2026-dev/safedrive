import io
import os
import json
import math
import hashlib
import logging
import unicodedata
from datetime import datetime
import numpy as np
import pandas as pd
import pygeohash as gh
import requests
import firebase_admin
import xgboost as xgb
from firebase_admin import credentials, firestore
from prophet import Prophet
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from autobot.config import (
    CATALOGO_CRIMES, TIPOS_LOCAL_PERMITIDOS, LIMITES_SP, ESQUEMA_RAW_CANONICO, 
    ESQUEMA_TRUSTED, COLUNAS_REFINED_EVENTOS, PALAVRAS_CHAVE_PERFIL, MAPA_SEMANTICO_COLUNAS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class MotorSafeDriver:
    def __init__(self, habilitar_firestore=True):
        self.data_execucao = pd.Timestamp(datetime.now().date())
        self.janela_inicio = self.data_execucao - pd.Timedelta(days=730)
        self.periodo_historico = range(self.janela_inicio.year, self.data_execucao.year + 1)
        self.sessao = self._criar_sessao()
        self.db = self._conectar_firebase() if habilitar_firestore else None
        self.auditoria = {"volume_raw": 0, "mae": 0.0, "rmse": 0.0, "sync": 0}
        for p in ['raw', 'trusted', 'refined', 'metadata', 'reports']: os.makedirs(f'datalake/{p}', exist_ok=True)

    def _conectar_firebase(self):
        ch = os.environ.get('FIREBASE_JSON')
        if not ch or not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(ch))
            firebase_admin.initialize_app(cred)
        return firestore.client()

    def _criar_sessao(self):
        s = requests.Session()
        ret = Retry(total=5, backoff_factor=2, status_forcelist=[403, 429, 500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=ret))
        return s

    def _limpar_texto(self, t):
        if pd.isna(t): return ""
        n = unicodedata.normalize('NFKD', str(t))
        return "".join([c for c in n if not unicodedata.combining(c)]).upper().strip()

    def _ingerir_excel(self, content):
        excel = pd.ExcelFile(io.BytesIO(content))
        abas_dados = [s for i, s in enumerate(excel.sheet_names) if i > 0]
        dfs = []
        for aba in abas_dados:
            df = excel.parse(aba, dtype=str)
            df.columns = [self._limpar_texto(c) for c in df.columns]
            for can, aliases in MAPA_SEMANTICO_COLUNAS.items():
                match = [a for a in aliases if a in df.columns]
                if match: df[can] = df[match[0]]
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _etrar_trusted(self, df_raw, ano):
        df = df_raw.copy()
        df['ANO_BASE'] = int(ano)
        df['LATITUDE'] = df['LATITUDE'].replace(['0', 0, '-', '0.0'], np.nan)
        df['LONGITUDE'] = df['LONGITUDE'].replace(['0', 0, '-', '0.0'], np.nan)
        for col, tipo in ESQUEMA_TRUSTED.items():
            if col not in df.columns: continue
            if tipo == 'float': df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            elif tipo == 'datetime': df[col] = pd.to_datetime(df[col], errors='coerce')
            elif tipo == 'int': df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'DATA_OCORRENCIA_BO'])
        df['DATA_OCORRENCIA_BO'] = df['DATA_OCORRENCIA_BO'].dt.normalize()
        mask = (df['DATA_OCORRENCIA_BO'].between(self.janela_inicio, self.data_execucao)) & \
               (df['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1]))
        df_t = df[mask].copy()
        df_r = df_t[df_t['NATUREZA_APURADA'].isin(CATALOGO_CRIMES.keys())][COLUNAS_REFINED_EVENTOS].copy()
        return df_t, df_r

    def _ml_features(self, df_r):
        df = df_r.copy()
        df['perfis'] = df.apply(lambda x: [p for p, kws in PALAVRAS_CHAVE_PERFIL.items() if any(k in str(x).upper() for k in kws)] or ['Indefinido'], axis=1)
        df = df.explode('perfis')
        df['gh'] = [gh.encode(la, lo, precision=7) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]
        df['hr'] = df['HORA_OCORRENCIA_BO'].str[:2].replace('', '00').astype(int)
        df['turno'] = df['hr'].apply(lambda h: 'Madrugada' if 0<=h<6 else 'Manha' if 6<=h<12 else 'Tarde' if 12<=h<18 else 'Noite')
        df['w'] = df['NATUREZA_APURADA'].apply(lambda x: CATALOGO_CRIMES.get(x, {}).get('peso', 1.0))
        pnl = df.groupby(['gh', 'perfis', 'turno', 'DATA_OCORRENCIA_BO'], as_index=False).agg(y=('w', 'sum'), la=('LATITUDE', 'mean'), lo=('LONGITUDE', 'mean'))
        pnl = pnl.sort_values(['gh', 'perfis', 'turno', 'DATA_OCORRENCIA_BO'])
        pnl['l7'] = pnl.groupby(['gh', 'perfis', 'turno'])['y'].shift(7).fillna(0)
        pnl['target'] = pnl.groupby(['gh', 'perfis', 'turno'])['y'].transform(lambda x: x.shift(-7).rolling(7, min_periods=1).sum())
        macro = pnl.groupby('DATA_OCORRENCIA_BO')['y'].sum().reset_index().rename(columns={'DATA_OCORRENCIA_BO': 'ds', 'y': 'y'})
        m_p = Prophet(yearly_seasonality=True, weekly_seasonality=True).fit(macro)
        prj = m_p.predict(m_p.make_future_dataframe(periods=14))[['ds', 'yhat']]
        pnl = pnl.merge(prj, left_on='DATA_OCORRENCIA_BO', right_on='ds', how='left')
        pnl['ft'] = pnl['yhat'] / max(pnl['yhat'].mean(), 1.0)
        return pnl, prj

    def _treinar_sync(self, pnl, prj):
        pnl_v = pnl.dropna(subset=['target']).copy()
        le_p, le_t = LabelEncoder().fit(pnl['perfis']), LabelEncoder().fit(pnl['turno'])
        for d in [pnl_v, pnl]:
            d['pe'], d['te'] = le_p.transform(d['perfis']), le_t.transform(d['turno'])
        fs = ['la', 'lo', 'l7', 'ft', 'pe', 'te']
        md = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05).fit(pnl_v[fs], pnl_v['target'])
        grid = pnl.sort_values('DATA_OCORRENCIA_BO').groupby(['gh', 'perfis', 'turno']).tail(1).copy()
        grid['ft'] = prj.iloc[-1]['yhat'] / max(prj['yhat'].mean(), 1.0)
        grid['sp'] = np.clip(md.predict(grid[fs]), 0, None)
        esc = max(grid['sp'].quantile(0.95), 1.0)
        grid['sn'] = ((grid['sp'] / esc) * 10).clip(0.5, 10.0).round(2)
        grid['rp'] = (1.0 + (grid['sn'] * 0.2)).round(2)
        if self.db:
            cl = self.db.collection('niveis_risco')
            bt = self.db.batch()
            for _, l in grid.iterrows():
                did = f"{l['gh']}_{l['perfis']}_{l['te']}"
                bt.set(cl.document(did), {'sc': float(l['sn']), 'rp': float(l['rp']), 'gh': l['gh'], 'g4': l['gh'][:4], 'pf': l['perfis'], 'ua': firestore.SERVER_TIMESTAMP}, merge=True)
            bt.commit()

    def executar_pipeline_completo(self):
        df_m = pd.DataFrame()
        for a in self.periodo_historico:
            res = self.sessao.get(f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{a}.xlsx")
            if res.status_code == 200:
                df_r = self._ingerir_excel(res.content)
                self.auditoria['volume_raw'] += len(df_r)
                dt, dr = self._etrar_trusted(df_r, a)
                df_m = pd.concat([df_m, dr])
        pnl, prj = self._ml_features(df_m)
        self._treinar_sync(pnl, prj)
        with open(f"datalake/reports/runbook_{datetime.now().strftime('%Y%m%d')}.md", 'w') as f: f.write(f"# SafeDriver\nRAW: {self.auditoria['volume_raw']}")

if __name__ == "__main__": MotorSafeDriver().executar_pipeline_completo()
