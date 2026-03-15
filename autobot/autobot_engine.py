import io, os, json, math, shutil, hashlib, logging, unicodedata, warnings, difflib
from datetime import datetime
import numpy as np
import pandas as pd
import pygeohash as gh
import requests, firebase_admin
import xgboost as xgb
from firebase_admin import credentials, firestore
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import DBSCAN
from autobot.config import *

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class MotorSafeDriver:
    def __init__(self, habilitar_firestore=True):
        self.ts_execucao = datetime.now()
        self.data_execucao = pd.Timestamp(self.ts_execucao.date())
        self.janela_inicio = self.data_execucao - pd.Timedelta(days=730)
        self.lock_file = 'datalake/metadata/baseline.lock'
        self.db = self._conectar_firebase() if habilitar_firestore else None
        self.auditoria = {"modo": "INCREMENTAL", "volume_raw": 0, "dq_latlon": 0, "clusters_ativos": 0, "mae": 0.0}
        
        # COLD START: Se não há lock, zera a RAW e reconstrói
        if not os.path.exists(self.lock_file):
            self.auditoria["modo"] = "FULL_RELOAD"
            if os.path.exists('datalake'): shutil.rmtree('datalake', ignore_errors=True)
        
        for p in ['raw', 'trusted', 'refined', 'metadata', 'reports']: os.makedirs(f'datalake/{p}', exist_ok=True)

    def _conectar_firebase(self):
        ch = os.environ.get('FIREBASE_JSON')
        if not ch or not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(ch))
            firebase_admin.initialize_app(cred)
        return firestore.client()

    def _higienizar(self, t):
        if pd.isna(t): return ""
        n = unicodedata.normalize('NFKD', str(t))
        return "".join([c for c in n if not unicodedata.combining(c)]).upper().strip()

    def _normalizar_coluna(self, col):
        col_limpa = self._higienizar(col).replace(" ", "_")
        for can, aliases in MAPA_SEMANTICO_COLUNAS.items():
            if col_limpa in aliases: return can
        return col_limpa

    def _formatar_data_pt(self, dt):
        meses = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro']
        return f"{dt.day} de {meses[dt.month - 1]} de {dt.year} às {dt.strftime('%H:%M')}"

    def _notificar_discord(self, sucesso=True, erro=None):
        webhook = os.environ.get('DISCORD_SUCESSO') if sucesso else os.environ.get('DISCORD_ERRO')
        if not webhook: return
        embed = {
            "title": f"🚀 SafeDriver City-Scale Engine: {self.auditoria['modo']} Concluído",
            "color": 3066993 if sucesso else 15158332,
            "fields": [
                {"name": "📊 DQ e Volume", "value": f"RAW: {self.auditoria['volume_raw']:,}\nLatLon Válida: {self.auditoria['dq_latlon']:.1f}%", "inline": True},
                {"name": "🧠 Inteligência", "value": f"Clusters: {self.auditoria['clusters_ativos']}\nMAE: {self.auditoria['mae']}", "inline": True}
            ] if sucesso else [{"name": "Erro", "value": str(erro)}],
            "footer": {"text": f"{self._formatar_data_pt(datetime.now())}"}
        }
        requests.post(webhook, json={"embeds": [embed]})

    def _ingerir_fonte(self, content):
        excel = pd.ExcelFile(io.BytesIO(content))
        dfs = []
        for aba in excel.sheet_names:
            if any(x in aba.upper() for x in ["CAMPOS", "METADADOS", "DICIONARIO"]): continue
            df = excel.parse(aba, dtype=str)
            df.columns = [self._normalizar_coluna(c) for c in df.columns]
            
            # RESOLUÇÃO PARA DUPLICIDADE (Deduplicação por Fusão)
            df = df.loc[:, ~df.columns.duplicated()].copy()
            
            for col in ESQUEMA_RAW_CANONICO.keys():
                if col not in df.columns: df[col] = np.nan
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _qualificar_dados(self, df_raw, ano):
        df = df_raw.copy()
        df['ANO_BASE'] = int(ano)
        valid_latlon = df['LATITUDE'].replace(['0', 0, '-', ' '], np.nan).notna().sum()
        self.auditoria['dq_latlon'] = (valid_latlon / max(len(df), 1)) * 100
        
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'DATA_OCORRENCIA_BO'])
        mask = (df['DATA_OCORRENCIA_BO'].between(self.janela_inicio, self.data_execucao)) & \
               (df['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1]))
        
        df_t = df[mask].copy()
        df_r = df_t[df_t['NATUREZA_APURADA'].isin(CATALOGO_CRIMES.keys())].copy()
        return df_t, df_r

    def _gerar_inteligencia(self, df_r):
        if df_r.empty: return pd.DataFrame(), None
        df = df_r.copy()
        
        # PESO POR RECÊNCIA E FEATURES ESPACIAIS
        df['dias_desde_evento'] = (self.data_execucao - df['DATA_OCORRENCIA_BO']).dt.days
        df['peso_recencia'] = np.exp(-df['dias_desde_evento'] / 180)
        df['hora'] = df['HORA_OCORRENCIA_BO'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)
        df['periodo_dia'] = df['hora'].apply(lambda h: 'Noite' if h>=18 or h<6 else 'Dia')
        df['gh'] = [gh.encode(la, lo, precision=7) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]
        df['peso_gravidade'] = df['NATUREZA_APURADA'].apply(lambda x: CATALOGO_CRIMES.get(x, {}).get('peso', 1.0))
        df['score_evento'] = df['peso_recencia'] * df['peso_gravidade']
        
        # CLUSTERS DBSCAN (Hotspots emergentes)
        clustering = DBSCAN(eps=0.005, min_samples=10).fit(df[['LATITUDE', 'LONGITUDE']])
        df['cluster_id'] = clustering.labels_
        self.auditoria['clusters_ativos'] = len(set(clustering.labels_)) - 1
        
        pnl = df.groupby(['gh', 'periodo_dia', 'DATA_OCORRENCIA_BO'], as_index=False).agg(
            y=('score_evento', 'sum'), vol_crimes=('NUM_BO', 'count'), la=('LATITUDE', 'mean'), lo=('LONGITUDE', 'mean'), cluster_id=('cluster_id', 'max')
        )
        pnl['vol_365d'] = pnl.groupby('gh')['vol_crimes'].transform('sum')
        pnl['target'] = pnl.groupby(['gh', 'periodo_dia'])['y'].transform(lambda x: x.shift(-7).rolling(7, min_periods=1).sum())
        
        macro = pnl.groupby('DATA_OCORRENCIA_BO')['vol_crimes'].sum().reset_index().rename(columns={'DATA_OCORRENCIA_BO': 'ds', 'vol_crimes': 'y'})
        if len(macro) >= 2:
            m_p = Prophet().fit(macro)
            prj = m_p.predict(m_p.make_future_dataframe(periods=14))[['ds', 'yhat']]
            pnl = pnl.merge(prj, left_on='DATA_OCORRENCIA_BO', right_on='ds', how='left')
            pnl['ft'] = pnl['yhat'] / max(pnl['yhat'].mean(), 1.0)
        else: pnl['ft'] = 1.0; prj = pd.DataFrame({'ds': [self.data_execucao], 'yhat': [1.0]})
            
        return pnl, prj

    def _treinar_disseminar(self, pnl, prj):
        pnl_v = pnl.dropna(subset=['target']).copy()
        if pnl_v.empty: return
        
        fs = ['la', 'lo', 'vol_365d', 'ft', 'cluster_id']
        md = xgb.XGBRegressor(n_estimators=100).fit(pnl_v[fs], pnl_v['target'])
        self.auditoria['mae'] = round(float(mean_absolute_error(pnl_v['target'], md.predict(pnl_v[fs]))), 3)

        grid = pnl.sort_values('DATA_OCORRENCIA_BO').groupby(['gh', 'periodo_dia']).tail(1).copy()
        grid['ft'] = prj.iloc[-1]['yhat'] / max(prj['yhat'].mean(), 1.0)
        grid['score_raw'] = np.clip(md.predict(grid[fs]), 0, None)
        
        # SUAVIZAÇÃO E CATEGORIZAÇÃO (MASTIGADO PARA FLUTTER)
        scaler = MinMaxScaler(feature_range=(0, 100))
        grid['risk_score'] = scaler.fit_transform(grid[['score_raw']]).round(1)
        grid['routing_penalty'] = (1.0 + (grid['risk_score'] / 50.0)).round(2)
        
        if self.db:
            cl = self.db.collection('niveis_risco')
            bt, ops = self.db.batch(), 0
            for _, l in grid.iterrows():
                did = f"{l['gh']}_{l['periodo_dia']}"
                payload = {
                    'risk_score': float(l['risk_score']),
                    'routing_penalty': float(l['routing_penalty']),
                    'geohash': l['gh'],
                    'geohash_prefix_4': l['gh'][:4],
                    'periodo': l['periodo_dia'],
                    'ultima_atualizacao': firestore.SERVER_TIMESTAMP
                }
                bt.set(cl.document(did), payload, merge=True)
                ops += 1
                if ops >= 450: bt.commit(); bt, ops = self.db.batch(), 0
            if ops > 0: bt.commit()

    def executar_pipeline_completo(self):
        try:
            df_master = pd.DataFrame()
            anos = range(2022, self.data_execucao.year + 1) if self.auditoria["modo"] == "FULL_RELOAD" else [self.data_execucao.year]
            for a in anos:
                res = requests.get(f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{a}.xlsx")
                if res.status_code == 200:
                    df_r = self._ingerir_fonte(res.content)
                    self.auditoria['volume_raw'] += len(df_r)
                    df_r.to_parquet(f'datalake/raw/ssp_{a}.parquet', index=False)
                    dt, dr = self._qualificar_dados(df_r, a)
                    if not dr.empty: df_master = pd.concat([df_master, dr])
            
            if not df_master.empty:
                pnl, prj = self._gerar_inteligencia(df_master)
                self._treinar_disseminar(pnl, prj)
            with open(self.lock_file, 'w') as f: f.write(str(datetime.now()))
            self._notificar_discord(sucesso=True)
        except Exception as e: self._notificar_discord(sucesso=False, erro=e); raise

if __name__ == "__main__": MotorSafeDriver().executar_pipeline_completo()
