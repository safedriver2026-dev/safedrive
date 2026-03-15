import io
import os
import json
import math
import shutil
import hashlib
import logging
import unicodedata
import warnings
import difflib
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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import DBSCAN
from autobot.config import (
    CATALOGO_CRIMES, LIMITES_SP, ESQUEMA_RAW_CANONICO, 
    COLUNAS_REFINED_EVENTOS, PALAVRAS_CHAVE_PERFIL, MAPA_SEMANTICO_COLUNAS
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class MotorSafeDriver:
    def __init__(self, habilitar_firestore=True):
        self.ts_execucao = datetime.now()
        self.data_execucao = pd.Timestamp(self.ts_execucao.date())
        self.janela_inicio = self.data_execucao - pd.Timedelta(days=730)
        self.periodo_historico = range(self.janela_inicio.year, self.data_execucao.year + 1)
        self.lock_file = 'datalake/metadata/baseline.lock'
        self.sessao = self._criar_sessao()
        self.db = self._conectar_firebase() if habilitar_firestore else None
        
        self.auditoria = {
            "modo": "INCREMENTAL", "volume_raw": 0, "volume_trusted": 0, "volume_refined": 0,
            "mae": 0.0, "rmse": 0.0, "docs_atualizados": 0, "docs_removidos": 0,
            "dq_latlon": 0, "dq_hora": 0, "clusters_ativos": 0
        }
        
        # LÓGICA DE COLD START: Se não houver lock, apaga tudo e reconstrói
        if not os.path.exists(self.lock_file):
            self.auditoria["modo"] = "FULL_RELOAD"
            if os.path.exists('datalake'): 
                shutil.rmtree('datalake', ignore_errors=True)
        
        for p in ['raw', 'trusted', 'refined', 'metadata', 'reports']: 
            os.makedirs(f'datalake/{p}', exist_ok=True)

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

    def _higienizar(self, t):
        if pd.isna(t): return ""
        n = unicodedata.normalize('NFKD', str(t))
        return "".join([c for c in n if not unicodedata.combining(c)]).upper().strip()

    def _normalizar_coluna(self, col):
        col_limpa = self._higienizar(col).replace(" ", "_")
        for can, aliases in MAPA_SEMANTICO_COLUNAS.items():
            matches = difflib.get_close_matches(col_limpa, aliases, n=1, cutoff=0.8)
            if matches or col_limpa in aliases: return can
        return col_limpa

    def _formatar_data_pt(self, dt):
        meses = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro']
        return f"{dt.day} de {meses[dt.month - 1]} de {dt.year} às {dt.strftime('%H:%M')}"

    def _notificar_discord(self, sucesso=True, erro=None):
        webhook = os.environ.get('DISCORD_SUCESSO') if sucesso else os.environ.get('DISCORD_ERRO')
        if not webhook: return
        
        if sucesso:
            embed = {
                "title": f"🚀 SafeDriver City-Scale Engine: {self.auditoria['modo']} Concluído",
                "color": 3066993,
                "fields": [
                    {"name": "📊 Qualidade de Dados (DQ)", "value": f"Registros Lidos: {self.auditoria['volume_raw']:,}\nCom Lat/Lon Válida: {self.auditoria['dq_latlon']:.1f}%\nCom Hora Válida: {self.auditoria['dq_hora']:.1f}%", "inline": False},
                    {"name": "🧠 Performance e Clusters", "value": f"Hotspots Emergentes (DBSCAN): {self.auditoria['clusters_ativos']}\nErro MAE Espacial: {self.auditoria['mae']}", "inline": False},
                    {"name": "☁️ Sincronização Firebase", "value": f"Mutações: {self.auditoria['docs_atualizados']:,} | Removidos: {self.auditoria['docs_removidos']:,}", "inline": False}
                ],
                "footer": {"text": f"Duração: {(datetime.now() - self.ts_execucao).total_seconds():.1f}s • {self._formatar_data_pt(datetime.now())}"}
            }
        else:
            embed = {
                "title": "🚨 Falha Crítica no Motor SafeDriver", "color": 15158332,
                "description": f"Ocorreu um erro irrecuperável.\n\n**Traceback:**\n```{str(erro)}```",
                "footer": {"text": self._formatar_data_pt(datetime.now())}
            }
        requests.post(webhook, json={"embeds": [embed]})

    def _ingerir_fonte(self, content):
        excel = pd.ExcelFile(io.BytesIO(content))
        dfs = []
        for aba in excel.sheet_names:
            if any(x in aba.upper() for x in ["CAMPOS", "METADADOS", "DICIONARIO", "LEGENDA"]): continue
            df = excel.parse(aba, dtype=str)
            
            # NORMALIZAÇÃO SEMÂNTICA
            df.columns = [self._normalizar_coluna(c) for c in df.columns]
            
            # SOLUÇÃO PARA DUPLICADOS: Agrupar colunas com mesmo nome e fundir os dados
            # Se houver duas colunas 'NUM_BO', pegamos a informação da primeira que não for nula
            df = df.groupby(level=0, axis=1).first()
            
            # Garantir colunas mínimas
            for col in ESQUEMA_RAW_CANONICO.keys():
                if col not in df.columns: df[col] = np.nan
            dfs.append(df)
            
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _qualificar_dados(self, df_raw, ano):
        df = df_raw.copy()
        df['ANO_BASE'] = int(ano)
        
        total_rows = max(len(df), 1)
        valid_latlon = df['LATITUDE'].notna().sum()
        valid_hora = df['HORA_OCORRENCIA_BO'].notna().sum()
        self.auditoria['dq_latlon'] = (valid_latlon / total_rows) * 100
        self.auditoria['dq_hora'] = (valid_hora / total_rows) * 100

        df['LATITUDE'] = df['LATITUDE'].replace(['0', 0, '-', '0.0', ' ', 'NULL'], np.nan)
        df['LONGITUDE'] = df['LONGITUDE'].replace(['0', 0, '-', '0.0', ' ', 'NULL'], np.nan)
        
        for col, tipo in ESQUEMA_RAW_CANONICO.items():
            if col not in df.columns: df[col] = np.nan
            if tipo == 'float': df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
            elif tipo == 'datetime': df[col] = pd.to_datetime(df[col], errors='coerce')
            else: df[col] = df[col].astype(str).apply(self._higienizar)
            
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'DATA_OCORRENCIA_BO'])
        df['DATA_OCORRENCIA_BO'] = df['DATA_OCORRENCIA_BO'].dt.normalize()
        
        mask = (df['DATA_OCORRENCIA_BO'].between(self.janela_inicio, self.data_execucao)) & \
               (df['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1])) & \
               (df['LONGITUDE'].between(LIMITES_SP['lon'][0], LIMITES_SP['lon'][1]))
               
        df_t = df[mask].copy()
        self.auditoria['volume_trusted'] += len(df_t)
        
        df_r = df_t[df_t['NATUREZA_APURADA'].isin(CATALOGO_CRIMES.keys())].copy()
        if not df_r.empty: 
            df_r = df_r[COLUNAS_REFINED_EVENTOS].copy()
            self.auditoria['volume_refined'] += len(df_r)
        return df_t, df_r

    def _gerar_inteligencia(self, df_r):
        if df_r.empty: return pd.DataFrame(), None
        df = df_r.copy()
        
        # PESO POR RECÊNCIA E FEATURES TEMPORAIS
        df['dias_desde_evento'] = (self.data_execucao - df['DATA_OCORRENCIA_BO']).dt.days
        df['peso_recencia'] = np.exp(-df['dias_desde_evento'] / 180)
        df['hora'] = df['HORA_OCORRENCIA_BO'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)
        df['periodo_dia'] = df['hora'].apply(lambda h: 'Madrugada' if 0<=h<6 else 'Manha' if 6<=h<12 else 'Tarde' if 12<=h<18 else 'Noite')
        
        df['perfis'] = df.apply(lambda x: [p for p, kws in PALAVRAS_CHAVE_PERFIL.items() if any(k in str(x).upper() for k in kws)] or ['Indefinido'], axis=1)
        df = df.explode('perfis')
        df['gh'] = [gh.encode(la, lo, precision=7) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]
        
        df['peso_gravidade'] = df['NATUREZA_APURADA'].apply(lambda x: CATALOGO_CRIMES.get(x, {}).get('peso', 1.0))
        df['score_evento'] = df['peso_recencia'] * df['peso_gravidade']
        
        # CLUSTERS DBSCAN
        coords = df[['LATITUDE', 'LONGITUDE']].values
        clustering = DBSCAN(eps=0.005, min_samples=15).fit(coords)
        df['cluster_id'] = clustering.labels_
        self.auditoria['clusters_ativos'] = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        pnl = df.groupby(['gh', 'perfis', 'periodo_dia', 'DATA_OCORRENCIA_BO'], as_index=False).agg(
            y=('score_evento', 'sum'), vol_crimes=('NUM_BO', 'count'),
            la=('LATITUDE', 'mean'), lo=('LONGITUDE', 'mean'), cluster_id=('cluster_id', 'max')
        )
        pnl = pnl.sort_values(['gh', 'perfis', 'periodo_dia', 'DATA_OCORRENCIA_BO'])
        
        # DENSIDADES
        pnl['is_30d'] = ((self.data_execucao - pnl['DATA_OCORRENCIA_BO']).dt.days <= 30).astype(int) * pnl['vol_crimes']
        pnl['is_90d'] = ((self.data_execucao - pnl['DATA_OCORRENCIA_BO']).dt.days <= 90).astype(int) * pnl['vol_crimes']
        pnl['densidade_30d'] = pnl.groupby(['gh'])['is_30d'].transform('sum')
        pnl['densidade_90d'] = pnl.groupby(['gh'])['is_90d'].transform('sum')
        pnl['vol_historico_365d'] = pnl.groupby(['gh'])['vol_crimes'].transform('sum')

        pnl['target'] = pnl.groupby(['gh', 'perfis', 'periodo_dia'])['y'].transform(lambda x: x.shift(-7).rolling(7, min_periods=1).sum())
        
        macro = pnl.groupby('DATA_OCORRENCIA_BO')['vol_crimes'].sum().reset_index().rename(columns={'DATA_OCORRENCIA_BO': 'ds', 'vol_crimes': 'y'})
        if len(macro) >= 2:
            m_p = Prophet(yearly_seasonality=True, weekly_seasonality=True).fit(macro)
            prj = m_p.predict(m_p.make_future_dataframe(periods=14))[['ds', 'yhat']]
            pnl = pnl.merge(prj, left_on='DATA_OCORRENCIA_BO', right_on='ds', how='left')
            pnl['ft'] = pnl['yhat'] / max(pnl['yhat'].mean(), 1.0)
        else:
            pnl['ft'] = 1.0
            prj = pd.DataFrame({'ds': [self.data_execucao], 'yhat': [1.0]})
            
        return pnl, prj

    def _treinar_disseminar(self, pnl, prj):
        if pnl.empty: return
        pnl_v = pnl.dropna(subset=['target']).copy()
        if pnl_v.empty: return
        
        le_p, le_t = LabelEncoder().fit(pnl['perfis']), LabelEncoder().fit(pnl['periodo_dia'])
        for d in [pnl_v, pnl]:
            d['pe'], d['te'] = le_p.transform(d['perfis']), le_t.transform(d['periodo_dia'])
            
        fs = ['la', 'lo', 'densidade_30d', 'densidade_90d', 'ft', 'pe', 'te', 'cluster_id']
        md = xgb.XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.05).fit(pnl_v[fs], pnl_v['target'])
        
        previsoes = np.clip(md.predict(pnl_v[fs]), 0, None)
        self.auditoria['mae'] = round(float(mean_absolute_error(pnl_v['target'], previsoes)), 3)

        grid = pnl.sort_values('DATA_OCORRENCIA_BO').groupby(['gh', 'perfis', 'periodo_dia']).tail(1).copy()
        grid['ft'] = prj.iloc[-1]['yhat'] / max(prj['yhat'].mean(), 1.0)
        grid['xgb_raw'] = np.clip(md.predict(grid[fs]), 0, None)
        
        # SUAVIZAÇÃO ESPACIAL
        gh_scores = grid.groupby('gh')['xgb_raw'].mean().to_dict()
        def get_neighbor_score(g):
            neighbors = gh.neighbors(g)
            scores = [gh_scores.get(n, 0) for n in neighbors]
            return np.mean(scores) if scores else 0
            
        grid['suavizacao'] = grid['gh'].apply(get_neighbor_score)
        grid['score_final'] = (grid['xgb_raw'] * 0.7) + (grid['suavizacao'] * 0.3)
        
        # RISK SCORE 0-100 PARA FLUTTER
        scaler = MinMaxScaler(feature_range=(0, 100))
        grid['risk_score'] = scaler.fit_transform(grid[['score_final']]).round(1)
        grid['routing_penalty'] = (1.0 + (grid['risk_score'] / 50.0)).round(2)
        
        if self.db:
            cl = self.db.collection('niveis_risco')
            documentos_atuais = {doc.id: doc.to_dict().get('hash_registro') for doc in cl.stream()}
            bt, ops, ids_vivos = self.db.batch(), 0, set()
            
            for _, l in grid.iterrows():
                did = f"{l['gh']}_{l['perfis']}_{l['te']}"
                ids_vivos.add(did)
                
                payload = {
                    'risk_score': float(l['risk_score']),
                    'routing_penalty': float(l['routing_penalty']),
                    'geohash': l['gh'],
                    'geohash_prefix_4': l['gh'][:4],
                    'perfil': l['perfis'],
                    'periodo': l['periodo_dia']
                }
                
                hr = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
                payload['hash_registro'] = hr
                payload['ultima_atualizacao'] = firestore.SERVER_TIMESTAMP
                
                if did not in documentos_atuais or documentos_atuais[did] != hr:
                    bt.set(cl.document(did), payload, merge=True)
                    ops += 1
                    self.auditoria['docs_atualizados'] += 1
                    if ops >= 450: bt.commit(); bt, ops = self.db.batch(), 0
                    
            obsoletos = set(documentos_atuais.keys()) - ids_vivos
            for d in obsoletos:
                bt.delete(cl.document(d))
                ops += 1
                self.auditoria['docs_removidos'] += 1
                if ops >= 450: bt.commit(); bt, ops = self.db.batch(), 0
                
            if ops > 0: bt.commit()

    def executar_pipeline_completo(self):
        try:
            df_master = pd.DataFrame()
            anos = self.periodo_historico if self.auditoria["modo"] == "FULL_RELOAD" else [self.data_execucao.year]
            
            for a in anos:
                res = self.sessao.get(f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{a}.xlsx", timeout=120)
                if res.status_code == 200:
                    df_r = self._ingerir_fonte(res.content)
                    self.auditoria['volume_raw'] += len(df_r)
                    
                    # SALVAR RAW SEM DUPLICADOS
                    df_r.to_parquet(f'datalake/raw/ssp_{a}.parquet', index=False)
                    
                    dt, dr = self._qualificar_dados(df_r, a)
                    if not dr.empty: df_master = pd.concat([df_master, dr])
            
            if not df_master.empty:
                pnl, prj = self._gerar_inteligencia(df_master)
                self._treinar_disseminar(pnl, prj)
                
            with open(self.lock_file, 'w') as f: f.write(str(datetime.now()))
            self._gerar_runbook()
            self._notificar_discord(sucesso=True)
            
        except Exception as e:
            self._notificar_discord(sucesso=False, erro=e)
            raise

if __name__ == "__main__":
    MotorSafeDriver().executar_pipeline_completo()
