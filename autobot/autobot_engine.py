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
    CATALOGO_CRIMES, TIPOS_LOCAL_PERMITIDOS, SUBTIPOS_LOCAL_PERMITIDOS, LIMITES_SP,
    ESQUEMA_RAW_CANONICO, ESQUEMA_TRUSTED, COLUNAS_REFINED_EVENTOS,
    PALAVRAS_CHAVE_PERFIL, MAPA_SEMANTICO_COLUNAS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

class MotorSafeDriver:
    def __init__(self, habilitar_firestore=True, forcar_recarga=False):
        self.data_execucao = pd.Timestamp(datetime.now().date())
        self.janela_inicio = self.data_execucao - pd.Timedelta(days=730)
        self.periodo_historico = range(self.janela_inicio.year, self.data_execucao.year + 1)
        self.forcar_recarga = forcar_recarga
        self.sessao_web = self._criar_sessao_resiliente()
        self.banco_nuvem = self._estabelecer_conexao_nuvem() if habilitar_firestore else None
        
        self.auditoria = {
            "timestamp": str(datetime.now()), "volume_raw": 0, "volume_trusted": 0, 
            "volume_refined": 0, "falhas_integridade": 0, "malha_motorista": 0, 
            "malha_motociclista": 0, "malha_pedestre": 0, "malha_ciclista": 0, 
            "documentos_sincronizados": 0, "documentos_atualizados": 0, 
            "novos_dados_baixados": False, "mae_modelo": 0.0, "rmse_modelo": 0.0
        }
        
        for pasta in ['raw', 'trusted', 'refined', 'metadata', 'reports']:
            os.makedirs(f'datalake/{pasta}', exist_ok=True)

    def _estabelecer_conexao_nuvem(self):
        chave = os.environ.get('FIREBASE_JSON')
        if not chave or not firebase_admin._apps:
            credenciais = credentials.Certificate(json.loads(chave))
            firebase_admin.initialize_app(credenciais)
        return firestore.client()

    def _criar_sessao_resiliente(self):
        sessao = requests.Session()
        retentativas = Retry(total=5, backoff_factor=2, status_forcelist=[403, 429, 500, 502, 503, 504])
        adaptador = HTTPAdapter(max_retries=retentativas)
        sessao.mount("http://", adaptador)
        sessao.mount("https://", adaptador)
        sessao.headers.update({'User-Agent': 'Mozilla/5.0 SafeDriver/8.0'})
        return sessao

    def _notificar_sucesso(self):
        webhook = os.environ.get('DISCORD_SUCESSO')
        if not webhook: return
        payload = {
            "embeds": [{
                "title": "Pipeline SafeDriver MLOps OK",
                "color": 3066993,
                "fields": [
                    {"name": "Métricas", "value": f"MAE: {self.auditoria['mae_modelo']}\nRMSE: {self.auditoria['rmse_modelo']}"},
                    {"name": "Sync", "value": f"Novos: {self.auditoria['documentos_atualizados']}"}
                ]
            }]
        }
        self.sessao_web.post(webhook, json=payload)

    def _notificar_erro(self, erro):
        webhook = os.environ.get('DISCORD_ERRO')
        if not webhook: return
        payload = {"embeds": [{"title": "Falha no Pipeline", "color": 15158332, "description": str(erro)}]}
        requests.post(webhook, json=payload)

    def _higienizar_texto(self, texto):
        if pd.isna(texto): return ""
        norm = unicodedata.normalize('NFKD', str(texto))
        return "".join([c for c in norm if not unicodedata.combining(c)]).upper().strip()

    def _verificar_atualizacao(self, url, ano):
        meta_path = f"datalake/metadata/tamanho_{ano}.json"
        try:
            head = self.sessao_web.head(url, timeout=30, allow_redirects=True)
            tamanho = int(head.headers.get('Content-Length', 0))
            if os.path.exists(meta_path) and not self.forcar_recarga:
                with open(meta_path, 'r') as f:
                    if json.load(f).get('tamanho') == tamanho: return False, tamanho
            return True, tamanho
        except: return True, 0

    def _construir_raw_operacional(self, df_raw, ano):
        df = df_raw.copy()
        df.columns = [self._higienizar_texto(c) for c in df.columns]
        for canonico, aliases in MAPA_SEMANTICO_COLUNAS.items():
            cols = [a for a in [self._higienizar_texto(al) for al in aliases] if a in df.columns]
            if cols:
                df[canonico] = df[cols[0]]
                for c in cols[1:]: df[canonico] = df[canonico].combine_first(df[c])
        for col in ESQUEMA_RAW_CANONICO.keys():
            if col not in df.columns and col != "ANO_BASE": df[col] = pd.Series(dtype='object')
        df["DESCR_TIPOLOCAL"] = df["DESCR_TIPOLOCAL"].astype(str).replace('nan', '')
        df["DESCR_SUBTIPOLOCAL"] = df["DESCR_SUBTIPOLOCAL"].astype(str).replace('nan', '')
        df["DESCR_SUBTIPOLOCAL"] = df["DESCR_SUBTIPOLOCAL"].replace('', np.nan).combine_first(df["DESCR_TIPOLOCAL"].replace('', np.nan)).fillna('')
        mask = df["DESCR_SUBTIPOLOCAL"].map(self._higienizar_texto).isin(SUBTIPOS_LOCAL_PERMITIDOS)
        df.loc[(df["DESCR_TIPOLOCAL"] == '') & mask, "DESCR_TIPOLOCAL"] = "VIA PUBLICA"
        df = df[list(ESQUEMA_RAW_CANONICO.keys() - {"ANO_BASE"})].copy()
        df.to_parquet(f"datalake/raw/ssp_{ano}.parquet", index=False)
        return df

    def _processar_trusted_refined(self, df_raw, ano):
        df = df_raw.copy()
        for col in ESQUEMA_TRUSTED.keys():
            if col not in df.columns and col != 'ANO_BASE': df[col] = np.nan
        vol = len(df)
        df['ANO_BASE'] = str(ano)
        for col, tipo in ESQUEMA_TRUSTED.items():
            if col not in df.columns: continue
            if tipo == 'string': df[col] = df[col].astype(str).replace('nan', '')
            elif tipo == 'float': df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors="coerce")
            elif tipo == 'datetime': df[col] = pd.to_datetime(df[col], errors='coerce')
            elif tipo == 'int': df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        df['DATA_OCORRENCIA_BO'] = df['DATA_OCORRENCIA_BO'].dt.normalize()
        m_t = df['DATA_OCORRENCIA_BO'].between(self.janela_inicio, self.data_execucao)
        m_g = (df['LATITUDE'].notna() & df['LONGITUDE'].notna() & (df['LATITUDE'] != 0) & 
               df['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1]) &
               df['LONGITUDE'].between(LIMITES_SP['lon'][0], LIMITES_SP['lon'][1]))
        df_t = df[m_t & m_g].copy()
        self.auditoria['falhas_integridade'] += (vol - len(df_t))
        self.auditoria['volume_trusted'] += len(df_t)
        m_n = (df_t['NATUREZA_APURADA'].isin(CATALOGO_CRIMES.keys()) & 
               df_t['DESCR_TIPOLOCAL'].isin(TIPOS_LOCAL_PERMITIDOS) & 
               df_t['DESCR_SUBTIPOLOCAL'].isin(SUBTIPOS_LOCAL_PERMITIDOS))
        df_r = df_t[m_n][COLUNAS_REFINED_EVENTOS].copy()
        self.auditoria['volume_refined'] += len(df_r)
        return df_t, df_r

    def _inferir_perfil(self, linha):
        pfs = set()
        tx = f"{linha.get('NATUREZA_APURADA','')} {linha.get('DESCR_CONDUTA','')} {linha.get('RUBRICA','')}".upper()
        for p, kws in PALAVRAS_CHAVE_PERFIL.items():
            if any(k in tx for k in kws): pfs.add(p)
        if not pfs: pfs.update(CATALOGO_CRIMES.get(linha.get('NATUREZA_APURADA'), {}).get('perfis', []))
        return list(pfs) if pfs else ['Indefinido']

    def _turno(self, h_s):
        try:
            h = int(str(h_s).split(':')[0])
            if 0 <= h < 6: return 'Madrugada'
            if 6 <= h < 12: return 'Manha'
            if 12 <= h < 18: return 'Tarde'
            return 'Noite'
        except: return 'Noite'

    def _features(self, df_r):
        df = df_r.copy()
        df['perfis'] = df.apply(self._inferir_perfil, axis=1)
        df = df.explode('perfis').dropna(subset=['perfis'])
        df['geohash'] = [gh.encode(la, lo, precision=7) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]
        df['turno'] = df['HORA_OCORRENCIA_BO'].apply(self._turno)
        df['peso'] = df['NATUREZA_APURADA'].apply(lambda x: CATALOGO_CRIMES.get(x, {}).get('peso', 1.0))
        pnl = df.groupby(['geohash', 'perfis', 'turno', 'DATA_OCORRENCIA_BO'], as_index=False).agg(
            y_d=('peso', 'sum'), lat=('LATITUDE', 'mean'), lon=('LONGITUDE', 'mean')
        ).sort_values(['geohash', 'perfis', 'turno', 'DATA_OCORRENCIA_BO'])
        ks = ['geohash', 'perfis', 'turno']
        pnl['l7'] = pnl.groupby(ks)['y_d'].shift(7).fillna(0)
        pnl['l14'] = pnl.groupby(ks)['y_d'].shift(14).fillna(0)
        pnl['m30'] = pnl.groupby(ks)['y_d'].transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean()).fillna(0)
        pnl['target'] = pnl.groupby(ks)['y_d'].transform(lambda x: x.shift(-7).rolling(7, min_periods=1).sum())
        macro = pnl.groupby('DATA_OCORRENCIA_BO')['y_d'].sum().reset_index().rename(columns={'DATA_OCORRENCIA_BO': 'ds', 'y_d': 'y'})
        m_p = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False).fit(macro)
        prj = m_p.predict(m_p.make_future_dataframe(periods=14))[['ds', 'yhat']]
        pnl = pnl.merge(prj, left_on='DATA_OCORRENCIA_BO', right_on='ds', how='left')
        pnl['ft'] = pnl['yhat'] / max(pnl['yhat'].mean(), 1.0)
        self.prj = prj
        return pnl

    def _treinar(self, pnl):
        pnl_v = pnl.dropna(subset=['target']).copy()
        c_v = pnl_v['DATA_OCORRENCIA_BO'].max() - pd.Timedelta(days=30)
        tr, ts = pnl_v[pnl_v['DATA_OCORRENCIA_BO'] <= c_v], pnl_v[pnl_v['DATA_OCORRENCIA_BO'] > c_v]
        fs = ['lat', 'lon', 'l7', 'l14', 'm30', 'ft']
        e_p, e_t = LabelEncoder().fit(pnl['perfis']), LabelEncoder().fit(pnl['turno'])
        for d in [tr, ts, pnl]:
            d.loc[:, 'pe'] = e_p.transform(d['perfis'])
            d.loc[:, 'te'] = e_t.transform(d['turno'])
        fs.extend(['pe', 'te'])
        md = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.05, max_depth=6)
        md.fit(tr[fs], tr['target'])
        if not ts.empty:
            pv = np.clip(md.predict(ts[fs]), 0, None)
            self.auditoria['mae_modelo'] = round(float(mean_absolute_error(ts['target'], pv)), 3)
            self.auditoria['rmse_modelo'] = round(float(math.sqrt(mean_squared_error(ts['target'], pv))), 3)
        md.fit(pnl_v[fs], pnl_v['target'])
        f_g = pnl.sort_values('DATA_OCORRENCIA_BO').groupby(['geohash', 'perfis', 'turno']).tail(1).copy()
        f_g['l7'], f_g['l14'] = f_g['y_d'], f_g.groupby(['geohash', 'perfis', 'turno'])['y_d'].shift(7).fillna(0)
        f_g['ft'] = self.prj.iloc[-1]['yhat'] / max(self.prj['yhat'].mean(), 1.0)
        f_g['sp'] = np.clip(md.predict(f_g[fs]), 0, None)
        esc = max(f_g['sp'].quantile(0.95), 1.0)
        f_g['sn'] = ((f_g['sp'] / esc) * 10).clip(0.5, 10.0).round(2)
        f_g['rp'] = (1.0 + (f_g['sn'] * 0.20)).round(2)
        self.auditoria['malha_motorista'] = len(f_g[f_g['perfis'] == 'Motorista'])
        self.auditoria['malha_motociclista'] = len(f_g[f_g['perfis'] == 'Motociclista'])
        self.auditoria['malha_pedestre'] = len(f_g[f_g['perfis'] == 'Pedestre'])
        self.auditoria['malha_ciclista'] = len(f_g[f_g['perfis'] == 'Ciclista'])
        return f_g[['geohash', 'perfis', 'turno', 'sn', 'rp']]

    def _sync(self, mlh):
        cl = self.banco_nuvem.collection('niveis_risco')
        ats = {d.id: d.to_dict().get('hr') for d in cl.stream()}
        bt, ops, vvs = self.banco_nuvem.batch(), 0, set()
        self.auditoria['documentos_sincronizados'] = len(mlh)
        for _, l in mlh.iterrows():
            did = f"{l['geohash']}_{l['perfis']}_{l['turno']}"
            vvs.add(did)
            py = {'sc': round(float(l['sn']), 2), 'rp': round(float(l['rp']), 2), 'gh': l['geohash'], 'g4': l['geohash'][:4], 'pf': l['perfis'], 'pd': l['turno']}
            hr = hashlib.sha256(json.dumps(py, sort_keys=True).encode('utf-8')).hexdigest()
            py['hr'], py['ua'] = hr, firestore.SERVER_TIMESTAMP
            if did not in ats or ats[did] != hr:
                bt.set(cl.document(did), py, merge=True)
                ops += 1
                self.auditoria['documentos_atualizados'] += 1
                if ops >= 450: bt.commit(); bt, ops = self.banco_nuvem.batch(), 0
        obs = set(ats.keys()) - vvs
        for d in obs:
            bt.delete(cl.document(d))
            ops += 1
            if ops >= 450: bt.commit(); bt, ops = self.banco_nuvem.batch(), 0
        if ops > 0: bt.commit()

    def _runbook(self):
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        with open(f"datalake/reports/runbook_{ts}.md", 'w') as f:
            f.write(f"# Runbook SafeDriver\n**TS:** {self.auditoria['timestamp']}\n- RAW: {self.auditoria['volume_raw']}\n- MAE: {self.auditoria['mae_modelo']}\n- Sync: {self.auditoria['documentos_atualizados']}")

    def executar_pipeline_completo(self):
        df_m = pd.DataFrame()
        try:
            for a in self.periodo_historico:
                u = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{a}.xlsx"
                c_r = f'datalake/raw/ssp_{a}.parquet'
                bk, sz = self._verificar_atualizacao(u, a)
                if bk or not os.path.exists(c_r):
                    rs = self.sessao_web.get(u, timeout=120)
                    if rs.status_code != 200: raise ConnectionError()
                    lp = pd.read_excel(io.BytesIO(rs.content), nrows=50, header=None)
                    lc = next((i for i, r in lp.iterrows() if any(t in [self._higienizar_texto(str(c)) for c in r.values] for t in ['NUM_BO', 'LATITUDE'])), None)
                    df_t = pd.read_excel(io.BytesIO(rs.content), skiprows=lc, dtype=str)
                    df_t = self._construir_raw_operacional(df_t, a)
                    with open(f"datalake/metadata/tamanho_{a}.json", 'w') as f: json.dump({'tamanho': sz}, f)
                    self.auditoria['novos_dados_baixados'] = True
                else: df_t = self._construir_raw_operacional(pd.read_parquet(c_r), a)
                self.auditoria['volume_raw'] += len(df_t)
                dt, dr = self._processar_trusted_refined(df_t, a)
                dt.to_parquet(f'datalake/trusted/ssp_trusted_{a}.parquet', index=False)
                if not dr.empty: df_m = pd.concat([df_m, dr])
            p_f = self._features(df_m)
            mlh = self._treinar(p_f)
            mlh.to_parquet("datalake/refined/malha_analitica.parquet", index=False)
            if self.banco_nuvem: self._sync(mlh)
            self._runbook(); self._notificar_sucesso()
        except Exception as e: self._notificar_erro(e); raise

if __name__ == "__main__": MotorSafeDriver().executar_pipeline_completo()
