import io
import os
import json
import math
import shutil
import hashlib
import logging
import unicodedata
import warnings
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
            "mae": 0.0, "rmse": 0.0, "docs_atualizados": 0, "docs_removidos": 0
        }
        
        if not os.path.exists(self.lock_file):
            self.auditoria["modo"] = "FULL_RELOAD"
            if os.path.exists('datalake'): shutil.rmtree('datalake')
        
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

    def _notificar_discord(self, sucesso=True, erro=None):
        webhook = os.environ.get('DISCORD_SUCESSO') if sucesso else os.environ.get('DISCORD_ERRO')
        if not webhook: return
        
        if sucesso:
            embed = {
                "title": f"🚀 SafeDriver Engine: Processamento {self.auditoria['modo']} Concluído",
                "color": 3066993,
                "fields": [
                    {"name": "📊 Volumetria de Qualidade", "value": f"RAW Ingerido: {self.auditoria['volume_raw']:,}\nTRUSTED Válido: {self.auditoria['volume_trusted']:,}\nREFINED (Crimes Foco): {self.auditoria['volume_refined']:,}", "inline": False},
                    {"name": "🧠 Performance Preditiva", "value": f"Erro Médio Absoluto (MAE): {self.auditoria['mae']}\nRaiz do Erro Quadrático (RMSE): {self.auditoria['rmse']}", "inline": False},
                    {"name": "☁️ Sincronização Firebase", "value": f"Mutações: {self.auditoria['docs_atualizados']:,}\nObsoletos Removidos: {self.auditoria['docs_removidos']:,}", "inline": False}
                ],
                "footer": {"text": f"Duração: {(datetime.now() - self.ts_execucao).total_seconds():.1f}s"}
            }
        else:
            embed = {
                "title": "🚨 Falha Crítica no Motor SafeDriver",
                "color": 15158332,
                "description": f"Ocorreu um erro irrecuperável durante a execução da esteira.\n\n**Traceback:**\n```{str(erro)}```"
            }
            
        requests.post(webhook, json={"embeds": [embed]})

    def _gerar_runbook(self):
        ts = self.ts_execucao.strftime('%Y%m%d_%H%M%S')
        conteudo = f"""# SafeDriver Engine MLOps Runbook
**Timestamp:** {self.ts_execucao}
**Modo de Execução:** {self.auditoria['modo']}

## 1. Integridade de Dados (ETL)
- **Registros Lidos da SSP (RAW):** {self.auditoria['volume_raw']:,}
- **Registros Úteis com Coordenadas (TRUSTED):** {self.auditoria['volume_trusted']:,}
- **Eventos Analíticos Filtrados (REFINED):** {self.auditoria['volume_refined']:,}
- **Taxa de Aproveitamento Geral:** {round((self.auditoria['volume_refined'] / max(self.auditoria['volume_raw'], 1)) * 100, 2)}%

## 2. Observabilidade de IA Preditiva
- **Modelo Base:** Prophet (Sazonalidade) + XGBoost Regressor (Espacial)
- **MAE:** {self.auditoria['mae']}
- **RMSE:** {self.auditoria['rmse']}

## 3. Topologia da Nuvem (Delta Sync)
- **Documentos Atualizados/Criados:** {self.auditoria['docs_atualizados']:,}
- **Documentos Expirados/Removidos:** {self.auditoria['docs_removidos']:,}
"""
        with open(f"datalake/reports/runbook_{ts}.md", 'w', encoding='utf-8') as f:
            f.write(conteudo)

    def _ingerir_fonte(self, content):
        excel = pd.ExcelFile(io.BytesIO(content))
        dfs = []
        for aba in excel.sheet_names:
            if any(x in aba.upper() for x in ["CAMPOS", "METADADOS", "DICIONARIO", "LEGENDA"]): continue
            df = excel.parse(aba, dtype=str)
            df.columns = [self._higienizar(c) for c in df.columns]
            for can, aliases in MAPA_SEMANTICO_COLUNAS.items():
                match = [a for a in aliases if a in df.columns]
                if match: df[can] = df[match[0]]
            for col in ESQUEMA_RAW_CANONICO.keys():
                if col not in df.columns: df[col] = np.nan
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    def _qualificar_dados(self, df_raw, ano):
        df = df_raw.copy()
        df['ANO_BASE'] = int(ano)
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
        df['perfis'] = df.apply(lambda x: [p for p, kws in PALAVRAS_CHAVE_PERFIL.items() if any(k in str(x).upper() for k in kws)] or ['Indefinido'], axis=1)
        df = df.explode('perfis')
        df['gh'] = [gh.encode(la, lo, precision=7) for la, lo in zip(df['LATITUDE'], df['LONGITUDE'])]
        df['hr'] = df['HORA_OCORRENCIA_BO'].astype(str).str.extract(r'(\d+)').fillna(0).astype(int)
        df['turno'] = df['hr'].apply(lambda h: 'Madrugada' if 0<=h<6 else 'Manha' if 6<=h<12 else 'Tarde' if 12<=h<18 else 'Noite')
        df['w'] = df['NATUREZA_APURADA'].apply(lambda x: CATALOGO_CRIMES.get(x, {}).get('peso', 1.0))
        
        pnl = df.groupby(['gh', 'perfis', 'turno', 'DATA_OCORRENCIA_BO'], as_index=False).agg(y=('w', 'sum'), la=('LATITUDE', 'mean'), lo=('LONGITUDE', 'mean'))
        pnl = pnl.sort_values(['gh', 'perfis', 'turno', 'DATA_OCORRENCIA_BO'])
        pnl['l7'] = pnl.groupby(['gh', 'perfis', 'turno'])['y'].shift(7).fillna(0)
        pnl['target'] = pnl.groupby(['gh', 'perfis', 'turno'])['y'].transform(lambda x: x.shift(-7).rolling(7, min_periods=1).sum())
        
        macro = pnl.groupby('DATA_OCORRENCIA_BO')['y'].sum().reset_index().rename(columns={'DATA_OCORRENCIA_BO': 'ds', 'y': 'y'})
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
        
        le_p, le_t = LabelEncoder().fit(pnl['perfis']), LabelEncoder().fit(pnl['turno'])
        for d in [pnl_v, pnl]:
            d['pe'], d['te'] = le_p.transform(d['perfis']), le_t.transform(d['turno'])
            
        fs = ['la', 'lo', 'l7', 'ft', 'pe', 'te']
        md = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05).fit(pnl_v[fs], pnl_v['target'])
        
        previsoes = np.clip(md.predict(pnl_v[fs]), 0, None)
        self.auditoria['mae'] = round(float(mean_absolute_error(pnl_v['target'], previsoes)), 3)
        self.auditoria['rmse'] = round(float(math.sqrt(mean_squared_error(pnl_v['target'], previsoes))), 3)

        grid = pnl.sort_values('DATA_OCORRENCIA_BO').groupby(['gh', 'perfis', 'turno']).tail(1).copy()
        grid['ft'] = prj.iloc[-1]['yhat'] / max(prj['yhat'].mean(), 1.0)
        grid['sp'] = np.clip(md.predict(grid[fs]), 0, None)
        esc = max(grid['sp'].quantile(0.95), 1.0)
        grid['sn'] = ((grid['sp'] / esc) * 10).clip(0.5, 10.0).round(2)
        grid['rp'] = (1.0 + (grid['sn'] * 0.2)).round(2)
        
        if self.db:
            cl = self.db.collection('niveis_risco')
            documentos_atuais = {doc.id: doc.to_dict().get('hr') for doc in cl.stream()}
            bt, ops, ids_vivos = self.db.batch(), 0, set()
            
            for _, l in grid.iterrows():
                did = f"{l['gh']}_{l['perfis']}_{l['te']}"
                ids_vivos.add(did)
                
                payload = {'sc': float(l['sn']), 'rp': float(l['rp']), 'gh': l['gh'], 'g4': l['gh'][:4], 'pf': l['perfis'], 'pd': l['turno']}
                hr = hashlib.sha256(json.dumps(payload, sort_keys=True).encode('utf-8')).hexdigest()
                payload['hr'] = hr
                payload['ua'] = firestore.SERVER_TIMESTAMP
                
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
