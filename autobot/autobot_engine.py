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

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.FileHandler("datalake/logs_sistema.log"), logging.StreamHandler()])

class MotorSeguranca:
    def __init__(self, persistencia=True):
        self.versao = "4.0"
        self.persistencia = persistencia
        self.banco = self._conectar_nuvem() if persistencia else None
        self.sessao = self._gerar_sessao()
        self.auditoria = {"raw": 0, "trusted": 0, "refined": 0, "metricas": {}, "nuvem": {"total": 0, "delta": 0}}
        for camada in ['bronze_raw', 'silver_trusted', 'gold_refined']: os.makedirs(f'datalake/{camada}', exist_ok=True)

    def _conectar_nuvem(self):
        config = os.environ.get('FIREBASE_JSON')
        if config and not firebase_admin._apps:
            try:
                cred = credentials.Certificate(json.loads(config))
                firebase_admin.initialize_app(cred)
                return firestore.client()
            except: return None
        return None

    def _gerar_sessao(self):
        s = requests.Session()
        r = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=r))
        return s

    def _limpar_texto(self, t):
        if pd.isna(t) or not isinstance(t, str): return ""
        return "".join([c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c)]).upper().strip()

    def _definir_peso(self, linha):
        texto = " ".join([str(v) for v in linha.values if pd.api.types.is_scalar(v) and pd.notnull(v)])
        t = self._limpar_texto(texto)
        if any(w in t for w in ["LATROCINIO", "SEQUESTRO"]): return 10.0
        if "ROUBO" in t and any(w in t for w in ["VEICULO", "CARRO", "MOTO", "AUTO"]): return 8.5
        if "ROUBO" in t and "CARGA" in t: return 8.0
        if "ROUBO" in t and any(w in t for w in ["CELULAR", "PEDESTRE", "TRANSEUNTE"]): return 7.5
        if "FURTO" in t and any(w in t for w in ["VEICULO", "CARRO", "MOTO"]): return 4.0
        return 3.0 if "FURTO" in t else 1.0

    def _gerar_camada_ouro(self, df):
        df['id_h3'] = df.apply(lambda r: h3.latlng_to_cell(float(r['LATITUDE']), float(r['LONGITUDE']), 10), axis=1)
        df['peso_calculado'] = df.apply(self._definir_peso, axis=1)
        
        filtros = {"Pedestre": ["CELULAR", "ONIBUS", "PEDESTRE"], "Motorista": ["VEICULO", "CARRO", "CARGA"], "Ciclista": ["BICI"], "Motociclista": ["MOTO"]}
        def classificar(r):
            t = self._limpar_texto(" ".join([str(v) for v in r.values if pd.api.types.is_scalar(v)]))
            m = [p for p, words in filtros.items() if any(w in t for w in words)]
            return m if m else ["Geral"]
        
        df['perfis'] = df.apply(classificar, axis=1)
        df = df.explode('perfis')
        
        def set_turno(h):
            try:
                h = int(str(h).split(':')[0])
                if 0<=h<6: return 'Madrugada'
                if 6<=h<12: return 'Manha'
                if 12<=h<18: return 'Tarde'
                return 'Noite'
            except: return 'Noite'
        df['turno'] = df['HORA_OCORRENCIA_BO'].apply(set_turno)
        
        le = LabelEncoder()
        df['turno_cod'] = le.fit_transform(df['turno'])
        X, y = df[['LATITUDE', 'LONGITUDE', 'turno_cod']], df['peso_calculado']
        
        if len(X) >= 10:
            xt, xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = xgb.XGBRegressor(n_estimators=100).fit(xt, yt)
            yp = modelo.predict(xv)
            mae, r2 = mean_absolute_error(yv, yp), r2_score(yv, yp)
        else: mae, r2 = 0.0, 1.0
        
        self.auditoria['metricas'] = {"mae": round(mae, 4), "r2": round(r2, 4), "acc": round(max(0.0, 100.0 - ((mae/10)*100)), 2)}
        
        res = df.groupby(['id_h3', 'perfis', 'turno']).agg({'peso_calculado': ['mean', 'count'], 'LATITUDE': 'mean', 'LONGITUDE': 'mean'}).reset_index()
        res.columns = ['h3', 'perfil', 'turno', 'peso_m', 'freq', 'lat', 'lon']
        res['score'] = MinMaxScaler(feature_range=(0.5, 10.0)).fit_transform(res[['peso_m']]).round(1) if len(res)>1 else 5.0

        res.to_parquet('datalake/gold_refined/mapa_risco_auditavel.parquet', index=False)

        dim_local = res[['h3', 'lat', 'lon']].drop_duplicates()
        dim_perfil = pd.DataFrame({'id_perfil': range(len(res['perfil'].unique())), 'perfil': res['perfil'].unique()})
        dim_tempo = pd.DataFrame({'id_tempo': range(len(res['turno'].unique())), 'turno': res['turno'].unique()})
        
        fato = res.merge(dim_perfil, left_on='perfil', right_on='perfil').merge(dim_tempo, left_on='turno', right_on='turno')
        fato = fato[['h3', 'id_perfil', 'id_tempo', 'score', 'freq']]
        fato['data_carga'] = datetime.now()
        
        dim_local.to_csv('datalake/gold_refined/dim_localizacao.csv', index=False)
        dim_perfil.to_csv('datalake/gold_refined/dim_perfil.csv', index=False)
        dim_tempo.to_csv('datalake/gold_refined/dim_tempo.csv', index=False)
        fato.to_csv('datalake/gold_refined/fato_risco.csv', index=False)
        
        return res

    def _sincronizar(self, df):
        if not self.db or df.empty: return
        ref = 'datalake/silver_trusted/referencia_delta.parquet'
        delta = df.copy()
        if os.path.exists(ref):
            try:
                hist = pd.read_parquet(ref)
                df['sig'] = df['h3'] + df['perfil'] + df['turno'] + df['score'].astype(str)
                hist['sig'] = hist['h3'] + hist['perfil'] + hist['turno'] + hist['score'].astype(str)
                delta = df[~df['sig'].isin(hist['sig'])].copy()
                df.drop(columns=['sig'], inplace=True)
            except: pass
            
        lote = self.db.batch()
        c, s = 0, 0
        for _, r in delta.iterrows():
            lote.set(self.db.collection('risco_geografico').document(f"{r['perfil'].lower()}_{r['h3']}"), {
                "h3": r['h3'], "perfil": r['perfil'],
                f"turnos.{r['turno']}": {"score": r['score'], "ts": firestore.SERVER_TIMESTAMP}
            }, merge=True)
            c += 1; s += 1
            if c >= 450: lote.commit(); lote = self.db.batch(); c = 0
        if c > 0: lote.commit()
        self.auditoria['nuvem'] = {"total": len(df), "delta": s}

    def iniciar(self):
        try:
            mestre = pd.DataFrame()
            meta_p, meta = 'datalake/bronze_raw/metadados.json', {}
            if os.path.exists(meta_p):
                with open(meta_p, 'r') as f: meta = json.load(f)
            
            for ano in range(2023, datetime.now().year + 1):
                path_bruto = f'datalake/bronze_raw/ssp_{ano}.parquet'
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                try:
                    head = self.session.head(url, timeout=20)
                    tamanho = int(head.headers.get('Content-Length', 0))
                    if os.path.exists(path_bruto) and meta.get(str(ano)) == tamanho:
                        dados = pd.read_parquet(path_bruto)
                    else:
                        r = self.session.get(url, timeout=300)
                        dados = pd.read_excel(io.BytesIO(r.content), dtype=str)
                        dados.columns = [self._limpar_texto(c) for c in dados.columns]
                        dados.to_parquet(path_bruto, index=False); meta[str(ano)] = tamanho
                except: dados = pd.read_parquet(path_bruto) if os.path.exists(path_bruto) else pd.DataFrame()
                if not dados.empty:
                    self.auditoria['raw'] += len(dados)
                    mestre = pd.concat([mestre, dados])

            if not mestre.empty:
                trusted = mestre[mestre['LATITUDE'].notna()].copy()
                self.auditoria['trusted'] = len(trusted)
                refined = self._gerar_camada_ouro(trusted)
                if not refined.empty:
                    self.auditoria['refined'] = len(refined)
                    self._sincronizar(refined)
                    refined.to_parquet('datalake/silver_trusted/referencia_delta.parquet', index=False)
                    with open(meta_p, 'w') as f: json.dump(meta, f)
                    self._notificar(True)
        except Exception as e:
            self._notificar(False, str(e))

    def _notificar(self, ok, err=None):
        url = os.environ.get('DISCORD_SUCESSO')
        if not url: return
        if ok:
            t, d = self.auditoria['nuvem']['total'], self.auditoria['nuvem']['delta']
            eco = ((t - d) / t * 100) if t > 0 else 0
            corpo = {"embeds": [{"title": "✅ PIPELINE ATUALIZADO", "color": 3066993, "fields": [
                {"name": "📉 MÉTRICAS IA", "value": f"Acurácia: {self.auditoria['metricas']['acc']}% | $R^2$: {self.auditoria['metricas']['r2']}", "inline": False},
                {"name": "📦 MEDALHÃO", "value": f"Raw: {self.auditoria['raw']} | Trusted: {self.auditoria['trusted']} | Refined: {self.auditoria['refined']}", "inline": True},
                {"name": "💰 ECONOMIA CLOUD", "value": f"Delta: {d} | Poupança: **{eco:.1f}%**", "inline": True}]}]}
        else:
            corpo = {"embeds": [{"title": "❌ ERRO NO PIPELINE", "color": 15158332, "description": f"Log: `{err}`"}]}
        requests.post(url, json=corpo)

if __name__ == "__main__":
    MotorSeguranca().iniciar()
