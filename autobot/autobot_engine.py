import pandas as pd
import numpy as np
import os, io, requests, json, unicodedata, sys
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import h3
from prophet import Prophet
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_absolute_error, mean_squared_error

class AutobotSafeDriver:
    def __init__(self, persistencia=True):
        self.identidade = "Autobot SafeDriver"
        self.ano_atual = datetime.now().year
        self.periodo_historico = range(2022, self.ano_atual + 1)
        self.persistencia_ativa = persistencia
        self.banco_nuvem = self._conectar_nuvem() if persistencia else None
        self.sessao_rede = self._gerar_sessao_resiliente()
        self.auditoria = {
            "volume_bruto": 0, "volume_refinado": 0,
            "hexagonos_processados": 0, "mae": 0.0, "rmse": 0.0
        }
        self.biblioteca_crimes = {
            "LATROCINIO": 5.0, "EXTORSAO MEDIANTE SEQUESTRO": 5.0,
            "ROUBO DE VEICULO": 4.5, "ROUBO DE CARGA": 4.0,
            "ROUBO A TRANSEUNTE": 3.5, "FURTO DE VEICULO": 3.0, "OUTROS": 1.0
        }
        self.limites_sp = {"lat": (-25.5, -19.5), "lon": (-53.5, -44.0)}
        
        # Estrutura de pastas para o Data Lake
        for pasta in ['bruto', 'confiavel', 'refinado']: 
            os.makedirs(f'datalake/{pasta}', exist_ok=True)

    def _conectar_nuvem(self):
        config = os.environ.get('FIREBASE_JSON')
        if config and not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(config))
            firebase_admin.initialize_app(cred)
            return firestore.client()
        return None

    def _gerar_sessao_resiliente(self):
        s = requests.Session()
        r = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=r))
        s.headers.update({'User-Agent': 'Mozilla/5.0'})
        return s

    def _higienizar(self, texto):
        if pd.isna(texto) or not isinstance(texto, str): return str(texto) if not pd.isna(texto) else ""
        return "".join([c for c in unicodedata.normalize('NFKD', texto) if not unicodedata.combining(c)]).upper().strip()

    def _normalizar(self, coluna):
        limpo = self._higienizar(coluna)
        mapa = {
            "NUM_BO": ["NUM_BO", "NUMERO_BO", "N_BO"],
            "DATA_OCORRENCIA_BO": ["DATA_OCORRENCIA_BO", "DATA_FATO", "DATAOCORRENCIA"],
            "HORA_OCORRENCIA_BO": ["HORA_OCORRENCIA_BO", "HORA_FATO"],
            "LATITUDE": ["LATITUDE", "LAT", "LATITUDEDECIMAL"],
            "LONGITUDE": ["LONGITUDE", "LON", "LONGITUDEDECIMAL"],
            "NATUREZA_APURADA": ["NATUREZA_APURADA", "NATUREZA", "RUBRICA"]
        }
        for k, v in mapa.items():
            if limpo in v: return k
        return limpo

    def _corrigir_gps(self, v, lat=True):
        try:
            val = float(str(v).replace(',', '.'))
            lim = self.limites_sp['lat'] if lat else self.limites_sp['lon']
            if val < lim[0] or val > lim[1]: val = val / (10 ** (len(str(int(abs(val)))) - 2))
            return val
        except: return np.nan

    def _processar_ia(self, df):
        if len(df) < 20: return pd.DataFrame()
        
        # Identificação de manchas de crime (Hotspots)
        df['hotspot'] = DBSCAN(eps=0.001, min_samples=3).fit_predict(df[['LATITUDE', 'LONGITUDE']].values)
        df = df[df['hotspot'] != -1].copy()
        
        def extrair_hora(celula):
            try: return int(str(celula).split(':')[0].strip())
            except: return 0
            
        df['turno'] = df['HORA_OCORRENCIA_BO'].apply(lambda x: 'Madrugada' if 0<=extrair_hora(x)<6 else 'Manha' if 6<=extrair_hora(x)<12 else 'Tarde' if 12<=extrair_hora(x)<18 else 'Noite')
        df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'])
        
        # Sazonalidade Temporal (Prophet)
        hist = df.groupby('DATA_OCORRENCIA_BO').size().reset_index(name='y').rename(columns={'DATA_OCORRENCIA_BO': 'ds'})
        f_saz = 1.0
        if len(hist) >= 2:
            m_p = Prophet(yearly_seasonality=True, daily_seasonality=False).fit(hist)
            prev = m_p.predict(pd.DataFrame({'ds': [datetime.now() + timedelta(days=1)]}))
            f_saz = max(0.5, prev['yhat'].values[0] / hist['y'].mean())

        # Motor de Inferência (XGBoost)
        enc_t = LabelEncoder()
        df['t_cod'] = enc_t.fit_transform(df['turno'])
        df['peso'] = df['NATUREZA_APURADA'].apply(lambda x: self.biblioteca_crimes.get(x, 1.0))
        
        X = df[['LATITUDE', 'LONGITUDE', 't_cod', 'hotspot']]
        y = df['peso']
        m_x = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05).fit(X, y)
        
        # Métricas de Validação
        preds = m_x.predict(X)
        self.auditoria['mae'] = round(float(mean_absolute_error(y, preds)), 4)
        self.auditoria['rmse'] = round(float(np.sqrt(mean_squared_error(y, preds))), 4)
        
        # Geração da Malha H3 Nível 10
        df['h3'] = df.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 10), axis=1)
        res = df.groupby(['h3', 'turno']).agg({'peso': ['mean', 'count']}).reset_index()
        res.columns = ['h3', 'turno', 'peso_medio', 'frequencia']
        
        # Pontuação Dinâmica (Logarítmica para evitar notas iguais)
        res['pt'] = (res['peso_medio'] * np.log1p(res['frequencia']) * f_saz * 2).clip(0.5, 10.0).round(1)
        res['pn'] = (1 + (res['pt'] * 0.2)).round(1)
        
        return res

    def _sincronizar(self, malha):
        if not self.banco_nuvem or malha.empty: return
        colecao = self.banco_nuvem.collection('malha_seguranca')
        agrupado = malha.groupby('h3')
        batch = self.banco_nuvem.batch()
        contador = 0

        for h3_id, dados in agrupado:
            doc_ref = colecao.document(h3_id)
            scores = {row['turno']: {"pt": row['pt'], "pn": row['pn']} for _, row in dados.iterrows()}
            batch.set(doc_ref, {"id_h3": h3_id, "scores": scores, "ultima_atualizacao": firestore.SERVER_TIMESTAMP}, merge=True)
            contador += 1
            if contador >= 400:
                batch.commit()
                batch = self.banco_nuvem.batch()
                contador = 0
        batch.commit()
        self.auditoria['hexagonos_processados'] = len(agrupado)

    def executar(self):
        mestre = pd.DataFrame()
        for ano in self.periodo_historico:
            caminho_bruto = f'datalake/bruto/ssp_{ano}.parquet'
            
            if os.path.exists(caminho_bruto):
                df = pd.read_parquet(caminho_bruto)
            else:
                try:
                    url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                    r = self.sessao_rede.get(url, timeout=300)
                    df = pd.read_excel(io.BytesIO(r.content), dtype=str)
                    df.columns = [self._normalizar(c) for c in df.columns]
                    df = df.loc[:, ~df.columns.duplicated()].copy()
                    df['LATITUDE'] = df['LATITUDE'].apply(lambda x: self._corrigir_gps(x, True))
                    df['LONGITUDE'] = df['LONGITUDE'].apply(lambda x: self._corrigir_gps(x, False))
                    df.to_parquet(caminho_bruto, index=False)
                except: continue
            
            mestre = pd.concat([mestre, df[df['LATITUDE'].notna()]])
            self.auditoria['volume_bruto'] += len(df)
            
        if not mestre.empty:
            self.auditoria['volume_refinado'] = len(mestre)
            previsoes = self._processar_ia(mestre)
            previsoes.to_parquet('datalake/refinado/malha_final.parquet', index=False)
            self._sincronizar(previsoes)
            self._notificar_discord()

    def _notificar_discord(self):
        webhook = os.environ.get('DISCORD_SUCESSO')
        if not webhook: return
        payload = {
            "embeds": [{
                "title": f"🚀 {self.identidade}",
                "color": 3066993,
                "fields": [
                    {"name": "📉 Métricas IA", "value": f"**MAE:** {self.auditoria['mae']}\n**RMSE:** {self.auditoria['rmse']}", "inline": False},
                    {"name": "🌊 Camada Bruta", "value": f"Registros: {self.auditoria['volume_bruto']}", "inline": True},
                    {"name": "☁️ Firestore", "value": f"Hexágonos H3: {self.auditoria['hexagonos_processados']}", "inline": True}
                ]
            }]
        }
        requests.post(webhook, json=payload)

if __name__ == "__main__":
    AutobotSafeDriver().executar()
