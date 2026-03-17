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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
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
            "camadas": {"bruta": 0, "confiavel": 0, "refinada": 0},
            "perfis": {"Pedestre": 0, "Motorista": 0, "Ciclista": 0, "Motociclista": 0, "Geral": 0},
            "metricas": {"mae": 0.0, "rmse": 0.0},
            "nuvem": {"documentos": 0}
        }
        
        self.pesos_crimes = {
            "LATROCINIO": 10.0, "EXTORSAO MEDIANTE SEQUESTRO": 10.0,
            "ROUBO DE VEICULO": 8.5, "ROUBO DE CARGA": 8.0,
            "ROUBO A TRANSEUNTE": 7.5, "FURTO DE VEICULO": 4.0, "OUTROS": 1.0
        }
        
        self.mapa_palavras_perfil = {
            "Pedestre": ["PEDESTRE", "TRANSEUNTE", "CELULAR", "CALCADA", "PONTO DE ONIBUS"],
            "Motorista": ["VEICULO", "CARRO", "CAMINHAO", "AUTOMOVEL", "CARGA"],
            "Ciclista": ["BICI", "CICLO", "BICICLETA", "PEDALAR"],
            "Motociclista": ["MOTO", "MOTOCICLETA", "CAPACETE", "MOTOBOY"]
        }
        
        self.limites_sp = {"lat": (-24.5, -23.0), "lon": (-47.0, -45.5)}
        for p in ['bruto', 'confiavel', 'refinado']: os.makedirs(f'datalake/{p}', exist_ok=True)

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

    def _higienizar(self, t):
        if pd.isna(t) or not isinstance(t, str): return str(t) if not pd.isna(t) else ""
        return "".join([c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c)]).upper().strip()

    def _normalizar_coluna(self, c):
        limpo = self._higienizar(c)
        mapa = {
            "NUM_BO": ["NUM_BO", "NUMERO_BO", "N_BO"],
            "DATA_OCORRENCIA_BO": ["DATA_OCORRENCIA_BO", "DATA_FATO", "DATAOCORRENCIA"],
            "HORA_OCORRENCIA_BO": ["HORA_OCORRENCIA_BO", "HORA_FATO"],
            "LATITUDE": ["LATITUDE", "LAT", "LATITUDEDECIMAL"],
            "LONGITUDE": ["LONGITUDE", "LON", "LONGITUDEDECIMAL"],
            "NATUREZA_APURADA": ["NATUREZA_APURADA", "NATUREZA", "RUBRICA"],
            "LOCAL": ["DESCR_TIPOLOCAL", "TIPOLOCAL", "LOCAL"]
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

    def _atribuir_perfis(self, linha):
        texto = f"{linha['NATUREZA_APURADA']} {linha['LOCAL']}".upper()
        encontrados = [p for p, palavras in self.mapa_palavras_perfil.items() if any(w in texto for w in palavras)]
        return encontrados if encontrados else ["Geral"]

    def _processar_ia(self, df):
        if len(df) < 20: return pd.DataFrame()
        
        # Expansão por Perfis e Limpeza Espacial
        df['perfis'] = df.apply(self._atribuir_perfis, axis=1)
        df = df.explode('perfis')
        df['hotspot'] = DBSCAN(eps=0.001, min_samples=5).fit_predict(df[['LATITUDE', 'LONGITUDE']].values)
        df = df[df['hotspot'] != -1].copy()
        
        def h_int(x):
            try: return int(str(x).split(':')[0].strip())
            except: return 0
        df['turno'] = df['HORA_OCORRENCIA_BO'].apply(lambda x: 'Madrugada' if 0<=h_int(x)<6 else 'Manha' if 6<=h_int(x)<12 else 'Tarde' if 12<=h_int(x)<18 else 'Noite')
        
        # IA Preditiva (XGBoost)
        enc_t, enc_p = LabelEncoder(), LabelEncoder()
        df['t_cod'] = enc_t.fit_transform(df['turno'])
        df['p_cod'] = enc_p.fit_transform(df['perfis'])
        df['peso'] = df['NATUREZA_APURADA'].apply(lambda x: self.pesos_crimes.get(x, 1.0))
        
        X = df[['LATITUDE', 'LONGITUDE', 't_cod', 'p_cod', 'hotspot']]
        y = df['peso']
        modelo = xgb.XGBRegressor(n_estimators=150, learning_rate=0.05).fit(X, y)
        
        self.auditoria['metricas']['mae'] = round(float(mean_absolute_error(y, modelo.predict(X))), 4)
        self.auditoria['metricas']['rmse'] = round(float(np.sqrt(mean_squared_error(y, modelo.predict(X)))), 4)
        
        # Agregação H3 Nível 10
        df['h3'] = df.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 10), axis=1)
        res = df.groupby(['h3', 'perfis', 'turno']).agg({'peso': ['mean', 'count']}).reset_index()
        res.columns = ['h3', 'perfil', 'turno', 'peso_medio', 'freq']
        
        # Score com Variância (Frequência Logarítmica)
        res['pt_bruta'] = res['peso_medio'] * np.log2(res['freq'] + 1)
        scaler = MinMaxScaler(feature_range=(0.5, 10.0))
        res['pt'] = scaler.fit_transform(res[['pt_bruta']]).round(1)
        res['pn'] = (1 + (res['pt'] * 0.15)).round(2)
        
        return res

    def _sincronizar(self, malha):
        if not self.banco_nuvem or malha.empty: return
        col = self.banco_nuvem.collection('malha_seguranca')
        batch = self.banco_nuvem.batch()
        count = 0

        for _, r in malha.iterrows():
            # Documentos Atômicos: perfil_hexagono
            doc_id = f"{r['perfil'].lower()}_{r['h3']}"
            batch.set(col.document(doc_id), {
                "id_h3": r['h3'],
                "perfil": r['perfil'],
                "scores": {r['turno']: {"pt": r['pt'], "pn": r['pn']}},
                "data": firestore.SERVER_TIMESTAMP
            }, merge=True)
            
            count += 1
            if count >= 400:
                batch.commit()
                batch = self.banco_nuvem.batch()
                count = 0
        batch.commit()
        self.auditoria['nuvem']['documentos'] = count + (self.auditoria['nuvem']['documentos'])

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
                    df.columns = [self._normalizar_coluna(c) for c in df.columns]
                    df = df.loc[:, ~df.columns.duplicated()].copy()
                    df['LATITUDE'] = df['LATITUDE'].apply(lambda x: self._corrigir_gps(x, True))
                    df['LONGITUDE'] = df['LONGITUDE'].apply(lambda x: self._corrigir_gps(x, False))
                    df.to_parquet(caminho_bruto, index=False)
                except: continue
            
            self.auditoria['camadas']['bruta'] += len(df)
            val = df[df['LATITUDE'].notna()].copy()
            self.auditoria['camadas']['confiavel'] += len(val)
            mestre = pd.concat([mestre, val])

        if not mestre.empty:
            prev = self._processar_ia(mestre)
            self.auditoria['camadas']['refinada'] = len(prev)
            
            # Contagem detalhada por perfil para o relatório
            for p in self.auditoria['perfis'].keys():
                self.auditoria['perfis'][p] = int(len(prev[prev['perfil'] == p]))
                
            prev.to_parquet('datalake/refinado/malha_final.parquet', index=False)
            self._sincronizar(prev)
            self._notificar()

    def _notificar(self):
        webhook = os.environ.get('DISCORD_SUCESSO')
        if not webhook: return
        perf_str = "\n".join([f"**{k}:** {v} hexágonos de risco" for k, v in self.auditoria['perfis'].items()])
        payload = {
            "embeds": [{
                "title": f"🚀 {self.identidade} - Relatório de Missão",
                "color": 3066993,
                "fields": [
                    {"name": "🌊 Camadas do Lakehouse", "value": f"Bruto: {self.auditoria['camadas']['bruta']}\nConfiavel: {self.auditoria['camadas']['confiavel']}\nRefinado: {self.auditoria['camadas']['refinada']}", "inline": True},
                    {"name": "📉 IA (MAE/RMSE)", "value": f"{self.auditoria['metricas']['mae']} / {self.auditoria['metricas']['rmse']}", "inline": True},
                    {"name": "👥 Volumetria por Perfil", "value": perf_str, "inline": False},
                    {"name": "☁️ Sincronização", "value": f"{self.auditoria['camadas']['refinada']} Documentos", "inline": True}
                ]
            }]
        }
        requests.post(webhook, json=payload)

if __name__ == "__main__":
    AutobotSafeDriver().executar()
