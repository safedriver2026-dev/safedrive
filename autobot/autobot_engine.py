import pandas as pd
import numpy as np
import os, io, requests, json, unicodedata, gc, re
from datetime import datetime, timedelta
import firebase_admin
from firebase_admin import credentials, firestore
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import h3
from prophet import Prophet
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

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
            "metricas": {"mae": 0.0, "rmse": 0.0, "acerto": 0.0},
            "nuvem": {"documentos": 0}
        }
        
        self.pesos_crimes = {
            "LATROCINIO": 10.0, "EXTORSAO MEDIANTE SEQUESTRO": 10.0,
            "ROUBO DE VEICULO": 8.5, "ROUBO DE CARGA": 8.0,
            "ROUBO A TRANSEUNTE": 7.5, "FURTO DE VEICULO": 4.0, "OUTROS": 1.0
        }
        
        self.mapa_palavras_perfil = {
            "Pedestre": ["PEDESTRE", "TRANSEUNTE", "CELULAR", "CALCADA", "ONIBUS"],
            "Motorista": ["VEICULO", "CARRO", "CAMINHAO", "AUTOMOVEL", "CARGA", "MOTORISTA"],
            "Ciclista": ["BICICLETA", "CICLISTA", "PEDALAR", "BICI"],
            "Motociclista": ["MOTO", "MOTOCICLETA", "CAPACETE", "MOTOBOY", "MOTONETA"]
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
        texto_global = " ".join([str(val) for val in linha.values if pd.notnull(val)])
        texto_limpo = self._higienizar(texto_global)
        
        encontrados = []
        for p, palavras in self.mapa_palavras_perfil.items():
            if any(re.search(rf'\b{w}\b', texto_limpo) for w in palavras):
                encontrados.append(p)
        return encontrados if encontrados else ["Geral"]

    def _processar_ia(self, df):
        min_linhas = 20 if self.persistencia_ativa else 2
        min_densidade = 5 if self.persistencia_ativa else 1
        
        if len(df) < min_linhas: return pd.DataFrame()
        
        df['h3'] = df.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 10), axis=1)
        h3_counts = df['h3'].value_counts()
        hotspots_validos = h3_counts[h3_counts >= min_densidade].index
        df = df[df['h3'].isin(hotspots_validos)].copy()
        
        del h3_counts; gc.collect()
        
        if len(df) < min_linhas: return pd.DataFrame()
        
        df['perfis'] = df.apply(self._atribuir_perfis, axis=1)
        df = df.explode('perfis')
        
        def h_int(x):
            try: return int(str(x).split(':')[0].strip())
            except: return 0
        df['turno'] = df['HORA_OCORRENCIA_BO'].apply(lambda x: 'Madrugada' if 0<=h_int(x)<6 else 'Manha' if 6<=h_int(x)<12 else 'Tarde' if 12<=h_int(x)<18 else 'Noite')
        
        df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        hist = df.groupby('DATA_OCORRENCIA_BO').size().reset_index(name='y').rename(columns={'DATA_OCORRENCIA_BO': 'ds'})
        f_saz = 1.0
        if len(hist) >= 2:
            m_p = Prophet(yearly_seasonality=True, daily_seasonality=False).fit(hist)
            prev = m_p.predict(pd.DataFrame({'ds': [datetime.now() + timedelta(days=1)]}))
            f_saz = max(0.5, prev['yhat'].values[0] / hist['y'].mean())

        enc_t = LabelEncoder()
        df['t_cod'] = enc_t.fit_transform(df['turno'])
        
        # Correção do Bug de IA (Peso Constante): Higienizar a Natureza do Crime antes de buscar no dicionário
        df['peso'] = df['NATUREZA_APURADA'].apply(lambda x: self.pesos_crimes.get(self._higienizar(str(x)), 1.0))
        
        X = df[['LATITUDE', 'LONGITUDE', 't_cod']]
        y = df['peso']
        
        if len(X) > 50:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6).fit(X_train, y_train)
            mae = float(mean_absolute_error(y_test, modelo.predict(X_test)))
            rmse = float(np.sqrt(mean_squared_error(y_test, modelo.predict(X_test))))
        else:
            modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6).fit(X, y)
            mae = float(mean_absolute_error(y, modelo.predict(X)))
            rmse = float(np.sqrt(mean_squared_error(y, modelo.predict(X))))
            
        taxa_acerto = max(0.0, 100.0 - ((mae / 10.0) * 100.0))
        
        self.auditoria['metricas']['mae'] = round(mae, 4)
        self.auditoria['metricas']['rmse'] = round(rmse, 4)
        self.auditoria['metricas']['acerto'] = round(taxa_acerto, 2)
        
        res = df.groupby(['h3', 'perfis', 'turno']).agg({'peso': ['mean', 'count']}).reset_index()
        res.columns = ['h3', 'perfil', 'turno', 'peso_medio', 'freq']
        
        res['pt_bruta'] = res['peso_medio'] * np.log2(res['freq'] + 1) * f_saz
        scaler = MinMaxScaler(feature_range=(0.5, 10.0))
        
        if len(res) > 1:
            res['pt'] = scaler.fit_transform(res[['pt_bruta']]).round(1)
        else:
            res['pt'] = 5.0
            
        res['pn'] = (1 + (res['pt'] * 0.15)).round(2)
        
        return res

    def _sincronizar(self, malha):
        if not self.banco_nuvem or malha.empty: return
        col = self.banco_nuvem.collection('malha_seguranca')
        batch = self.banco_nuvem.batch()
        count = 0
        total_docs = 0

        for _, r in malha.iterrows():
            doc_id = f"{r['perfil'].lower()}_{r['h3']}"
            batch.set(col.document(doc_id), {
                "id_h3": r['h3'],
                "perfil": r['perfil'],
                "scores": {r['turno']: {"pt": r['pt'], "pn": r['pn']}},
                "data": firestore.SERVER_TIMESTAMP
            }, merge=True)
            
            count += 1
            total_docs += 1
            if count >= 400:
                batch.commit()
                batch = self.banco_nuvem.batch()
                count = 0
        batch.commit()
        self.auditoria['nuvem']['documentos'] = total_docs

    def executar(self):
        mestre = pd.DataFrame()
        metadata_path = 'datalake/bruto/metadata.json'
        metadata = {}
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            except: pass

        for ano in self.periodo_historico:
            caminho_bruto = f'datalake/bruto/ssp_{ano}.parquet'
            url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            df_ano = pd.DataFrame()
            tamanho_remoto = None
            
            try:
                head_req = self.sessao_rede.head(url, timeout=30)
                if head_req.status_code == 200 and 'Content-Length' in head_req.headers:
                    tamanho_remoto = int(head_req.headers['Content-Length'])
            except: pass

            precisa_baixar = False
            if not os.path.exists(caminho_bruto):
                precisa_baixar = True
            elif tamanho_remoto and metadata.get(str(ano)) != tamanho_remoto:
                precisa_baixar = True

            if precisa_baixar:
                try:
                    r = self.sessao_rede.get(url, timeout=300)
                    if r.status_code == 200:
                        df_ano = pd.read_excel(io.BytesIO(r.content), dtype=str)
                        df_ano.columns = [self._normalizar_coluna(c) for c in df_ano.columns]
                        df_ano = df_ano.loc[:, ~df_ano.columns.duplicated()].copy()
                        
                        colunas_essenciais = ['LATITUDE', 'LONGITUDE', 'NATUREZA_APURADA', 'LOCAL', 'HORA_OCORRENCIA_BO', 'DATA_OCORRENCIA_BO']
                        for col in colunas_essenciais:
                            if col not in df_ano.columns: df_ano[col] = ""
                                
                        df_ano['LATITUDE'] = df_ano['LATITUDE'].apply(lambda x: self._corrigir_gps(x, True))
                        df_ano['LONGITUDE'] = df_ano['LONGITUDE'].apply(lambda x: self._corrigir_gps(x, False))
                        df_ano.to_parquet(caminho_bruto, index=False)
                        
                        if tamanho_remoto:
                            metadata[str(ano)] = tamanho_remoto
                            with open(metadata_path, 'w') as f:
                                json.dump(metadata, f)
                except Exception:
                    if os.path.exists(caminho_bruto):
                        df_ano = pd.read_parquet(caminho_bruto)
            else:
                df_ano = pd.read_parquet(caminho_bruto)
            
            if not df_ano.empty:
                self.auditoria['camadas']['bruta'] += len(df_ano)
                val = df_ano[df_ano['LATITUDE'].notna()].copy()
                self.auditoria['camadas']['confiavel'] += len(val)
                mestre = pd.concat([mestre, val])
                
            del df_ano; gc.collect()

        if not mestre.empty:
            prev = self._processar_ia(mestre)
            self.auditoria['camadas']['refinada'] = len(prev)
            
            for p in self.auditoria['perfis'].keys():
                self.auditoria['perfis'][p] = int(len(prev[prev['perfil'] == p]))
                
            prev.to_parquet('datalake/refinado/malha_final.parquet', index=False)
            self._sincronizar(prev)
            self._notificar()

    def _notificar(self):
        webhook = os.environ.get('DISCORD_SUCESSO')
        if not webhook: return
        perf_str = "\n".join([f"**{k}:** {v} células de risco" for k, v in self.auditoria['perfis'].items()])
        payload = {
            "embeds": [{
                "title": f"🚀 {self.identidade} - Relatório de Missão",
                "color": 3066993,
                "fields": [
                    {"name": "🎯 Taxa de Acerto IA", "value": f"**{self.auditoria['metricas']['acerto']}%**", "inline": False},
                    {"name": "🌊 Lakehouse", "value": f"Bruto (Crimes): {self.auditoria['camadas']['bruta']}\nConfiavel (GPS): {self.auditoria['camadas']['confiavel']}\nCélulas Malha: {self.auditoria['camadas']['refinada']}", "inline": True},
                    {"name": "📉 Margem Erro", "value": f"MAE: {self.auditoria['metricas']['mae']}\nRMSE: {self.auditoria['metricas']['rmse']}", "inline": True},
                    {"name": "👥 Perfis na Malha", "value": perf_str, "inline": False},
                    {"name": "☁️ Nuvem", "value": f"{self.auditoria['nuvem']['documentos']} Docs Atualizados", "inline": True}
                ]
            }]
        }
        requests.post(webhook, json=payload)

if __name__ == "__main__":
    AutobotSafeDriver().executar()
