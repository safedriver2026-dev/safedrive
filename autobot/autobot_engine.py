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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN

class MotorSafeDriver:
    def __init__(self, persistencia=True):
        self.identidade = "Autobot SafeDriver Engine v3.0"
        self.ano_vigente = datetime.now().year
        self.periodo_historico = range(2022, self.ano_vigente + 1)
        self.persistencia_ativa = persistencia
        self.banco_dados = self._estabelecer_conexao_firestore() if persistencia else None
        self.sessao_http = self._configurar_sessao_resiliente()
        self.auditoria = {
            "volume_raw": 0, "volume_trusted": 0, "volume_refined": 0,
            "registros_inconsistentes": 0, "manchas_identificadas": 0,
            "h3_avaliados": 0, "h3_mutados": 0, "dados_origem_atualizados": False,
            "modo": "OPERACIONAL"
        }
        
        self.catalogo_conhecimento = {
            "LATROCINIO": {"peso": 5.0}, "EXTORSAO MEDIANTE SEQUESTRO": {"peso": 5.0},
            "ROUBO DE VEICULO": {"peso": 4.0}, "ROUBO DE CARGA": {"peso": 4.0},
            "ROUBO A TRANSEUNTE": {"peso": 4.0}, "FURTO DE VEICULO": {"peso": 3.0},
            "FURTO DE CARGA": {"peso": 3.0}, "FURTO DE CELULAR": {"peso": 3.0},
            "DANO": {"peso": 2.0}, "OUTROS": {"peso": 1.0}
        }
        self.matriz_comportamental = {
            "Ciclista": ["BICI", "CICLO", "BICICLETA", "PEDALAR"],
            "Motociclista": ["MOTO", "MOTOCICLETA", "CAPACETE", "MOTOBOY"],
            "Motorista": ["VEICULO", "CARGA", "CARRO", "CAMINHAO", "AUTOMOVEL"],
            "Pedestre": ["TRANSEUNTE", "CELULAR", "PEDESTRE", "CALCADA", "PONTO DE ONIBUS"]
        }
        self.limites_territoriais = {"lat": (-25.5, -19.5), "lon": (-53.5, -44.0)}
        self.esquema_canonico = {
            "NUM_BO": "string", "DATA_OCORRENCIA_BO": "datetime", "HORA_OCORRENCIA_BO": "string",
            "LATITUDE": "float", "LONGITUDE": "float", "NATUREZA_APURADA": "string", 
            "DESCR_TIPOLOCAL": "string", "ANO_BASE": "int"
        }
        self.dicionario_semantico = {
            "NUM_BO": ["NUM_BO", "NUMERO_BO", "BO_NUMERO", "N_BO"],
            "DATA_OCORRENCIA_BO": ["DATA_OCORRENCIA_BO", "DATA_FATO", "DATAOCORRENCIA", "DATA"],
            "HORA_OCORRENCIA_BO": ["HORA_OCORRENCIA_BO", "HORA_FATO", "HORAOCORRENCIA"],
            "LATITUDE": ["LATITUDE", "LAT", "LATITUDEDECIMAL", "LATITUDE_Y"],
            "LONGITUDE": ["LONGITUDE", "LON", "LONG", "LONGITUDEDECIMAL", "LONGITUDE_X"],
            "NATUREZA_APURADA": ["NATUREZA_APURADA", "NATUREZA", "TIPO_CRIME", "RUBRICA"],
            "DESCR_TIPOLOCAL": ["DESCR_TIPOLOCAL", "TIPOLOCAL", "LOCAL", "TIPO_LOCAL"]
        }

        self._iniciar_sistemas_internos()

    def _iniciar_sistemas_internos(self):
        for pasta in ['raw', 'trusted', 'refined', 'metadata']:
            os.makedirs(f'datalake/{pasta}', exist_ok=True)
        if not os.path.exists('datalake/metadata/baseline.lock'):
            self.auditoria["modo"] = "HARD_RESET"

    def _estabelecer_conexao_firestore(self):
        chave_json = os.environ.get('FIREBASE_JSON')
        if not chave_json or not firebase_admin._apps:
            if chave_json:
                credenciais = credentials.Certificate(json.loads(chave_json))
                firebase_admin.initialize_app(credenciais)
                return firestore.client()
        return None

    def _configurar_sessao_resiliente(self):
        sessao = requests.Session()
        politica = Retry(total=5, backoff_factor=2, status_forcelist=[403, 429, 500, 502, 503, 504])
        sessao.mount("http://", HTTPAdapter(max_retries=politica))
        sessao.mount("https://", HTTPAdapter(max_retries=politica))
        sessao.headers.update({'User-Agent': self.identidade})
        return sessao

    def _comunicar_status(self, nivel, mensagem=None):
        webhook = os.environ.get('DISCORD_AUDITORIA')
        if not webhook or not self.persistencia_ativa: return
        
        status = "Novos dados processados." if self.auditoria['dados_origem_atualizados'] else "Predicao de D+1 finalizada via Delta Sync."
        if nivel == "OPERACIONAL":
            payload = {
                "embeds": [{
                    "title": f"Terminal Autobot: {self.identidade}",
                    "description": status,
                    "color": 3066993,
                    "fields": [
                        {"name": "Ingestao", "value": f"Raw: {self.auditoria['volume_raw']}\nRefined: {self.auditoria['volume_refined']}", "inline": False},
                        {"name": "Sync Nuvem", "value": f"H3 Avaliados: {self.auditoria['h3_avaliados']}\nH3 Mutados (Delta): {self.auditoria['h3_mutados']}", "inline": False}
                    ]
                }]
            }
        else:
            payload = {"embeds": [{"title": "Falha Critica", "description": str(mensagem), "color": 15158332}]}
        requests.post(webhook, json=payload)

    def _higienizar_string(self, texto):
        if pd.isna(texto) or not isinstance(texto, str): return str(texto) if not pd.isna(texto) else ""
        return "".join([c for c in unicodedata.normalize('NFKD', texto) if not unicodedata.combining(c)]).upper().strip()

    def _normalizar(self, coluna):
        limpa = self._higienizar_string(coluna)
        for k, v in self.dicionario_semantico.items():
            if limpa in v: return k
        return limpa

    def _classificar_crime(self, natureza):
        if pd.isna(natureza): return np.nan
        limpa = self._higienizar_string(natureza)
        return limpa if limpa in self.catalogo_conhecimento else "OUTROS"

    def _corrigir_ponto_decimal(self, valor, is_lat=True):
        try:
            v = float(str(valor).replace(',', '.'))
            limites = self.limites_territoriais['lat'] if is_lat else self.limites_territoriais['lon']
            if v < limites[0] or v > limites[1]:
                v = v / (10 ** (len(str(int(abs(v)))) - 2))
            return v
        except: return np.nan

    def _verificar_fonte(self, url, ano):
        if self.auditoria["modo"] == "HARD_RESET": return True, 0
        metadados = f"datalake/metadata/fonte_{ano}.json"
        try:
            cabecalho = self.sessao_http.head(url, timeout=30, allow_redirects=True)
            tamanho = int(cabecalho.headers.get('Content-Length', 0))
            if os.path.exists(metadados):
                with open(metadados, 'r') as f:
                    if json.load(f).get('tamanho') == tamanho: return False, tamanho
            return True, tamanho
        except: return True, 0

    def _qualificar(self, df):
        df_bruto = df.copy()
        for col in self.esquema_canonico.keys():
            if col not in df_bruto.columns:
                if col == 'ANO_BASE': df_bruto[col] = self.ano_vigente
                elif col == 'HORA_OCORRENCIA_BO': df_bruto[col] = "00:00:00"
                else: df_bruto[col] = np.nan

        for col, tipo in self.esquema_canonico.items():
            if tipo == 'string': df_bruto[col] = df_bruto[col].apply(self._higienizar_string)
            elif tipo == 'float': df_bruto[col] = df_bruto[col].apply(lambda x: self._corrigir_ponto_decimal(x, col == 'LATITUDE'))
            elif tipo == 'datetime': df_bruto[col] = pd.to_datetime(df_bruto[col], errors='coerce')
            elif tipo == 'int': df_bruto[col] = pd.to_numeric(df_bruto[col], errors='coerce').fillna(0).astype(int)

        df_bruto['NATUREZA_APURADA'] = df_bruto['NATUREZA_APURADA'].apply(self._classificar_crime)
        mascara = (df_bruto['LATITUDE'].notna() & df_bruto['LONGITUDE'].notna() &
                   df_bruto['LATITUDE'].between(self.limites_territoriais['lat'][0], self.limites_territoriais['lat'][1]) &
                   df_bruto['LONGITUDE'].between(self.limites_territoriais['lon'][0], self.limites_territoriais['lon'][1]))
        
        df_trusted = df_bruto[mascara].copy()[list(self.esquema_canonico.keys())]
        df_refined = df_trusted[df_trusted['NATUREZA_APURADA'].notna()].copy()
        return df_trusted, df_refined

    def _mapear_territorio(self, df):
        df['perfil_alvo'] = df.apply(lambda r: [p for p, words in self.matriz_comportamental.items() if any(w in f"{r['NATUREZA_APURADA']} {r['DESCR_TIPOLOCAL']}".upper() for w in words)] or ['Geral'], axis=1)
        df = df.explode('perfil_alvo')
        vetor = TfidfVectorizer(max_features=50).fit_transform(df['NATUREZA_APURADA'].fillna('') + " " + df['DESCR_TIPOLOCAL'].fillna(''))
        df['vetor_comportamental'] = KMeans(n_clusters=4, random_state=42, n_init="auto").fit_predict(vetor)
        df['gravidade'] = df['NATUREZA_APURADA'].apply(lambda x: self.catalogo_conhecimento.get(x, {}).get('peso', 1.0))
        df['id_hotspot'] = DBSCAN(eps=0.001, min_samples=3, metric='euclidean').fit_predict(df[['LATITUDE', 'LONGITUDE']].values)
        return df[df['id_hotspot'] != -1].copy()

    def _processar_predicao_diaria(self, df_hotspots):
        df_hotspots['turno'] = df_hotspots['HORA_OCORRENCIA_BO'].apply(lambda x: 'Madrugada' if 0<=int(str(x).split(':')[0])<6 else 'Manha' if 6<=int(str(x).split(':')[0])<12 else 'Tarde' if 12<=int(str(x).split(':')[0])<18 else 'Noite')
        enc_t, enc_p = LabelEncoder(), LabelEncoder()
        df_hotspots['turno_enc'], df_hotspots['perfil_enc'] = enc_t.fit_transform(df_hotspots['turno']), enc_p.fit_transform(df_hotspots['perfil_alvo'])
        df_hotspots['DATA_OCORRENCIA_BO'] = pd.to_datetime(df_hotspots['DATA_OCORRENCIA_BO'])
        
        serie = df_hotspots.groupby('DATA_OCORRENCIA_BO').size().reset_index(name='y').rename(columns={'DATA_OCORRENCIA_BO': 'ds'})
        prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False).fit(serie)
        
        baseline = df_hotspots.groupby(['id_hotspot', 'perfil_enc', 'vetor_comportamental', 'turno_enc']).agg({'LATITUDE': 'mean', 'LONGITUDE': 'mean', 'gravidade': ['mean', 'count']}).reset_index()
        baseline.columns = ['id_hotspot', 'perfil_enc', 'vetor_comportamental', 'turno_enc', 'lat', 'lon', 'risco_base', 'volume']
        
        df_t = df_hotspots.groupby(['id_hotspot', 'perfil_enc', 'vetor_comportamental', 'turno_enc', 'DATA_OCORRENCIA_BO']).agg({'gravidade': 'sum'}).reset_index()
        df_t = df_t.merge(baseline, on=['id_hotspot', 'perfil_enc', 'vetor_comportamental', 'turno_enc'])
        df_t['sazonalidade'] = prophet.predict(df_t[['DATA_OCORRENCIA_BO']].rename(columns={'DATA_OCORRENCIA_BO': 'ds'}))['yhat'].values
        
        X = df_t[['perfil_enc', 'vetor_comportamental', 'turno_enc', 'risco_base', 'sazonalidade']]
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100).fit(X, df_t['gravidade'])
        
        amanha = datetime.now() + timedelta(days=1)
        df_f = baseline.copy()
        df_f['sazonalidade'] = prophet.predict(pd.DataFrame({'ds': [amanha]}))['yhat'].values[0]
        df_f['pred'] = model.predict(df_f[['perfil_enc', 'vetor_comportamental', 'turno_enc', 'risco_base', 'sazonalidade']])
        
        df_f['score'] = ((df_f['pred'] * df_f['volume']) / (df_f['pred'] * df_f['volume']).max() * 10).clip(0.5, 10.0).round(1)
        df_f['penalidade'] = (1 + (df_f['score'] * 0.2)).round(1)
        df_f['codigo_h3'] = df_f.apply(lambda r: h3.latlng_to_cell(r['lat'], r['lon'], 10), axis=1)
        df_f['turno_desc'] = enc_t.inverse_transform(df_f['turno_enc'])
        
        self.auditoria['manchas_identificadas'] = len(df_f['id_hotspot'].unique())
        return df_f.groupby(['codigo_h3', 'turno_desc']).agg({'score': 'max', 'penalidade': 'max', 'vetor_comportamental': 'first'}).reset_index()

    def _sincronizar_delta(self, df_h3):
        if not self.banco_dados or not self.persistencia_ativa: return
        
        colecao = self.banco_dados.collection('malha_preditiva_diaria')
        turnos = df_h3['turno_desc'].unique()
        self.auditoria['h3_avaliados'] = len(df_h3)

        if self.auditoria["modo"] == "HARD_RESET":
            for doc in colecao.stream(): colecao.document(doc.id).delete()

        for turno in turnos:
            df_turno = df_h3[df_h3['turno_desc'] == turno]
            ref_doc = colecao.document(f"turno_{turno}")
            doc_snap = ref_doc.get()
            dados_nuvem = doc_snap.to_dict().get("h3_dados", {}) if doc_snap.exists else {}
            
            payload_delta = {}
            for _, linha in df_turno.iterrows():
                h3_id = linha['codigo_h3']
                s = round(float(linha['score']), 1)
                p = round(float(linha['penalidade']), 1)
                c = int(linha['vetor_comportamental'])
                
                novo_dado = {"score": s, "penalidade": p, "cluster": c}
                if h3_id not in dados_nuvem or dados_nuvem[h3_id] != novo_dado:
                    payload_delta[f"h3_dados.{h3_id}"] = novo_dado
                    self.auditoria['h3_mutados'] += 1

            if payload_delta:
                payload_delta["ultima_atualizacao"] = firestore.SERVER_TIMESTAMP
                if doc_snap.exists:
                    ref_doc.update(payload_delta)
                else:
                    estrutura_inicial = {"h3_dados": {k.split('.')[1]: v for k, v in payload_delta.items() if k != "ultima_atualizacao"}, "ultima_atualizacao": firestore.SERVER_TIMESTAMP}
                    ref_doc.set(estrutura_inicial)
                    
        if self.auditoria["modo"] == "HARD_RESET":
            with open('datalake/metadata/baseline.lock', 'w') as f: f.write(str(datetime.now()))
            self.auditoria["modo"] = "OPERACIONAL"

    def rodar(self):
        df_master = pd.DataFrame()
        try:
            for ano in self.periodo_historico:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                caminho = f'datalake/raw/ssp_{ano}.parquet'
                
                baixar, tamanho = self._verificar_fonte(url, ano)
                if baixar or not os.path.exists(caminho):
                    res = self.sessao_http.get(url, timeout=120)
                    if res.status_code != 200: continue
                    amostra = pd.read_excel(io.BytesIO(res.content), nrows=50, header=None)
                    header = next((i for i, l in amostra.iterrows() if any(t in [self._higienizar_string(str(c)) for c in l.values] for t in ['NUM_BO', 'LATITUDE', 'NATUREZA_APURADA'])), None)
                    if header is None: continue
                    df_temp = pd.read_excel(io.BytesIO(res.content), skiprows=header, dtype=str)
                    df_temp.columns = [self._normalizar(c) for c in df_temp.columns]
                    df_temp.to_parquet(caminho, index=False)
                    with open(f"datalake/metadata/fonte_{ano}.json", 'w') as f: json.dump({'tamanho': tamanho}, f)
                    self.auditoria['dados_origem_atualizados'] = True
                else:
                    df_temp = pd.read_parquet(caminho)

                df_temp['ANO_BASE'] = ano
                self.auditoria['volume_raw'] += len(df_temp)
                df_trusted, df_refined = self._qualificar(df_temp)
                
                if self.persistencia_ativa: df_trusted.to_parquet(f'datalake/trusted/ssp_trusted_{ano}.parquet', index=False)
                df_master = pd.concat([df_master, df_refined])
                self.auditoria['volume_trusted'] += len(df_trusted)
                self.auditoria['volume_refined'] += len(df_refined)

            if df_master.empty: return

            df_manchas = self._mapear_territorio(df_master)
            malha_h3 = self._processar_predicao_diaria(df_manchas)
            
            if self.persistencia_ativa:
                malha_h3.to_parquet("datalake/refined/malha_h3_diaria.parquet", index=False)
            
            self._sincronizar_delta(malha_h3)
            self._comunicar_status("OPERACIONAL")

        except Exception as e:
            self._comunicar_status("FALHA", str(e))
            raise

if __name__ == "__main__":
    motor = MotorSafeDriver()
    motor.rodar()
