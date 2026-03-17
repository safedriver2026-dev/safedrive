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
from sklearn.metrics import mean_absolute_error, mean_squared_error

class MotorSafeDriver:
    def __init__(self, persistencia=True):
        self.identidade = "Autobot SafeDriver Engine v3.5"
        self.ano_vigente = datetime.now().year
        self.periodo_historico = range(2022, self.ano_vigente + 1)
        self.persistencia_ativa = persistencia
        self.banco_dados = self._estabelecer_conexao_firestore() if persistencia else None
        self.sessao_http = self._configurar_sessao_resiliente()
        self.auditoria = {
            "volume_raw": 0, "volume_trusted": 0, "volume_refined": 0,
            "h3_avaliados": 0, "h3_mutados": 0, "perfis_h3": {},
            "dados_origem_atualizados": False, "modo": "OPERACIONAL",
            "mae": 0.0, "rmse": 0.0
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

        for p in ['raw', 'trusted', 'refined', 'metadata']: os.makedirs(f'datalake/{p}', exist_ok=True)
        if not os.path.exists('datalake/metadata/baseline.lock'): self.auditoria["modo"] = "HARD_RESET"

    def _estabelecer_conexao_firestore(self):
        chave_json = os.environ.get('FIREBASE_JSON')
        if chave_json and not firebase_admin._apps:
            credenciais = credentials.Certificate(json.loads(chave_json))
            firebase_admin.initialize_app(credenciais)
            return firestore.client()
        return None

    def _configurar_sessao_resiliente(self):
        sessao = requests.Session()
        politica = Retry(total=5, backoff_factor=3, status_forcelist=[403, 429, 500, 502, 503, 504])
        sessao.mount("http://", HTTPAdapter(max_retries=politica))
        sessao.mount("https://", HTTPAdapter(max_retries=politica))
        sessao.headers.update({'User-Agent': 'Mozilla/5.0 (SafeDriver Autobot)'})
        return sessao

    def _comunicar_status(self, nivel, mensagem=None):
        webhook_sucesso = os.environ.get('DISCORD_SUCESSO')
        webhook_erro = os.environ.get('DISCORD_ERRO')
        if not self.persistencia_ativa: return
        
        if nivel == "OPERACIONAL" and webhook_sucesso:
            dist_perfis = "\n".join([f"**{k}:** {v}" for k, v in self.auditoria['perfis_h3'].items()])
            payload = {
                "embeds": [{
                    "title": "📋 SafeDriver Engine - Relatório de Inteligência",
                    "description": f"**Status:** {self.auditoria['modo']}\n**Previsão:** Amanhã (D+1)",
                    "color": 3066993,
                    "fields": [
                        {"name": "🌊 Data Lake", "value": f"RAW: {self.auditoria['volume_raw']}\nREFINED: {self.auditoria['volume_refined']}", "inline": True},
                        {"name": "☁️ Sincronização", "value": f"H3 Avaliados: {self.auditoria['h3_avaliados']}\nH3 Mutados: {self.auditoria['h3_mutados']}", "inline": True},
                        {"name": "📈 Precisão (MAE/RMSE)", "value": f"{self.auditoria['mae']} / {self.auditoria['rmse']}", "inline": False},
                        {"name": "🎯 Hexágonos por Perfil", "value": dist_perfis or "Processando...", "inline": False}
                    ],
                    "footer": {"text": self.identidade}
                }]
            }
            requests.post(webhook_sucesso, json=payload)
        elif nivel == "FALHA" and webhook_erro:
            requests.post(webhook_erro, json={"content": f"⚠️ **FALHA NO SUBSISTEMA:** {mensagem}"})

    def _higienizar_string(self, texto):
        if pd.isna(texto) or not isinstance(texto, str): return str(texto) if not pd.isna(texto) else ""
        return "".join([c for c in unicodedata.normalize('NFKD', texto) if not unicodedata.combining(c)]).upper().strip()

    def _normalizar(self, col):
        limpa = self._higienizar_string(col)
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

    def _download_resiliente(self, url):
        with self.sessao_http.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            buffer = io.BytesIO()
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk: buffer.write(chunk)
            buffer.seek(0)
            return buffer

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

    def _processar_ia_preditiva(self, df):
        df['perfil_alvo'] = df.apply(lambda r: [p for p, words in self.matriz_comportamental.items() if any(w in f"{r['NATUREZA_APURADA']} {r['DESCR_TIPOLOCAL']}".upper() for w in words)] or ['Geral'], axis=1)
        df = df.explode('perfil_alvo')
        df['id_hotspot'] = DBSCAN(eps=0.001, min_samples=3).fit_predict(df[['LATITUDE', 'LONGITUDE']].values)
        df = df[df['id_hotspot'] != -1].copy()
        df['turno'] = df['HORA_OCORRENCIA_BO'].apply(lambda x: 'Madrugada' if 0<=int(str(x).split(':')[0])<6 else 'Manha' if 6<=int(str(x).split(':')[0])<12 else 'Tarde' if 12<=int(str(x).split(':')[0])<18 else 'Noite')
        df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'])
        serie = df.groupby('DATA_OCORRENCIA_BO').size().reset_index(name='y').rename(columns={'DATA_OCORRENCIA_BO': 'ds'})
        prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False).fit(serie)
        enc_t, enc_p = LabelEncoder(), LabelEncoder()
        df['turno_enc'], df['perfil_enc'] = enc_t.fit_transform(df['turno']), enc_p.fit_transform(df['perfil_alvo'])
        df['gravidade'] = df['NATUREZA_APURADA'].apply(lambda x: self.catalogo_conhecimento.get(x, {}).get('peso', 1.0))
        baseline = df.groupby(['id_hotspot', 'perfil_enc', 'turno_enc']).agg({'LATITUDE': 'mean', 'LONGITUDE': 'mean', 'gravidade': ['mean', 'count']}).reset_index()
        baseline.columns = ['id_hotspot', 'perfil_enc', 'turno_enc', 'lat', 'lon', 'risco_base', 'volume']
        df_t = df.groupby(['id_hotspot', 'perfil_enc', 'turno_enc', 'DATA_OCORRENCIA_BO']).agg({'gravidade': 'sum'}).reset_index()
        df_t = df_t.merge(baseline, on=['id_hotspot', 'perfil_enc', 'turno_enc'])
        df_t['sazonalidade'] = prophet.predict(df_t[['DATA_OCORRENCIA_BO']].rename(columns={'DATA_OCORRENCIA_BO': 'ds'}))['yhat'].values
        X, y = df_t[['perfil_enc', 'turno_enc', 'risco_base', 'sazonalidade']], df_t['gravidade']
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100).fit(X, y)
        self.auditoria['mae'] = round(mean_absolute_error(y, model.predict(X)), 3)
        self.auditoria['rmse'] = round(np.sqrt(mean_squared_error(y, model.predict(X))), 3)
        amanha = datetime.now() + timedelta(days=1)
        df_f = baseline.copy()
        df_f['sazonalidade'] = prophet.predict(pd.DataFrame({'ds': [amanha]}))['yhat'].values[0]
        df_f['pred'] = model.predict(df_f[['perfil_enc', 'turno_enc', 'risco_base', 'sazonalidade']])
        df_f['score'] = ((df_f['pred'] * df_f['volume']) / (df_f['pred'] * df_f['volume']).max() * 10).clip(0.5, 10.0).round(1)
        df_f['penalidade'] = (1 + (df_f['score'] * 0.2)).round(1)
        df_f['codigo_h3'] = df_f.apply(lambda r: h3.latlng_to_cell(r['lat'], r['lon'], 10), axis=1)
        df_f['turno_desc'] = enc_t.inverse_transform(df_f['turno_enc'])
        df_f['perfil_desc'] = enc_p.inverse_transform(df_f['perfil_enc'])
        self.auditoria['perfis_h3'] = df_f.groupby('perfil_desc')['codigo_h3'].nunique().to_dict()
        return df_f.groupby(['codigo_h3', 'turno_desc']).agg({'score': 'max', 'penalidade': 'max'}).reset_index()

    def _sincronizar_delta(self, df_h3):
        if not self.banco_dados: return
        colecao = self.banco_dados.collection('malha_preditiva_diaria')
        if self.auditoria["modo"] == "HARD_RESET":
            for d in colecao.stream(): d.reference.delete()
        for turno in df_h3['turno_desc'].unique():
            df_turno = df_h3[df_h3['turno_desc'] == turno]
            ref = colecao.document(f"turno_{turno}")
            snap = ref.get()
            dn = snap.to_dict().get("h3_dados", {}) if snap.exists else {}
            pd = {}
            for _, r in df_turno.iterrows():
                h, s, p = r['codigo_h3'], round(float(r['score']), 1), round(float(r['penalidade']), 1)
                nv = {"score": s, "penalidade": p}
                if h not in dn or dn[h] != nv:
                    pd[f"h3_dados.{h}"] = nv
                    self.auditoria['h3_mutados'] += 1
            if pd:
                pd["ultima_atualizacao"] = firestore.SERVER_TIMESTAMP
                if snap.exists: ref.update(pd)
                else: ref.set({"h3_dados": {k.split('.')[1]: v for k, v in pd.items() if k != "ultima_atualizacao"}, "ultima_atualizacao": firestore.SERVER_TIMESTAMP})
        self.auditoria['h3_avaliados'] = len(df_h3)

    def rodar(self):
        df_master = pd.DataFrame()
        try:
            for ano in self.periodo_historico:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                buffer = self._download_resiliente(url)
                df_temp = pd.read_excel(buffer, skiprows=0, dtype=str)
                df_temp.columns = [self._normalizar(c) for c in df_temp.columns]
                df_temp = df_temp.loc[:, ~df_temp.columns.duplicated()].copy()
                t, r = self._qualificar(df_temp)
                df_master = pd.concat([df_master, r])
                self.auditoria['volume_raw'] += len(df_temp)
                self.auditoria['volume_refined'] += len(r)
            if not df_master.empty:
                malha_h3 = self._processar_ia_preditiva(df_master)
                self._sincronizar_delta(malha_h3)
                self._comunicar_status("OPERACIONAL")
        except Exception as e:
            self._comunicar_status("FALHA", str(e))
            raise

if __name__ == "__main__":
    MotorSafeDriver().rodar()
