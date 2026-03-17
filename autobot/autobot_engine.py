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
        self.identidade = "Autobot SafeDriver v4.8"
        self.ano_atual = datetime.now().year
        self.periodo_historico = range(2022, self.ano_atual + 1)
        self.persistencia_ativa = persistencia
        self.banco_nuvem = self._conectar_nuvem() if persistencia else None
        self.sessao_rede = self._gerar_sessao_resiliente()
        self.auditoria = {
            "volume_bruto": 0, "volume_confiavel": 0, "volume_refinado": 0,
            "hexagonos_processados": 0, "estado_operacional": "NORMAL",
            "erro_medio_absoluto": 0.0, "raiz_erro_quadratico": 0.0
        }
        self.biblioteca_crimes = {
            "LATROCINIO": 5.0, "EXTORSAO MEDIANTE SEQUESTRO": 5.0,
            "ROUBO DE VEICULO": 4.0, "ROUBO DE CARGA": 4.0,
            "ROUBO A TRANSEUNTE": 4.0, "FURTO DE VEICULO": 3.0,
            "FURTO DE CARGA": 3.0, "FURTO DE CELULAR": 3.0,
            "DANO": 2.0, "OUTROS": 1.0
        }
        self.mapa_comportamental = {
            "Ciclista": ["BICI", "CICLO", "BICICLETA", "PEDALAR"],
            "Motociclista": ["MOTO", "MOTOCICLETA", "CAPACETE", "MOTOBOY"],
            "Motorista": ["VEICULO", "CARGA", "CARRO", "CAMINHAO", "AUTOMOVEL"],
            "Pedestre": ["TRANSEUNTE", "CELULAR", "PEDESTRE", "CALCADA", "PONTO DE ONIBUS"]
        }
        self.limites_sao_paulo = {"lat": (-25.5, -19.5), "lon": (-53.5, -44.0)}
        self.estrutura_dados = {
            "NUM_BO": "string", "DATA_OCORRENCIA_BO": "datetime", "HORA_OCORRENCIA_BO": "string",
            "LATITUDE": "float", "LONGITUDE": "float", "NATUREZA_APURADA": "string", 
            "DESCR_TIPOLOCAL": "string", "ANO_BASE": "int"
        }
        self.sinonimos_colunas = {
            "NUM_BO": ["NUM_BO", "NUMERO_BO", "BO_NUMERO", "N_BO"],
            "DATA_OCORRENCIA_BO": ["DATA_OCORRENCIA_BO", "DATA_FATO", "DATAOCORRENCIA", "DATA"],
            "HORA_OCORRENCIA_BO": ["HORA_OCORRENCIA_BO", "HORA_FATO", "HORAOCORRENCIA"],
            "LATITUDE": ["LATITUDE", "LAT", "LATITUDEDECIMAL", "LATITUDE_Y"],
            "LONGITUDE": ["LONGITUDE", "LON", "LONG", "LONGITUDEDECIMAL", "LONGITUDE_X"],
            "NATUREZA_APURADA": ["NATUREZA_APURADA", "NATUREZA", "TIPO_CRIME", "RUBRICA"],
            "DESCR_TIPOLOCAL": ["DESCR_TIPOLOCAL", "TIPOLOCAL", "LOCAL", "TIPO_LOCAL"]
        }
        for subpasta in ['bruto', 'confiavel', 'refinado', 'metadados']:
            os.makedirs(f'datalake/{subpasta}', exist_ok=True)

    def _conectar_nuvem(self):
        configuracao = os.environ.get('FIREBASE_JSON')
        if configuracao and not firebase_admin._apps:
            credencial = credentials.Certificate(json.loads(configuracao))
            firebase_admin.initialize_app(credencial)
            return firestore.client()
        return None

    def _gerar_sessao_resiliente(self):
        conexao = requests.Session()
        tentativas = Retry(total=5, backoff_factor=2, status_forcelist=[403, 429, 500, 502, 503, 504])
        conexao.mount("https://", HTTPAdapter(max_retries=tentativas))
        conexao.headers.update({'User-Agent': 'Mozilla/5.0'})
        return conexao

    def _enviar_notificacao(self, tipo, aviso=None):
        canal_sucesso = os.environ.get('DISCORD_SUCESSO')
        canal_erro = os.environ.get('DISCORD_ERRO')
        if not self.persistencia_ativa: return
        if tipo == "SUCESSO" and canal_sucesso:
            payload = {
                "embeds": [{
                    "title": f"🚀 {self.identidade} - Missão Cumprida",
                    "color": 3066993,
                    "fields": [
                        {"name": "🌊 Fluxo", "value": f"Bruto: {self.auditoria['volume_bruto']}\nRefinado: {self.auditoria['volume_refinado']}", "inline": True},
                        {"name": "☁️ Nuvem", "value": f"Hexágonos: {self.auditoria['hexagonos_processados']}", "inline": True},
                        {"name": "📉 IA", "value": f"MAE: {self.auditoria['erro_medio_absoluto']}", "inline": False}
                    ]
                }]
            }
            requests.post(canal_sucesso, json=payload)
        elif tipo == "ERRO" and canal_erro:
            requests.post(canal_erro, json={"content": f"🚨 **FALHA:** {aviso}"})

    def _higienizar_texto(self, original):
        if pd.isna(original) or not isinstance(original, str): return str(original) if not pd.isna(original) else ""
        passo1 = unicodedata.normalize('NFKD', original)
        return "".join([c for c in passo1 if not unicodedata.combining(c)]).upper().strip()

    def _normalizar(self, nome_coluna):
        limpo = self._higienizar_texto(nome_coluna)
        for chave, lista in self.sinonimos_colunas.items():
            if limpo in lista: return chave
        return limpo

    def _classificar_crime(self, rubrica):
        if pd.isna(rubrica): return np.nan
        limpo = self._higienizar_texto(rubrica)
        return limpo if limpo in self.biblioteca_crimes else "OUTROS"

    def _corrigir_coordenadas(self, valor, latitude=True):
        try:
            v = float(str(valor).replace(',', '.'))
            limite = self.limites_sao_paulo['lat'] if latitude else self.limites_sao_paulo['lon']
            if v < limite[0] or v > limite[1]:
                v = v / (10 ** (len(str(int(abs(v)))) - 2))
            return v
        except: return np.nan

    def _baixar_dados(self, link):
        resposta = self.sessao_rede.get(link, timeout=180)
        resposta.raise_for_status()
        return io.BytesIO(resposta.content)

    def _qualificar_dados(self, df):
        copia = df.copy()
        for col in self.estrutura_dados.keys():
            if col not in copia.columns:
                if col == 'ANO_BASE': copia[col] = self.ano_atual
                elif col == 'HORA_OCORRENCIA_BO': copia[col] = "00:00:00"
                else: copia[col] = np.nan
        for col, tipo in self.estrutura_dados.items():
            if tipo == 'string': copia[col] = copia[col].apply(self._higienizar_texto)
            elif tipo == 'float': copia[col] = copia[col].apply(lambda x: self._corrigir_coordenadas(x, col == 'LATITUDE'))
            elif tipo == 'datetime': copia[col] = pd.to_datetime(copia[col], errors='coerce')
            elif tipo == 'int': copia[col] = pd.to_numeric(copia[col], errors='coerce').fillna(0).astype(int)
        copia['NATUREZA_APURADA'] = copia['NATUREZA_APURADA'].apply(self._classificar_crime)
        validados = (copia['LATITUDE'].notna() & copia['LONGITUDE'].notna() &
                   copia['LATITUDE'].between(self.limites_sao_paulo['lat'][0], self.limites_sao_paulo['lat'][1]) &
                   copia['LONGITUDE'].between(self.limites_sao_paulo['lon'][0], self.limites_sao_paulo['lon'][1]))
        return copia[validados].copy()[list(self.estrutura_dados.keys())], copia[validados & copia['NATUREZA_APURADA'].notna()].copy()

    def _processar_ia(self, df):
        if len(df) < 10: return pd.DataFrame()
        df['hotspot'] = DBSCAN(eps=0.001, min_samples=3).fit_predict(df[['LATITUDE', 'LONGITUDE']].values)
        df = df[df['hotspot'] != -1].copy()
        
        def extrair_hora(cel):
            try:
                p = str(cel).split(':')[0].strip()
                return int(p) if p else 0
            except: return 0
            
        df['turno'] = df['HORA_OCORRENCIA_BO'].apply(lambda x: 'Madrugada' if 0<=extrair_hora(x)<6 else 'Manha' if 6<=extrair_hora(x)<12 else 'Tarde' if 12<=extrair_hora(x)<18 else 'Noite')
        df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'])
        historico = df.groupby('DATA_OCORRENCIA_BO').size().reset_index(name='y').rename(columns={'DATA_OCORRENCIA_BO': 'ds'})
        
        fator_sazonal = 1.0
        if len(historico) >= 2:
            ia_temp = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False).fit(historico)
            prev = ia_temp.predict(pd.DataFrame({'ds': [datetime.now() + timedelta(days=1)]}))
            fator_sazonal = max(0.1, prev['yhat'].values[0] / historico['y'].mean())
        
        enc_t = LabelEncoder()
        df['turno_cod'] = enc_t.fit_transform(df['turno'])
        df['peso'] = df['NATUREZA_APURADA'].apply(lambda x: self.biblioteca_crimes.get(x, 1.0))
        
        ia_pred = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=50).fit(df[['LATITUDE', 'LONGITUDE', 'turno_cod']], df['peso'])
        self.auditoria['erro_medio_absoluto'] = round(mean_absolute_error(df['peso'], ia_pred.predict(df[['LATITUDE', 'LONGITUDE', 'turno_cod']])), 3)
        
        df['h3'] = df.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 10), axis=1)
        res = df.groupby(['h3', 'turno']).agg({'peso': 'mean'}).reset_index()
        res['pt'] = (res['peso'] * fator_sazonal * 10).clip(0.5, 10.0).round(1)
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
        try:
            for ano in self.periodo_historico:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                try:
                    arq = self._baixar_dados(url)
                    df_raw = pd.read_excel(arq, dtype=str)
                    df_raw.columns = [self._normalizar(c) for c in df_raw.columns]
                    df_raw = df_raw.loc[:, ~df_raw.columns.duplicated()].copy()
                    df_raw.to_parquet(f'datalake/bruto/{ano}.parquet', index=False)
                    self.auditoria['volume_bruto'] += len(df_raw)
                    conf, ref = self._qualificar_dados(df_raw)
                    conf.to_parquet(f'datalake/confiavel/{ano}.parquet', index=False)
                    mestre = pd.concat([mestre, ref])
                    self.auditoria['volume_refinado'] += len(ref)
                except Exception as e_ano:
                    print(f"Erro no ano {ano}: {e_ano}")
                    continue
            if not mestre.empty:
                self._sincronizar(self._processar_ia(mestre))
                self._enviar_notificacao("SUCESSO")
        except Exception as e:
            self._enviar_notificacao("ERRO", str(e))
            raise

if __name__ == "__main__":
    AutobotSafeDriver().executar()
