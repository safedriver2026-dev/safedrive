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

class AutobotSafeDriver:
    def __init__(self, persistencia=True):
        self.identidade = "Autobot SafeDriver v4.1"
        self.ano_atual = datetime.now().year
        self.periodo_historico = range(2022, self.ano_atual + 1)
        self.persistencia_ativa = persistencia
        self.banco_nuvem = self._conectar_nuvem() if persistencia else None
        self.sessao_rede = self._gerar_sessao_resiliente()
        self.auditoria = {
            "volume_bruto": 0, "volume_confiavel": 0, "volume_refinado": 0,
            "h3_analisados": 0, "h3_alterados": 0, "perfis_mapeados": {},
            "dados_novos": False, "estado_operacional": "NORMAL",
            "erro_medio_absoluto": 0.0, "raiz_erro_quadratico": 0.0
        }
        self.biblioteca_crimes = {
            "LATROCINIO": {"peso": 5.0}, "EXTORSAO MEDIANTE SEQUESTRO": {"peso": 5.0},
            "ROUBO DE VEICULO": {"peso": 4.0}, "ROUBO DE CARGA": {"peso": 4.0},
            "ROUBO A TRANSEUNTE": {"peso": 4.0}, "FURTO DE VEICULO": {"peso": 3.0},
            "FURTO DE CARGA": {"peso": 3.0}, "FURTO DE CELULAR": {"peso": 3.0},
            "DANO": {"peso": 2.0}, "OUTROS": {"peso": 1.0}
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
        if not os.path.exists('datalake/metadados/bloqueio_inicial.lock'):
            self.auditoria["estado_operacional"] = "REINICIALIZACAO_TOTAL"

    def _conectar_nuvem(self):
        configuracao = os.environ.get('FIREBASE_JSON')
        if configuracao and not firebase_admin._apps:
            credencial = credentials.Certificate(json.loads(configuracao))
            firebase_admin.initialize_app(credencial)
            return firestore.client()
        return None

    def _gerar_sessao_resiliente(self):
        conexao = requests.Session()
        tentativas = Retry(total=5, backoff_factor=2, status_forcelist=[403, 429, 500, 502, 503, 504], allowed_methods=["HEAD", "GET", "OPTIONS"])
        adaptador = HTTPAdapter(max_retries=tentativas)
        conexao.mount("http://", adaptador)
        conexao.mount("https://", adaptador)
        conexao.headers.update({'User-Agent': 'Mozilla/5.0'})
        return conexao

    def _enviar_notificacao(self, tipo, aviso=None):
        canal_sucesso = os.environ.get('DISCORD_SUCESSO')
        canal_erro = os.environ.get('DISCORD_ERRO')
        if not self.persistencia_ativa: return
        if tipo == "SUCESSO" and canal_sucesso:
            resumo_perfis = "\n".join([f"**{perfil}:** {qtd}" for perfil, qtd in self.auditoria['perfis_mapeados'].items()])
            mensagem = {
                "embeds": [{
                    "title": f"🚀 {self.identidade} - Processamento Concluído",
                    "description": f"**Modo:** {self.auditoria['estado_operacional']}\n**Previsão:** Amanhã (D+1)",
                    "color": 3066993,
                    "fields": [
                        {"name": "🌊 Fluxo de Dados", "value": f"Bruto: {self.auditoria['volume_bruto']}\nRefinado: {self.auditoria['volume_refinado']}", "inline": True},
                        {"name": "☁️ Sincronização", "value": f"H3 Analisados: {self.auditoria['h3_analisados']}\nAlterações: {self.auditoria['h3_alterados']}", "inline": True},
                        {"name": "📉 Precisão (MAE/RMSE)", "value": f"{self.auditoria['erro_medio_absoluto']} / {self.auditoria['raiz_erro_quadratico']}", "inline": False},
                        {"name": "🎯 Densidade por Perfil", "value": resumo_perfis or "Calculando...", "inline": False}
                    ]
                }]
            }
            requests.post(canal_sucesso, json=mensagem)
        elif tipo == "ERRO" and canal_erro:
            requests.post(canal_erro, json={"content": f"⚠️ **FALHA NO AUTOBOT:** {aviso}"})

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
        resposta = self.sessao_rede.get(link, timeout=120)
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
        df_confiavel = copia[validados].copy()[list(self.estrutura_dados.keys())]
        df_refinado = df_confiavel[df_confiavel['NATUREZA_APURADA'].notna()].copy()
        return df_confiavel, df_refinado

    def _processar_ia(self, df):
        df['perfil_alvo'] = df.apply(lambda r: [p for p, palavras in self.mapa_comportamental.items() if any(pal in f"{r['NATUREZA_APURADA']} {r['DESCR_TIPOLOCAL']}".upper() for pal in palavras)] or ['Geral'], axis=1)
        df = df.explode('perfil_alvo')
        df['hotspot'] = DBSCAN(eps=0.001, min_samples=3).fit_predict(df[['LATITUDE', 'LONGITUDE']].values)
        df = df[df['hotspot'] != -1].copy()
        df['turno'] = df['HORA_OCORRENCIA_BO'].apply(lambda x: 'Madrugada' if 0<=int(str(x).split(':')[0])<6 else 'Manha' if 6<=int(str(x).split(':')[0])<12 else 'Tarde' if 12<=int(str(x).split(':')[0])<18 else 'Noite')
        df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'])
        historico = df.groupby('DATA_OCORRENCIA_BO').size().reset_index(name='y').rename(columns={'DATA_OCORRENCIA_BO': 'ds'})
        ia_temporal = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False).fit(historico)
        enc_t, enc_p = LabelEncoder(), LabelEncoder()
        df['turno_cod'], df['perfil_cod'] = enc_t.fit_transform(df['turno']), enc_p.fit_transform(df['perfil_alvo'])
        df['peso'] = df['NATUREZA_APURADA'].apply(lambda x: self.biblioteca_crimes.get(x, {}).get('peso', 1.0))
        base = df.groupby(['hotspot', 'perfil_cod', 'turno_cod']).agg({'LATITUDE': 'mean', 'LONGITUDE': 'mean', 'peso': ['mean', 'count']}).reset_index()
        base.columns = ['hotspot', 'perfil_cod', 'turno_cod', 'lat', 'lon', 'risco_base', 'volume']
        df_treino = df.groupby(['hotspot', 'perfil_cod', 'turno_cod', 'DATA_OCORRENCIA_BO']).agg({'peso': 'sum'}).reset_index()
        df_treino = df_treino.merge(base, on=['hotspot', 'perfil_cod', 'turno_cod'])
        df_treino['sazonalidade'] = ia_temporal.predict(df_treino[['DATA_OCORRENCIA_BO']].rename(columns={'DATA_OCORRENCIA_BO': 'ds'}))['yhat'].values
        X, y = df_treino[['perfil_cod', 'turno_cod', 'risco_base', 'sazonalidade']], df_treino['peso']
        ia_preditiva = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100).fit(X, y)
        self.auditoria['erro_medio_absoluto'] = round(mean_absolute_error(y, ia_preditiva.predict(X)), 3)
        self.auditoria['raiz_erro_quadratico'] = round(np.sqrt(mean_squared_error(y, ia_preditiva.predict(X))), 3)
        amanha = datetime.now() + timedelta(days=1)
        futuro = base.copy()
        futuro['sazonalidade'] = ia_temporal.predict(pd.DataFrame({'ds': [amanha]}))['yhat'].values[0]
        futuro['predicao'] = ia_preditiva.predict(futuro[['perfil_cod', 'turno_cod', 'risco_base', 'sazonalidade']])
        futuro['pontuacao'] = ((futuro['predicao'] * futuro['volume']) / (futuro['predicao'] * futuro['volume']).max() * 10).clip(0.5, 10.0).round(1)
        futuro['penalidade'] = (1 + (futuro['pontuacao'] * 0.2)).round(1)
        futuro['h3_index'] = futuro.apply(lambda r: h3.latlng_to_cell(r['lat'], r['lon'], 10), axis=1)
        futuro['turno_nome'] = enc_t.inverse_transform(futuro['turno_cod'])
        futuro['perfil_nome'] = enc_p.inverse_transform(futuro['perfil_cod'])
        self.auditoria['perfis_mapeados'] = futuro.groupby('perfil_nome')['h3_index'].nunique().to_dict()
        return futuro.groupby(['h3_index', 'turno_nome']).agg({'pontuacao': 'max', 'penalidade': 'max'}).reset_index()

    def _sincronizar(self, malha):
        if not self.banco_nuvem: return
        colecao = self.banco_nuvem.collection('malha_autobot_preditiva')
        if self.auditoria["estado_operacional"] == "REINICIALIZACAO_TOTAL":
            for doc in colecao.stream(): doc.reference.delete()
        for turno in malha['turno_nome'].unique():
            df_turno = malha[malha['turno_nome'] == turno]
            referencia = colecao.document(f"turno_{turno}")
            snapshot = referencia.get()
            dados_nuvem = snapshot.to_dict().get("dados_h3", {}) if snapshot.exists else {}
            pacote_atualizacao = {}
            for _, r in df_turno.iterrows():
                id_h3, pt, pn = r['h3_index'], round(float(r['pontuacao']), 1), round(float(r['penalidade']), 1)
                novo = {"pontuacao": pt, "penalidade": pn}
                if id_h3 not in dados_nuvem or dados_nuvem[id_h3] != novo:
                    pacote_atualizacao[f"dados_h3.{id_h3}"] = novo
                    self.auditoria['h3_alterados'] += 1
            if pacote_atualizacao:
                pacote_atualizacao["ultima_sincronizacao"] = firestore.SERVER_TIMESTAMP
                if snapshot.exists: referencia.update(pacote_atualizacao)
                else: referencia.set({"dados_h3": {k.split('.')[1]: v for k, v in pacote_atualizacao.items() if k != "ultima_sincronizacao"}, "ultima_sincronizacao": firestore.SERVER_TIMESTAMP})
        if self.auditoria["estado_operacional"] == "REINICIALIZACAO_TOTAL":
            with open('datalake/metadados/bloqueio_inicial.lock', 'w') as f: f.write(str(datetime.now()))
        self.auditoria['h3_analisados'] = len(malha)

    def executar(self):
        mestre = pd.DataFrame()
        try:
            for ano in self.periodo_historico:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                arquivo = self._baixar_dados(url)
                df_temp = pd.read_excel(arquivo, skiprows=0, dtype=str)
                df_temp.columns = [self._normalizar(c) for c in df_temp.columns]
                df_temp = df_temp.loc[:, ~df_temp.columns.duplicated()].copy()
                t, r = self._qualificar_dados(df_temp)
                mestre = pd.concat([mestre, r])
                self.auditoria['volume_bruto'] += len(df_temp)
                self.auditoria['volume_refinado'] += len(r)
            if not mestre.empty:
                malha_h3 = self._processar_ia(mestre)
                self._sincronizar(malha_h3)
                self._enviar_notificacao("SUCESSO")
        except Exception as e:
            self._enviar_notificacao("ERRO", str(e))
            raise

if __name__ == "__main__":
    AutobotSafeDriver().executar()
