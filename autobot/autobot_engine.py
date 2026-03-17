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
        self.identidade = "Autobot SafeDriver Engine v2.1"
        self.ano_vigente = datetime.now().year
        self.periodo_historico = range(2022, self.ano_vigente + 1)
        self.persistencia_ativa = persistencia
        self.banco_dados = self._estabelecer_conexao_firestore() if persistencia else None
        self.sessao_http = self._configurar_sessao_resiliente()
        self.auditoria = {
            "volume_raw": 0, "volume_trusted": 0, "volume_refined": 0,
            "registros_inconsistentes": 0, "manchas_identificadas": 0,
            "h3_avaliados": 0, "dados_origem_atualizados": False,
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
        politica_retentativa = Retry(total=5, backoff_factor=2, status_forcelist=[403, 429, 500, 502, 503, 504])
        adaptador = HTTPAdapter(max_retries=politica_retentativa)
        sessao.mount("http://", adaptador)
        sessao.mount("https://", adaptador)
        sessao.headers.update({'User-Agent': self.identidade})
        return sessao

    def _comunicar_status(self, nivel, mensagem=None):
        endereco_webhook = os.environ.get('DISCORD_AUDITORIA')
        if not endereco_webhook or not self.persistencia_ativa: return

        if nivel == "OPERACIONAL":
            status = "Novos dados detectados na grade." if self.auditoria['dados_origem_atualizados'] else "Matriz preditiva retreinada sem alterações na fonte."
            if self.auditoria["modo"] == "HARD_RESET": status = "Sincronização primária e expurgo de dados concluídos (HARD_RESET)."
            
            payload = {
                "embeds": [{
                    "title": f"Terminal de Status: {self.identidade}",
                    "description": status,
                    "color": 3066993,
                    "fields": [
                        {"name": "Processamento de Dados", "value": f"Ingeridos: {self.auditoria['volume_raw']:,}\nRefinados: {self.auditoria['volume_refined']:,}", "inline": False},
                        {"name": "Inteligência Espacial", "value": f"Hotspots (DBSCAN): {self.auditoria['manchas_identificadas']:,}\nMalha H3 Atualizada: {self.auditoria['h3_avaliados']:,}", "inline": False}
                    ]
                }]
            }
        else:
            payload = {
                "embeds": [{
                    "title": "Aviso de Falha no Subsistema",
                    "color": 15158332,
                    "fields": [{"name": "Log de Erro", "value": str(mensagem), "inline": False}]
                }]
            }
        requests.post(endereco_webhook, json=payload)

    def _higienizar_string(self, texto):
        if pd.isna(texto) or not isinstance(texto, str): return str(texto) if not pd.isna(texto) else ""
        texto_limpo = unicodedata.normalize('NFKD', texto)
        return "".join([c for c in texto_limpo if not unicodedata.combining(c)]).upper().strip()

    def _normalizar(self, nome_coluna):
        nome_limpo = self._higienizar_string(nome_coluna)
        for chave_canonica, sinonimos in self.dicionario_semantico.items():
            if nome_limpo in sinonimos:
                return chave_canonica
        return nome_limpo

    def _classificar_crime(self, natureza):
        if pd.isna(natureza): return np.nan
        natureza_limpa = self._higienizar_string(natureza)
        if natureza_limpa in self.catalogo_conhecimento:
            return natureza_limpa
        return "OUTROS"

    def _corrigir_ponto_decimal(self, valor, is_lat=True):
        try:
            v = float(str(valor).replace(',', '.'))
            if is_lat and (v < self.limites_territoriais['lat'][0] or v > self.limites_territoriais['lat'][1]):
                v = v / (10 ** (len(str(int(abs(v)))) - 2))
            elif not is_lat and (v < self.limites_territoriais['lon'][0] or v > self.limites_territoriais['lon'][1]):
                v = v / (10 ** (len(str(int(abs(v)))) - 2))
            return v
        except: return np.nan

    def _verificar_fonte(self, url, ano):
        if self.auditoria["modo"] == "HARD_RESET": return True, 0
        caminho_metadados = f"datalake/metadata/fonte_{ano}.json"
        try:
            cabecalho = self.sessao_http.head(url, timeout=30, allow_redirects=True)
            tamanho = int(cabecalho.headers.get('Content-Length', 0))
            if os.path.exists(caminho_metadados):
                with open(caminho_metadados, 'r') as arquivo:
                    if json.load(arquivo).get('tamanho') == tamanho: return False, tamanho
            return True, tamanho
        except: return True, 0

    def _classificar_comportamento(self, linha):
        perfis = set()
        contexto = f"{linha.get('NATUREZA_APURADA','')} {linha.get('DESCR_TIPOLOCAL','')} ".upper()
        for perfil, palavras in self.matriz_comportamental.items():
            if any(palavra in contexto for palavra in palavras):
                perfis.add(perfil)
        return list(perfis) if perfis else ['Geral']

    def _qualificar(self, df_bruto):
        for col in self.esquema_canonico.keys():
            if col not in df_bruto.columns and col != 'ANO_BASE': df_bruto[col] = np.nan

        volume_entrada = len(df_bruto)
        
        for col, tipo in self.esquema_canonico.items():
            if col not in df_bruto.columns: continue
            if tipo == 'string': df_bruto[col] = df_bruto[col].apply(self._higienizar_string)
            elif tipo == 'float': 
                eh_lat = (col == 'LATITUDE')
                df_bruto[col] = df_bruto[col].apply(lambda x: self._corrigir_ponto_decimal(x, eh_lat))
            elif tipo == 'datetime': df_bruto[col] = pd.to_datetime(df_bruto[col], errors='coerce')
            elif tipo == 'int': df_bruto[col] = pd.to_numeric(df_bruto[col], errors='coerce').fillna(0).astype(int)

        df_bruto['NATUREZA_APURADA'] = df_bruto['NATUREZA_APURADA'].apply(self._classificar_crime)

        mascara_limites = (
            df_bruto['LATITUDE'].notna() & df_bruto['LONGITUDE'].notna() &
            (df_bruto['LATITUDE'] != 0) & (df_bruto['LONGITUDE'] != 0) &
            df_bruto['LATITUDE'].between(self.limites_territoriais['lat'][0], self.limites_territoriais['lat'][1]) &
            df_bruto['LONGITUDE'].between(self.limites_territoriais['lon'][0], self.limites_territoriais['lon'][1])
        )
        df_trusted = df_bruto[mascara_limites].copy()[list(self.esquema_canonico.keys())]
        self.auditoria['registros_inconsistentes'] += (volume_entrada - len(df_trusted))
        
        mascara_regras = df_trusted['NATUREZA_APURADA'].notna()
        df_refined = df_trusted[mascara_regras].copy()
        
        return df_trusted, df_refined

    def _mapear_territorio(self, df_eventos):
        df_eventos['perfil_alvo'] = df_eventos.apply(self._classificar_comportamento, axis=1)
        df_eventos = df_eventos.explode('perfil_alvo').dropna(subset=['perfil_alvo'])

        textos = df_eventos['NATUREZA_APURADA'].fillna('') + " " + df_eventos['DESCR_TIPOLOCAL'].fillna('')
        vetorizador = TfidfVectorizer(max_features=50)
        matriz_tfidf = vetorizador.fit_transform(textos)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init="auto")
        df_eventos['vetor_comportamental'] = kmeans.fit_predict(matriz_tfidf)
        df_eventos['gravidade'] = df_eventos['NATUREZA_APURADA'].apply(lambda x: self.catalogo_conhecimento.get(x, {}).get('peso', 1.0))

        coordenadas = df_eventos[['LATITUDE', 'LONGITUDE']].values
        dbscan = DBSCAN(eps=0.001, min_samples=3, metric='euclidean', n_jobs=-1)
        df_eventos['id_hotspot'] = dbscan.fit_predict(coordenadas)
        
        return df_eventos[df_eventos['id_hotspot'] != -1].copy()

    def _processar_redes_neurais(self, df_hotspots):
        def classificar_iluminacao(hora):
            try:
                h = int(str(hora).split(':')[0])
                return 'Madrugada' if 0<=h<6 else 'Manhã' if 6<=h<12 else 'Tarde' if 12<=h<18 else 'Noite'
            except: return 'Indefinido'
            
        df_hotspots['turno'] = df_hotspots['HORA_OCORRENCIA_BO'].apply(classificar_iluminacao)
        
        enc_turno = LabelEncoder()
        enc_perfil = LabelEncoder()
        df_hotspots['turno_enc'] = enc_turno.fit_transform(df_hotspots['turno'])
        df_hotspots['perfil_enc'] = enc_perfil.fit_transform(df_hotspots['perfil_alvo'])

        df_hotspots['DATA_OCORRENCIA_BO'] = pd.to_datetime(df_hotspots['DATA_OCORRENCIA_BO'])
        df_hotspots['dia_semana'] = df_hotspots['DATA_OCORRENCIA_BO'].dt.dayofweek
        df_hotspots['mes'] = df_hotspots['DATA_OCORRENCIA_BO'].dt.month

        serie = df_hotspots.groupby('DATA_OCORRENCIA_BO').size().reset_index(name='y').rename(columns={'DATA_OCORRENCIA_BO': 'ds'})
        prophet = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        prophet.fit(serie)

        baseline = df_hotspots.groupby(['id_hotspot', 'perfil_enc', 'vetor_comportamental', 'turno_enc']).agg(
            lat=('LATITUDE', 'mean'),
            lon=('LONGITUDE', 'mean'),
            risco_base=('gravidade', 'mean'),
            volume=('gravidade', 'count')
        ).reset_index()

        df_treino = df_hotspots.groupby(['id_hotspot', 'perfil_enc', 'vetor_comportamental', 'turno_enc', 'DATA_OCORRENCIA_BO', 'dia_semana', 'mes']).agg({'gravidade': 'sum'}).reset_index()
        df_treino = df_treino.merge(baseline[['id_hotspot', 'perfil_enc', 'vetor_comportamental', 'turno_enc', 'risco_base']], on=['id_hotspot', 'perfil_enc', 'vetor_comportamental', 'turno_enc'], how='left')
        
        fator_sazonal = prophet.predict(df_treino[['DATA_OCORRENCIA_BO']].rename(columns={'DATA_OCORRENCIA_BO': 'ds'}))
        df_treino['sazonalidade'] = fator_sazonal['yhat'].values

        X = df_treino[['perfil_enc', 'vetor_comportamental', 'turno_enc', 'dia_semana', 'mes', 'risco_base', 'sazonalidade']]
        y = df_treino['gravidade']
        
        xgboost = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=5)
        xgboost.fit(X, y)

        projecoes = []
        for dia in range(1, 8):
            data = datetime.now() + timedelta(days=dia)
            df_dia = baseline.copy()
            df_dia['dia_semana'] = data.weekday()
            df_dia['mes'] = data.month
            df_dia['sazonalidade'] = prophet.predict(pd.DataFrame({'ds': [data]}))['yhat'].values[0]
            
            X_futuro = df_dia[['perfil_enc', 'vetor_comportamental', 'turno_enc', 'dia_semana', 'mes', 'risco_base', 'sazonalidade']]
            df_dia['intensidade'] = xgboost.predict(X_futuro)
            projecoes.append(df_dia)
            
        df_semana = pd.concat(projecoes)
        
        df_resultado = df_semana.groupby(['id_hotspot', 'lat', 'lon', 'turno_enc', 'vetor_comportamental']).agg(
            risco_medio=('intensidade', 'mean'),
            volume_historico=('volume', 'first')
        ).reset_index()

        df_resultado['energia'] = df_resultado['risco_medio'] * df_resultado['volume_historico']
        limite = df_resultado['energia'].quantile(0.95)
        
        df_resultado['score'] = (df_resultado['energia'] / limite) * 10
        df_resultado['score'] = df_resultado['score'].clip(0.5, 10.0).round(2)
        df_resultado['penalidade'] = (1 + (df_resultado['score'] * 0.2)).round(2)
        df_resultado['turno_desc'] = enc_turno.inverse_transform(df_resultado['turno_enc'])
        
        df_resultado['codigo_h3'] = df_resultado.apply(lambda row: h3.latlng_to_cell(row['lat'], row['lon'], 10), axis=1)

        self.auditoria['manchas_identificadas'] = len(df_resultado['id_hotspot'].unique())
        
        return df_resultado.groupby(['codigo_h3', 'turno_desc']).agg(
            score=('score', 'max'),
            penalidade=('penalidade', 'max'),
            cluster=('vetor_comportamental', 'first')
        ).reset_index()

    def _executar_expurgo_firestore(self):
        if not self.banco_dados or not self.persistencia_ativa: return
        colecao = self.banco_dados.collection('malha_preditiva_semanal')
        lote = self.banco_dados.batch()
        for doc in colecao.stream():
            lote.delete(doc.reference)
        lote.commit()

    def _exportar_dados_terminais(self, df_malha_h3):
        if not self.banco_dados or not self.persistencia_ativa: return
        
        if self.auditoria["modo"] == "HARD_RESET":
            self._executar_expurgo_firestore()
        
        colecao = self.banco_dados.collection('malha_preditiva_semanal')
        turnos = df_malha_h3['turno_desc'].unique()
        self.auditoria['h3_avaliados'] = len(df_malha_h3)

        for turno in turnos:
            df_turno = df_malha_h3[df_malha_h3['turno_desc'] == turno]
            
            payload = {}
            for _, linha in df_turno.iterrows():
                payload[linha['codigo_h3']] = {
                    "score": float(linha['score']),
                    "penalidade": float(linha['penalidade']),
                    "cluster": int(linha['cluster'])
                }
            
            colecao.document(f"turno_{turno}").set({
                "ultima_atualizacao": firestore.SERVER_TIMESTAMP,
                "h3_dados": payload
            })
            
        if self.auditoria["modo"] == "HARD_RESET":
            with open('datalake/metadata/baseline.lock', 'w') as f: f.write(str(datetime.now()))
            self.auditoria["modo"] = "OPERACIONAL"

    def rodar(self):
        df_master = pd.DataFrame()
        try:
            for ano in self.periodo_historico:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                caminho_raw = f'datalake/raw/ssp_{ano}.parquet'
                caminho_trusted = f'datalake/trusted/ssp_trusted_{ano}.parquet'
                
                baixar, tamanho = self._verificar_fonte(url, ano)
                
                if baixar or not os.path.exists(caminho_raw):
                    res = self.sessao_http.get(url, timeout=120)
                    if res.status_code != 200: continue

                    amostra = pd.read_excel(io.BytesIO(res.content), nrows=50, header=None)
                    linha_header = next((i for i, l in amostra.iterrows() if any(t in [self._higienizar_string(str(c)) for c in l.values] for t in ['NUM_BO', 'LATITUDE', 'NATUREZA_APURADA'])), None)
                    if linha_header is None: continue

                    df_temp = pd.read_excel(io.BytesIO(res.content), skiprows=linha_header, dtype=str)
                    df_temp.columns = [self._normalizar(c) for c in df_temp.columns]
                    
                    df_temp.to_parquet(caminho_raw, index=False)
                    with open(f"datalake/metadata/fonte_{ano}.json", 'w') as f: json.dump({'tamanho': tamanho}, f)
                    self.auditoria['dados_origem_atualizados'] = True
                else:
                    df_temp = pd.read_parquet(caminho_raw)

                df_temp['ANO_BASE'] = ano
                self.auditoria['volume_raw'] += len(df_temp)
                df_trusted, df_refined = self._qualificar(df_temp)
                
                if self.persistencia_ativa: df_trusted.to_parquet(caminho_trusted, index=False)
                df_master = pd.concat([df_master, df_refined])
                self.auditoria['volume_trusted'] += len(df_trusted)
                self.auditoria['volume_refined'] += len(df_refined)

            if df_master.empty: return

            df_manchas = self._mapear_territorio(df_master)
            malha_h3 = self._processar_redes_neurais(df_manchas)
            
            if self.persistencia_ativa:
                malha_h3.to_parquet("datalake/refined/malha_h3_semanal.parquet", index=False)
            
            self._exportar_dados_terminais(malha_h3)
            self._comunicar_status(nivel="OPERACIONAL")

        except Exception as e:
            self._comunicar_status(nivel="FALHA", mensagem=str(e))
            raise

if __name__ == "__main__":
    motor = MotorSafeDriver()
    motor.rodar()
