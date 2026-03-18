import pandas as pd
import numpy as np
import os, io, requests, json, unicodedata, gc, re, logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AutobotEngine")

class AutobotSafeDriver:
    def __init__(self, persistencia: bool = True):
        self.identidade = "Autobot SafeDriver"
        self.ano_atual = datetime.now().year
        self.periodo_historico = range(2022, self.ano_atual + 1)
        self.persistencia_ativa = persistencia
        self.banco_nuvem = self._conectar_nuvem() if persistencia else None
        self.sessao_rede = self._gerar_sessao_resiliente()
        
        self.auditoria = {
            "camadas": {"bruta": 0, "confiavel": 0, "refinada": 0},
            "perfis": {"Pedestre": 0, "Motorista": 0, "Ciclista": 0, "Motociclista": 0, "Geral": 0},
            "metricas": {"mae": 0.0, "rmse": 0.0, "r2": 0.0, "acerto": 0.0},
            "nuvem": {"documentos": 0}
        }
        
        self.pesos_crimes = {
            "LATROCINIO": 10.0, "HOMICIDIO": 10.0, "ESTUPRO": 10.0, "SEQUESTRO E CARCERE PRIVADO": 10.0,
            "EXTORSAO MEDIANTE SEQUESTRO": 10.0, "ROUBO DE VEICULO": 8.5, "ROUBO DE CARGA": 8.5,
            "ROUBO A TRANSEUNTE": 8.0, "ROUBO": 7.5, "FURTO DE VEICULO": 5.0, 
            "FURTO QUALIFICADO": 4.5, "FURTO": 3.0, "OUTROS": 1.0
        }
        
        self.mapa_palavras_perfil = {
            "Pedestre": ["PEDESTRE", "TRANSEUNTE", "CELULAR", "CALCADA", "ONIBUS", "BOLSA"],
            "Motorista": ["VEICULO", "CARRO", "CAMINHAO", "AUTOMOVEL", "CARGA", "MOTORISTA", "SEMAFORO"],
            "Ciclista": ["BICICLETA", "CICLISTA", "PEDALAR", "BICI", "CICLOVIA"],
            "Motociclista": ["MOTO", "MOTOCICLETA", "CAPACETE", "MOTOBOY", "MOTONETA", "ENTREGADOR"]
        }
        
        self.limites_sp = {"lat": (-24.5, -23.0), "lon": (-47.0, -45.5)}
        for p in ['bruto', 'confiavel', 'refinado']: os.makedirs(f'datalake/{p}', exist_ok=True)

    def _conectar_nuvem(self) -> Any:
        config = os.environ.get('FIREBASE_JSON')
        if config and not firebase_admin._apps:
            try:
                cred = credentials.Certificate(json.loads(config))
                firebase_admin.initialize_app(cred)
                return firestore.client()
            except Exception as e:
                logger.error(f"Falha ao autenticar no Firebase: {e}")
        return None

    def _gerar_sessao_resiliente(self) -> requests.Session:
        s = requests.Session()
        r = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=r))
        s.headers.update({'User-Agent': 'Mozilla/5.0'})
        return s

    def _higienizar(self, texto: str) -> str:
        if pd.isna(texto) or not isinstance(texto, str): return str(texto) if not pd.isna(texto) else ""
        return "".join([c for c in unicodedata.normalize('NFKD', texto) if not unicodedata.combining(c)]).upper().strip()

    def _normalizar_coluna(self, coluna: str) -> str:
        limpo = self._higienizar(coluna)
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

    def _corrigir_gps(self, v: Any, lat: bool = True) -> float:
        try:
            val = float(str(v).replace(',', '.'))
            lim = self.limites_sp['lat'] if lat else self.limites_sp['lon']
            if val < lim[0] or val > lim[1]: val = val / (10 ** (len(str(int(abs(val)))) - 2))
            return val
        except: return np.nan

    def _atribuir_perfis(self, linha: pd.Series) -> List[str]:
        texto_global = " ".join([str(val) for val in linha.values if pd.notnull(val)])
        texto_limpo = self._higienizar(texto_global)
        encontrados = [p for p, palavras in self.mapa_palavras_perfil.items() if any(re.search(rf'\b{w}\b', texto_limpo) for w in palavras)]
        return encontrados if encontrados else ["Geral"]

    def _classificar_risco(self, nota: float, quantis: dict) -> str:
        if nota >= quantis[0.90]: return "CRITICO"
        if nota >= quantis[0.75]: return "ALTO"
        if nota >= quantis[0.50]: return "MEDIO"
        return "BAIXO"

    def _processar_ia(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Iniciando MLOps e feature engineering...")
        min_linhas = 20 if self.persistencia_ativa else 2
        min_densidade = 5 if self.persistencia_ativa else 1
        
        if len(df) < min_linhas: return pd.DataFrame()
        
        df['h3'] = df.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 10), axis=1)
        h3_counts = df['h3'].value_counts()
        df = df[df['h3'].isin(h3_counts[h3_counts >= min_densidade].index)].copy()
        del h3_counts; gc.collect()
        
        if len(df) < min_linhas: return pd.DataFrame()
        
        df['perfis'] = df.apply(self._atribuir_perfis, axis=1)
        df = df.explode('perfis')
        
        def h_int(x: Any) -> int:
            try: return int(str(x).split(':')[0].strip())
            except: return 0
            
        df['turno'] = df['HORA_OCORRENCIA_BO'].apply(lambda x: 'Madrugada' if 0<=h_int(x)<6 else 'Manha' if 6<=h_int(x)<12 else 'Tarde' if 12<=h_int(x)<18 else 'Noite').astype('category')
        df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        
        df['ano'] = df['DATA_OCORRENCIA_BO'].dt.year.fillna(self.ano_atual)
        df['dia_semana'] = df['DATA_OCORRENCIA_BO'].dt.dayofweek.fillna(0).astype('int8')
        df['mes'] = df['DATA_OCORRENCIA_BO'].dt.month.fillna(1).astype('int8')
        df['peso_temporal'] = np.exp((df['ano'] - self.ano_atual) * 0.3).astype('float32')
        
        hist = df.groupby('DATA_OCORRENCIA_BO').size().reset_index(name='y').rename(columns={'DATA_OCORRENCIA_BO': 'ds'})
        f_saz = 1.0
        if len(hist) >= 2:
            m_p = Prophet(yearly_seasonality=True, daily_seasonality=False).fit(hist)
            prev = m_p.predict(pd.DataFrame({'ds': [datetime.now() + timedelta(days=1)]}))
            f_saz = max(0.5, prev['yhat'].values[0] / hist['y'].mean())

        enc_t = LabelEncoder()
        df['t_cod'] = enc_t.fit_transform(df['turno']).astype('int8')
        df['peso_base'] = df['NATUREZA_APURADA'].apply(lambda x: self.pesos_crimes.get(self._higienizar(str(x)), 1.0)).astype('float32')
        df['peso_final_ocorrencia'] = (df['peso_base'] * df['peso_temporal']).astype('float32')
        
        X = df[['LATITUDE', 'LONGITUDE', 't_cod', 'dia_semana', 'mes']]
        y = df['peso_final_ocorrencia']
        
        if len(X) > 50:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, objective='reg:squarederror').fit(X_train, y_train)
            pred = modelo.predict(X_test)
            mae = float(mean_absolute_error(y_test, pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            r2 = float(r2_score(y_test, pred))
        else:
            modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6).fit(X, y)
            pred = modelo.predict(X)
            mae = float(mean_absolute_error(y, pred))
            rmse = float(np.sqrt(mean_squared_error(y, pred)))
            r2 = float(r2_score(y, pred))
            
        taxa_acerto = max(0.0, 100.0 - ((mae / 10.0) * 100.0))
        self.auditoria['metricas'] = {"mae": round(mae, 4), "rmse": round(rmse, 4), "r2": round(r2, 4), "acerto": round(taxa_acerto, 2)}
        
        res = df.groupby(['h3', 'perfis', 'turno']).agg({'peso_final_ocorrencia': ['mean', 'count']}).reset_index()
        res.columns = ['h3', 'perfil', 'turno', 'peso_medio', 'freq']
        
        res['pt_bruta'] = (res['peso_medio'] * np.log1p(res['freq']) * f_saz).astype('float32')
        
        if len(res) > 1:
            limite_superior = res['pt_bruta'].quantile(0.99)
            res['pt_bruta_suave'] = res['pt_bruta'].clip(upper=limite_superior)
            scaler = MinMaxScaler(feature_range=(0.5, 10.0))
            res['pt'] = scaler.fit_transform(res[['pt_bruta_suave']]).round(1).astype('float32')
        else:
            res['pt'] = np.float32(5.0)
            
        res['pn'] = (1 + (res['pt'] * 0.15)).round(2).astype('float32')
        
        quantis_pt = res['pt'].quantile([0.50, 0.75, 0.90]).to_dict()
        res['nivel'] = res['pt'].apply(lambda x: self._classificar_risco(x, quantis_pt))
        
        return res[['h3', 'perfil', 'turno', 'pt', 'pn', 'nivel']]

    def _sincronizar(self, malha: pd.DataFrame) -> None:
        if not self.banco_nuvem or malha.empty: return
        logger.info("Sincronizando com Firestore (Batches de alta performance)...")
        col = self.banco_nuvem.collection('malha_seguranca')
        batch = self.banco_nuvem.batch()
        count = 0
        total_docs = 0

        malha_agrupada = malha.groupby(['h3', 'perfil'])
        for (h3_id, perfil), grupo in malha_agrupada:
            doc_id = f"{perfil.lower()}_{h3_id}"
            scores = {}
            for _, r in grupo.iterrows():
                scores[r['turno']] = {"pt": float(r['pt']), "pn": float(r['pn']), "nivel": r['nivel']}
                
            batch.set(col.document(doc_id), {
                "id_h3": h3_id,
                "perfil": perfil,
                "scores": scores,
                "data": firestore.SERVER_TIMESTAMP
            }, merge=True)
            
            count += 1
            total_docs += 1
            if count >= 400:
                batch.commit()
                batch = self.banco_nuvem.batch()
                count = 0
                
        if count > 0: batch.commit()
        self.auditoria['nuvem']['documentos'] = total_docs

    def executar(self) -> None:
        mestre = pd.DataFrame()
        metadata_path = 'datalake/bruto/metadata.json'
        metadata = {}
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f: metadata = json.load(f)
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

            precisa_baixar = not os.path.exists(caminho_bruto) or (tamanho_remoto and metadata.get(str(ano)) != tamanho_remoto)

            if precisa_baixar:
                logger.info(f"Download e parsing: {ano}")
                try:
                    r = self.sessao_rede.get(url, timeout=300)
                    if r.status_code == 200:
                        df_ano = pd.read_excel(io.BytesIO(r.content), dtype=str)
                        df_ano.columns = [self._normalizar_coluna(c) for c in df_ano.columns]
                        df_ano = df_ano.loc[:, ~df_ano.columns.duplicated()].copy()
                        for col in ['LATITUDE', 'LONGITUDE', 'NATUREZA_APURADA', 'LOCAL', 'HORA_OCORRENCIA_BO', 'DATA_OCORRENCIA_BO']:
                            if col not in df_ano.columns: df_ano[col] = ""
                        df_ano['LATITUDE'] = df_ano['LATITUDE'].apply(lambda x: self._corrigir_gps(x, True)).astype('float32')
                        df_ano['LONGITUDE'] = df_ano['LONGITUDE'].apply(lambda x: self._corrigir_gps(x, False)).astype('float32')
                        df_ano.to_parquet(caminho_bruto, index=False)
                        if tamanho_remoto:
                            metadata[str(ano)] = tamanho_remoto
                            with open(metadata_path, 'w') as f: json.dump(metadata, f)
                except:
                    if os.path.exists(caminho_bruto): df_ano = pd.read_parquet(caminho_bruto)
            else:
                df_ano = pd.read_parquet(caminho_bruto)
            
            if not df_ano.empty:
                self.auditoria['camadas']['bruta'] += len(df_ano)
                val = df_ano[df_ano['LATITUDE'].notna()].copy()
                self.auditoria['camadas']['confiavel'] += len(val)
                mestre = pd.concat([mestre, val], ignore_index=True)
                
            del df_ano; gc.collect()

        if not mestre.empty:
            prev = self._processar_ia(mestre)
            self.auditoria['camadas']['refinada'] = len(prev)
            for p in self.auditoria['perfis'].keys():
                self.auditoria['perfis'][p] = int(len(prev[prev['perfil'] == p]))
            prev.to_parquet('datalake/refinado/malha_final.parquet', index=False)
            self._sincronizar(prev)
            self._notificar()

    def _notificar(self) -> None:
        webhook = os.environ.get('DISCORD_SUCESSO')
        if not webhook: return
        perf_str = "\n".join([f"**{k}:** {v} células mapeadas" for k, v in self.auditoria['perfis'].items()])
        payload = {
            "embeds": [{
                "title": f"🚀 {self.identidade} - Operação Concluída",
                "color": 3066993,
                "fields": [
                    {"name": "🎯 Taxa Acerto (Decaimento Temporal)", "value": f"**{self.auditoria['metricas']['acerto']}%**", "inline": False},
                    {"name": "🌊 Lakehouse", "value": f"Bruto: {self.auditoria['camadas']['bruta']}\nConfiavel: {self.auditoria['camadas']['confiavel']}\nMalha: {self.auditoria['camadas']['refinada']}", "inline": True},
                    {"name": "📉 Margem Erro", "value": f"MAE: {self.auditoria['metricas']['mae']}\nRMSE: {self.auditoria['metricas']['rmse']}\nR²: {self.auditoria['metricas']['r2']}", "inline": True},
                    {"name": "👥 Perfis na Malha", "value": perf_str, "inline": False},
                    {"name": "☁️ Nuvem Firestore", "value": f"{self.auditoria['nuvem']['documentos']} IDs Agrupados", "inline": True}
                ]
            }]
        }
        requests.post(webhook, json=payload)

if __name__ == "__main__":
    AutobotSafeDriver().executar()
