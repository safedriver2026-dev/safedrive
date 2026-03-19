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

for pasta in ['camada_bronze_bruta', 'camada_prata_confiavel', 'camada_ouro_refinada', 'camada_ouro_refinada/esquema_estrela', 'datalake']: 
    os.makedirs(f'datalake/{pasta}', exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s', 
    handlers=[logging.FileHandler("datalake/logs_processamento.log"), logging.StreamHandler()]
)

class MotorSeguranca:
    def __init__(self, persistencia=True):
        self.identificador = "SISTEMA-AUTONOMO-SAFE-DRIVER-V5"
        self.persistencia = persistencia
        self.banco = self._conectar_banco() if persistencia else None
        self.sessao = self._gerar_sessao()
        self.auditoria = {"bruta": 0, "confiavel": 0, "refinada": 0, "ia": {}, "nuvem": {"total": 0, "delta": 0}}

    def _conectar_banco(self):
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

    def _gerar_camada_ouro(self, dados):
        dados['LATITUDE'] = pd.to_numeric(dados['LATITUDE'], errors='coerce')
        dados['LONGITUDE'] = pd.to_numeric(dados['LONGITUDE'], errors='coerce')
        dados = dados.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()

        dados['id_h3'] = dados.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 10), axis=1)
        dados['peso_risco'] = dados.apply(self._definir_peso, axis=1)
        
        perfis = {"Pedestre": ["CELULAR", "ONIBUS", "PEDESTRE"], "Motorista": ["VEICULO", "CARRO", "CARGA"], "Ciclista": ["BICI"], "Motociclista": ["MOTO"]}
        def classificar(r):
            t = self._limpar_texto(" ".join([str(v) for v in r.values if pd.api.types.is_scalar(v)]))
            m = [p for p, palavras in perfis.items() if any(w in t for w in palavras)]
            return m if m else ["Geral"]
        
        dados['perfis_usuario'] = dados.apply(classificar, axis=1)
        dados = dados.explode('perfis_usuario')
        
        def set_turno(h):
            try:
                h = int(str(h).split(':')[0])
                if 0<=h<6: return 'Madrugada'
                if 6<=h<12: return 'Manha'
                if 12<=h<18: return 'Tarde'
                return 'Noite'
            except: return 'Noite'
        dados['turno_dia'] = dados['HORA_OCORRENCIA_BO'].apply(set_turno)
        
        le = LabelEncoder()
        dados['turno_cod'] = le.fit_transform(dados['turno_dia'])
        X, y = dados[['LATITUDE', 'LONGITUDE', 'turno_cod']], dados['peso_risco']
        
        if len(X) >= 10:
            xt, xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05).fit(xt, yt)
            yp = modelo.predict(xv)
            mae, r2 = mean_absolute_error(yv, yp), r2_score(yv, yp)
        else: mae, r2 = 0.0, 1.0
        
        self.auditoria['ia'] = {"mae": round(mae, 4), "r2": round(r2, 4), "acerto": round(max(0.0, 100.0 - ((mae/10)*100)), 2)}
        
        res = dados.groupby(['id_h3', 'perfis_usuario', 'turno_dia']).agg({'peso_risco': ['mean', 'count'], 'LATITUDE': 'mean', 'LONGITUDE': 'mean'}).reset_index()
        res.columns = ['h3', 'perfil', 'turno', 'peso_medio', 'frequencia', 'lat', 'lon']
        res['score_final'] = MinMaxScaler(feature_range=(0.5, 10.0)).fit_transform(res[['peso_medio']]).round(1) if len(res)>1 else 5.0

        res.to_parquet('datalake/camada_ouro_refinada/malha_mapa_auditavel.parquet', index=False)

        dim_local = res[['h3', 'lat', 'lon']].drop_duplicates()
        dim_perfil = pd.DataFrame({'id_perfil': range(len(res['perfil'].unique())), 'nome_perfil': res['perfil'].unique()})
        dim_tempo = pd.DataFrame({'id_tempo': range(len(res['turno'].unique())), 'nome_turno': res['turno'].unique()})
        
        fato = res.merge(dim_perfil, left_on='perfil', right_on='nome_perfil').merge(dim_tempo, left_on='turno', right_on='nome_turno')
        fato = fato[['h3', 'id_perfil', 'id_tempo', 'score_final', 'frequencia']]
        fato['data_processamento'] = datetime.now()
        
        dim_local.to_csv('datalake/camada_ouro_refinada/esquema_estrela/dim_localizacao.csv', index=False)
        dim_perfil.to_csv('datalake/camada_ouro_refinada/esquema_estrela/dim_perfil.csv', index=False)
        dim_tempo.to_csv('datalake/camada_ouro_refinada/esquema_estrela/dim_tempo.csv', index=False)
        fato.to_csv('datalake/camada_ouro_refinada/esquema_estrela/fato_risco.csv', index=False)
        
        return res

    def _sincronizar_delta(self, dados):
        if not self.db or dados.empty: return
        ref = 'datalake/camada_prata_confiavel/assinatura_anterior.parquet'
        delta = dados.copy()
        if os.path.exists(ref):
            try:
                hist = pd.read_parquet(ref)
                dados['hash'] = dados['h3'] + dados['perfil'] + dados['turno'] + dados['score_final'].astype(str)
                hist['hash'] = hist['h3'] + hist['perfil'] + hist['turno'] + hist['score_final'].astype(str)
                delta = dados[~dados['hash'].isin(hist['hash'])].copy()
                dados.drop(columns=['hash'], inplace=True)
            except: pass
            
        lote = self.db.batch()
        contador = 0
        for _, r in delta.iterrows():
            lote.set(self.db.collection('malha_risco').document(f"{r['perfil'].lower()}_{r['h3']}"), {
                "id_h3": r['h3'], "perfil": r['perfil'],
                f"turnos.{r['turno']}": {"nota": r['score_final'], "atualizado": firestore.SERVER_TIMESTAMP}
            }, merge=True)
            contador += 1
            if contador >= 400:
                lote.commit(); lote = self.db.batch(); contador = 0
        if contador > 0: lote.commit()
        self.auditoria['nuvem'] = {"total": len(dados), "delta": len(delta)}

    def processar(self):
        try:
            mestre = pd.DataFrame()
            meta_p, meta = 'datalake/camada_bronze_bruta/metadados.json', {}
            if os.path.exists(meta_p):
                with open(meta_p, 'r') as f: meta = json.load(f)
            
            ano_limite = datetime.now().year
            for ano in range(2022, ano_limite + 1):
                caminho = f'datalake/camada_bronze_bruta/ssp_{ano}.parquet'
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                try:
                    head = self.sessao.head(url, timeout=20)
                    tamanho = int(head.headers.get('Content-Length', 0))
                    if os.path.exists(caminho) and meta.get(str(ano)) == tamanho:
                        dados_ano = pd.read_parquet(caminho)
                    else:
                        r = self.session.get(url, timeout=300)
                        dados_ano = pd.read_excel(io.BytesIO(r.content), dtype=str)
                        dados_ano.columns = [self._limpar_texto(c) for c in dados_ano.columns]
                        dados_ano.to_parquet(caminho, index=False); meta[str(ano)] = tamanho
                except: dados_ano = pd.read_parquet(caminho) if os.path.exists(caminho) else pd.DataFrame()
                if not dados_ano.empty:
                    self.auditoria['bruta'] += len(dados_ano)
                    mestre = pd.concat([mestre, dados_ano])

            if not mestre.empty:
                confiavel = mestre[mestre['LATITUDE'].notna()].copy()
                self.auditoria['confiavel'] = len(confiavel)
                refinada = self._gerar_camada_ouro(confiavel)
                if not refinada.empty:
                    self.auditoria['refinada'] = len(refinada)
                    self._sincronizar_delta(refinada)
                    refinada.to_parquet('datalake/camada_prata_confiavel/assinatura_anterior.parquet', index=False)
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
            corpo = {"embeds": [{"title": "✅ STATUS: PIPELINE SAFE-DRIVER CONCLUÍDO", "color": 3066993, "fields": [
                {"name": "📈 MÉTRICAS PREDITIVAS", "value": f"Acerto: {self.auditoria['ia']['acerto']}% | $R^2$: {self.auditoria['ia']['r2']}", "inline": False},
                {"name": "🏗️ MEDALLION LAKEHOUSE", "value": f"Bronze (2022+): {self.auditoria['bruta']} | Refinada: {self.auditoria['refinada']}", "inline": True},
                {"name": "💰 EFICIÊNCIA NUVEM", "value": f"Sincronizados: {d} | Economia: **{eco:.1f}%**", "inline": True}]}]}
        else:
            corpo = {"embeds": [{"title": "❌ STATUS: FALHA NO PROCESSAMENTO", "color": 15158332, "description": f"Erro: `{err}`"}]}
        requests.post(url, json=corpo)

if __name__ == "__main__":
    MotorSeguranca().processar()
