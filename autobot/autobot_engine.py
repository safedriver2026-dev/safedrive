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

# PREPARAÇÃO DA ESTRUTURA DO LAGO DE DADOS (DATA LAKE)
for pasta in ['camada_bronze_bruta', 'camada_prata_confiavel', 'camada_ouro_refinada', 'camada_ouro_refinada/esquema_estrela', 'datalake']: 
    os.makedirs(f'datalake/{pasta}', exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s', 
    handlers=[logging.FileHandler("datalake/diario_sistema.log"), logging.StreamHandler()]
)

class MotorSeguranca:
    def __init__(self, persistencia=True):
        self.identificador = "MOTOR-AUTONOMO-SAFE-DRIVER"
        self.persistencia = persistencia
        self.banco = self._conectar_banco() if persistencia else None # PADRONIZADO: self.banco
        self.sessao = self._gerar_sessao()
        self.auditoria = {"bruta": 0, "confiavel": 0, "refinada": 0, "ia": {}, "nuvem": {"total": 0, "diferencial": 0}}

    def _conectar_banco(self):
        chave = os.environ.get('FIREBASE_JSON')
        if chave and not firebase_admin._apps:
            try:
                credencial = credentials.Certificate(json.loads(chave))
                firebase_admin.initialize_app(credencial)
                return firestore.client()
            except: return None
        return None

    def _gerar_sessao(self):
        s = requests.Session()
        tentativas = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=tentativas))
        return s

    def _limpar_texto(self, t):
        if pd.isna(t) or not isinstance(t, str): return ""
        return "".join([c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c)]).upper().strip()

    def _definir_peso(self, linha):
        conteudo = " ".join([str(v) for v in linha.values if pd.api.types.is_scalar(v) and pd.notnull(v)])
        t = self._limpar_texto(conteudo)
        if any(w in t for w in ["LATROCINIO", "SEQUESTRO"]): return 10.0
        if "ROUBO" in t and any(w in t for w in ["VEICULO", "CARRO", "MOTO", "AUTO"]): return 8.5
        if "ROUBO" in t and "CARGA" in t: return 8.0
        if "ROUBO" in t and any(w in t for w in ["CELULAR", "PEDESTRE", "TRANSEUNTE"]): return 7.5
        if "FURTO" in t and any(w in t for w in ["VEICULO", "CARRO", "MOTO"]): return 4.0
        return 3.0 if "FURTO" in t else 1.0

    def _gerar_camada_ouro(self, dados):
        # PADRONIZAÇÃO DE TIPAGEM PARA CÁLCULO
        dados['LATITUDE'] = pd.to_numeric(dados['LATITUDE'], errors='coerce')
        dados['LONGITUDE'] = pd.to_numeric(dados['LONGITUDE'], errors='coerce')
        dados = dados.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()

        # CRIAÇÃO DA MALHA GEOGRÁFICA
        dados['id_geometria'] = dados.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 10), axis=1)
        dados['nota_perigo'] = dados.apply(self._definir_peso, axis=1)
        
        perfis = {
            "Pedestre": ["CELULAR", "ONIBUS", "PEDESTRE"], 
            "Motorista": ["VEICULO", "CARRO", "CARGA"], 
            "Ciclista": ["BICI", "BICICLETA", "BIKE"], 
            "Motociclista": ["MOTO", "MOTOCICLETA"]
        }
        
        def classificar(r):
            t = self._limpar_texto(" ".join([str(v) for v in r.values if pd.api.types.is_scalar(v)]))
            m = [p for p, palavras in perfis.items() if any(w in t for w in palavras)]
            return m if m else ["Geral"]
        
        dados['perfil_usuario'] = dados.apply(classificar, axis=1)
        dados = dados.explode('perfil_usuario')
        
        def definir_periodo(h):
            try:
                h = int(str(h).split(':')[0])
                if 0<=h<6: return 'Madrugada'
                if 6<=h<12: return 'Manha'
                if 12<=h<18: return 'Tarde'
                return 'Noite'
            except: return 'Noite'
        
        # PADRONIZAÇÃO DE COLUNA TEMPORAL: periodo_dia
        dados['periodo_dia'] = dados['HORA_OCORRENCIA_BO'].apply(definir_periodo)
        
        codificador = LabelEncoder()
        dados['periodo_cod'] = codificador.fit_transform(dados['periodo_dia'])
        X, y = dados[['LATITUDE', 'LONGITUDE', 'periodo_cod']], dados['nota_perigo']
        
        if len(X) >= 10:
            xt, xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05).fit(xt, yt)
            yp = modelo.predict(xv)
            mae, r2 = mean_absolute_error(yv, yp), r2_score(yv, yp)
        else: mae, r2 = 0.0, 1.0
        
        self.auditoria['ia'] = {"mae": round(mae, 4), "r2": round(r2, 4), "acerto": round(max(0.0, 100.0 - ((mae/10)*100)), 2)}
        
        # AGREGAÇÃO FINAL (CAMADA REFINADA)
        res = dados.groupby(['id_geometria', 'perfil_usuario', 'periodo_dia']).agg({'nota_perigo': ['mean', 'count'], 'LATITUDE': 'mean', 'LONGITUDE': 'mean'}).reset_index()
        res.columns = ['geometria', 'perfil', 'periodo', 'media_peso', 'frequencia', 'lat', 'lon']
        res['nota_final'] = MinMaxScaler(feature_range=(0.5, 10.0)).fit_transform(res[['media_peso']]).round(1) if len(res)>1 else 5.0

        res.to_parquet('datalake/camada_ouro_refinada/mapa_auditavel.parquet', index=False)

        # GERAÇÃO DAS TABELAS DO ESQUEMA ESTRELA
        dim_local = res[['geometria', 'lat', 'lon']].drop_duplicates()
        dim_perfil = pd.DataFrame({'id_p': range(len(res['perfil'].unique())), 'nome_p': res['perfil'].unique()})
        dim_tempo = pd.DataFrame({'id_t': range(len(res['periodo'].unique())), 'nome_t': res['periodo'].unique()})
        
        fato = res.merge(dim_perfil, left_on='perfil', right_on='nome_p').merge(dim_tempo, left_on='periodo', right_on='nome_t')
        fato = fato[['geometria', 'id_p', 'id_t', 'nota_final', 'frequencia']]
        fato['data_carga'] = datetime.now()
        
        dim_local.to_csv('datalake/camada_ouro_refinada/esquema_estrela/dim_localizacao.csv', index=False)
        dim_perfil.to_csv('datalake/camada_ouro_refinada/esquema_estrela/dim_perfil.csv', index=False)
        dim_tempo.to_csv('datalake/camada_ouro_refinada/esquema_estrela/dim_tempo.csv', index=False)
        fato.to_csv('datalake/camada_ouro_refinada/esquema_estrela/fato_risco.csv', index=False)
        
        return res

    def _sincronizar_nuvem(self, dados):
        if not self.banco or dados.empty: return
        referencia = 'datalake/camada_prata_confiavel/assinatura_anterior.parquet'
        tabela_delta = dados.copy()
        if os.path.exists(referencia):
            try:
                antigo = pd.read_parquet(referencia)
                dados['hash'] = dados['geometria'] + dados['perfil'] + dados['periodo'] + dados['nota_final'].astype(str)
                antigo['hash'] = antigo['geometria'] + antigo['perfil'] + antigo['periodo'] + antigo['nota_final'].astype(str)
                tabela_delta = dados[~dados['hash'].isin(antigo['hash'])].copy()
                dados.drop(columns=['hash'], inplace=True, errors='ignore')
            except: pass
            
        lote = self.banco.batch() # PADRONIZADO: self.banco
        contador = 0
        for _, r in tabela_delta.iterrows():
            # ID ÚNICO PARA DOCUMENTO GRANULAR (EVITA ERRO DE 1MB)
            id_doc = f"{r['perfil'].lower()}_{r['geometria']}"
            ref_doc = self.banco.collection('malha_risco').document(id_doc)
            
            lote.set(ref_doc, {
                "id_geometria": r['geometria'], 
                "perfil": r['perfil'],
                f"pontuacao.{r['periodo']}": {
                    "nota": r['nota_final'], 
                    "atualizado": firestore.SERVER_TIMESTAMP
                }
            }, merge=True)
            
            contador += 1
            if contador >= 400:
                lote.commit(); lote = self.banco.batch(); contador = 0
        if contador > 0: lote.commit()
        self.auditoria['nuvem'] = {"total": len(dados), "diferencial": len(tabela_delta)}

    def iniciar_processamento(self):
        try:
            mestre = pd.DataFrame()
            meta_p, meta = 'datalake/camada_bronze_bruta/rastreio_sincronia.json', {}
            if os.path.exists(meta_p):
                with open(meta_p, 'r') as f: meta = json.load(f)
            
            #dataset completo 2022-2026
            for ano in range(2022, datetime.now().year + 1):
                caminho_bruto = f'datalake/camada_bronze_bruta/ssp_{ano}.parquet'
                url_fonte = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                try:
                    head = self.sessao.head(url_fonte, timeout=20)
                    tamanho_remoto = int(head.headers.get('Content-Length', 0))
                    if os.path.exists(caminho_bruto) and meta.get(str(ano)) == tamanho_remoto:
                        df_ano = pd.read_parquet(caminho_bruto)
                    else:
                        resp = self.sessao.get(url_fonte, timeout=300)
                        df_ano = pd.read_excel(io.BytesIO(resp.content), dtype=str)
                        df_ano.columns = [self._limpar_texto(c) for c in df_ano.columns]
                        df_ano.to_parquet(caminho_bruto, index=False); meta[str(ano)] = tamanho_remoto
                except: df_ano = pd.read_parquet(caminho_bruto) if os.path.exists(caminho_bruto) else pd.DataFrame()
                if not df_ano.empty:
                    self.auditoria['bruta'] += len(df_ano)
                    mestre = pd.concat([mestre, df_ano])

            if not mestre.empty:
                confiavel = mestre[mestre['LATITUDE'].notna()].copy()
                self.auditoria['confiavel'] = len(confiavel)
                refinada = self._gerar_camada_ouro(confiavel)
                if not refinada.empty:
                    self.auditoria['refinada'] = len(refinada)
                    self._sincronizar_nuvem(refinada)
                    refinada.to_parquet('datalake/camada_prata_confiavel/assinatura_anterior.parquet', index=False)
                    with open(meta_p, 'w') as f: json.dump(meta, f)
                    self._notificar(True)
        except Exception as e:
            self._notificar(False, str(e))

    def _notificar(self, ok, erro_msg=None):
        webhook_sucesso = os.environ.get('DISCORD_SUCESSO')
        webhook_erro = os.environ.get('DISCORD_ERRO')
        url = webhook_erro if not ok and webhook_erro else webhook_sucesso
        if not url: return

        if ok:
            t, d = self.auditoria['nuvem']['total'], self.auditoria['nuvem']['diferencial']
            economia = ((t - d) / t * 100) if t > 0 else 0
            corpo = {"embeds": [{"title": "✅ MOTOR SAFE-DRIVER: SUCESSO OPERACIONAL", "color": 3066993, "fields": [
                {"name": "📉 MÉTRICAS PREDITIVAS", "value": f"Acerto: {self.auditoria['ia']['acerto']}% | $R^2$: {self.auditoria['ia']['r2']}", "inline": False},
                {"name": "🏗️ CAMADAS DE DADOS", "value": f"Bronze (2022+): {self.auditoria['bruta']} | Refinada: {self.auditoria['refinada']}", "inline": True},
                {"name": "💰 EFICIÊNCIA NUVEM", "value": f"Novas Células: {d} | Economia: **{economia:.1f}%**", "inline": True}]}]}
        else:
            corpo = {"embeds": [{"title": "🚨 MOTOR SAFE-DRIVER: FALHA DETECTADA", "color": 15158332, "description": f"Erro identificado: `{erro_msg}`"}]}
        requests.post(url, json=corpo)

if __name__ == "__main__":
    MotorSeguranca().iniciar_processamento()
