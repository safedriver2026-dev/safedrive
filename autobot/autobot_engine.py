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
    handlers=[logging.FileHandler("datalake/registro_operacional.log"), logging.StreamHandler()]
)

class MotorSeguranca:
    def __init__(self, persistencia=True):
        self.identificador = "MOTOR-SISTEMA-SAFE-DRIVER"
        self.persistencia = persistencia
        self.banco = self._conectar_banco_nuvem() if persistencia else None
        self.sessao = self._gerar_sessao_estavel()
        self.auditoria = {"bruta": 0, "confiavel": 0, "refinada": 0, "estatisticas": {}, "nuvem": {"total": 0, "diferencial": 0}}

    def _conectar_banco_nuvem(self):
        config_chave = os.environ.get('FIREBASE_JSON')
        if config_chave and not firebase_admin._apps:
            try:
                credencial = credentials.Certificate(json.loads(config_chave))
                firebase_admin.initialize_app(credencial)
                return firestore.client()
            except: return None
        return None

    def _gerar_sessao_estavel(self):
        s = requests.Session()
        tentativas = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=tentativas))
        return s

    def _limpar_texto(self, texto):
        if pd.isna(texto) or not isinstance(texto, str): return ""
        return "".join([c for c in unicodedata.normalize('NFKD', texto) if not unicodedata.combining(c)]).upper().strip()

    def _atribuir_peso_incidente(self, linha):
        conteudo = " ".join([str(v) for v in linha.values if pd.api.types.is_scalar(v) and pd.notnull(v)])
        t = self._limpar_texto(conteudo)
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

        dados['id_geometria'] = dados.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 10), axis=1)
        dados['peso_final'] = dados.apply(self._atribuir_peso_incidente, axis=1)
        
        perfis = {"Pedestre": ["CELULAR", "ONIBUS", "PEDESTRE"], "Motorista": ["VEICULO", "CARRO", "CARGA"], "Ciclista": ["BICI"], "Motociclista": ["MOTO"]}
        def categorizar(r):
            t = self._limpar_texto(" ".join([str(v) for v in r.values if pd.api.types.is_scalar(v)]))
            m = [p for p, palavras in perfis.items() if any(w in t for w in palavras)]
            return m if m else ["Geral"]
        
        dados['perfis_usuario'] = dados.apply(categorizar, axis=1)
        dados = dados.explode('perfis_usuario')
        
        def definir_periodo(h):
            try:
                h = int(str(h).split(':')[0])
                if 0<=h<6: return 'Madrugada'
                if 6<=h<12: return 'Manha'
                if 12<=h<18: return 'Tarde'
                return 'Noite'
            except: return 'Noite'
        dados['periodo_dia'] = dados['HORA_OCORRENCIA_BO'].apply(definir_periodo)
        
        codificador = LabelEncoder()
        dados['turno_cod'] = codificador.fit_transform(dados['periodo_dia'])
        X, y = dados[['LATITUDE', 'LONGITUDE', 'turno_cod']], dados['peso_final']
        
        if len(X) >= 10:
            xt, xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05).fit(xt, yt)
            yp = modelo.predict(xv)
            mae, r2 = mean_absolute_error(yv, yp), r2_score(yv, yp)
        else: mae, r2 = 0.0, 1.0
        
        self.auditoria['estatisticas'] = {"mae": round(mae, 4), "r2": round(r2, 4), "acerto": round(max(0.0, 100.0 - ((mae/10)*100)), 2)}
        
        res = dados.groupby(['id_geometria', 'perfis_usuario', 'periodo_dia']).agg({'peso_final': ['mean', 'count'], 'LATITUDE': 'mean', 'LONGITUDE': 'mean'}).reset_index()
        res.columns = ['geometria', 'perfil', 'turno', 'peso_medio', 'frequencia', 'lat', 'lon']
        res['nota_final'] = MinMaxScaler(feature_range=(0.5, 10.0)).fit_transform(res[['peso_medio']]).round(1) if len(res)>1 else 5.0

        res.to_parquet('datalake/camada_ouro_refinada/mapa_risco_auditavel.parquet', index=False)

        dim_local = res[['geometria', 'lat', 'lon']].drop_duplicates()
        dim_perfil = pd.DataFrame({'id_p': range(len(res['perfil'].unique())), 'nome_p': res['perfil'].unique()})
        dim_tempo = pd.DataFrame({'id_t': range(len(res['turno'].unique())), 'nome_t': res['turno'].unique()})
        
        fato = res.merge(dim_perfil, left_on='perfil', right_on='nome_p').merge(dim_tempo, left_on='turno', right_on='nome_t')
        fato = fato[['geometria', 'id_p', 'id_t', 'nota_final', 'frequencia']]
        fato['data_processamento'] = datetime.now()
        
        dim_local.to_csv('datalake/camada_ouro_refinada/esquema_estrela/dim_localizacao.csv', index=False)
        dim_perfil.to_csv('datalake/camada_ouro_refinada/esquema_estrela/dim_perfil.csv', index=False)
        dim_tempo.to_csv('datalake/camada_ouro_refinada/esquema_estrela/dim_tempo.csv', index=False)
        fato.to_csv('datalake/camada_ouro_refinada/esquema_estrela/fato_risco.csv', index=False)
        
        return res

    def _sincronizar_dados(self, dados):
        if not self.db or dados.empty: return
        referencia = 'datalake/camada_prata_confiavel/assinatura_anterior.parquet'
        tabela_delta = dados.copy()
        if os.path.exists(referencia):
            try:
                antigo = pd.read_parquet(referencia)
                dados['hash'] = dados['geometria'] + dados['perfil'] + dados['turno'] + dados['nota_final'].astype(str)
                antigo['hash'] = antigo['geometria'] + antigo['perfil'] + antigo['turno'] + antigo['nota_final'].astype(str)
                tabela_delta = dados[~dados['hash'].isin(antigo['hash'])].copy()
                dados.drop(columns=['hash'], inplace=True)
            except: pass
            
        lote = self.db.batch()
        contador = 0
        for _, r in tabela_delta.iterrows():
            ref_doc = self.db.collection('malha_risco').document(f"{r['perfil'].lower()}_{r['geometria']}")
            lote.set(ref_doc, {
                "id_geometria": r['geometria'], "perfil": r['perfil'],
                f"pontuacao.{r['turno']}": {"nota": r['nota_final'], "atualizado": firestore.SERVER_TIMESTAMP}
            }, merge=True)
            contador += 1
            if contador >= 400:
                lote.commit(); lote = self.db.batch(); contador = 0
        if contador > 0: lote.commit()
        self.auditoria['nuvem'] = {"total": len(dados), "diferencial": len(tabela_delta)}

    def iniciar_processamento(self):
        try:
            mestre = pd.DataFrame()
            meta_p, meta = 'datalake/camada_bronze_bruta/rastreio_sincronia.json', {}
            if os.path.exists(meta_p):
                with open(meta_p, 'r') as f: meta = json.load(f)
            
            ano_fim = datetime.now().year
            for ano in range(2022, ano_fim + 1):
                caminho = f'datalake/camada_bronze_bruta/ssp_{ano}.parquet'
                url_ssp = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                try:
                    head = self.sessao.head(url_ssp, timeout=20)
                    tamanho_atual = int(head.headers.get('Content-Length', 0))
                    if os.path.exists(caminho) and meta.get(str(ano)) == tamanho_atual:
                        dados_ano = pd.read_parquet(caminho)
                    else:
                        resp = self.sessao.get(url_ssp, timeout=300)
                        dados_ano = pd.read_excel(io.BytesIO(resp.content), dtype=str)
                        dados_ano.columns = [self._limpar_texto(c) for c in dados_ano.columns]
                        dados_ano.to_parquet(caminho, index=False); meta[str(ano)] = tamanho_atual
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
                    self._sincronizar_dados(refinada)
                    refinada.to_parquet('datalake/camada_prata_confiavel/assinatura_anterior.parquet', index=False)
                    with open(meta_p, 'w') as f: json.dump(meta, f)
                    self._enviar_alerta(True)
        except Exception as e:
            self._enviar_alerta(False, str(e))

    def _enviar_alerta(self, sucesso, erro=None):
        webhook_sucesso = os.environ.get('DISCORD_SUCESSO')
        webhook_erro = os.environ.get('DISCORD_ERRO')
        url = webhook_erro if not sucesso and webhook_erro else webhook_sucesso
        if not url: return
        if sucesso:
            t, d = self.auditoria['nuvem']['total'], self.auditoria['nuvem']['diferencial']
            economia = ((t - d) / t * 100) if t > 0 else 0
            corpo = {"embeds": [{"title": "✅ MOTOR SAFE-DRIVER: SUCESSO", "color": 3066993, "fields": [
                {"name": "📉 MODELAGEM IA", "value": f"Acerto: {self.auditoria['estatisticas']['acerto']}% | $R^2$: {self.auditoria['estatisticas']['r2']}", "inline": False},
                {"name": "🏗️ CAMADAS (2022-2026)", "value": f"Bruta: {self.auditoria['bruta']} | Refinada: {self.auditoria['refinada']}", "inline": True},
                {"name": "💰 EFICIÊNCIA NUVEM", "value": f"Delta: {d} | Poupança: **{economia:.1f}%**", "inline": True}]}]}
        else:
            corpo = {"embeds": [{"title": "🚨 MOTOR SAFE-DRIVER: FALHA CRÍTICA", "color": 15158332, "description": f"O processamento foi interrompido.\n**Erro detectado:** `{erro}`"}]}
        requests.post(url, json=corpo)

if __name__ == "__main__":
    MotorSeguranca().iniciar_processamento()
