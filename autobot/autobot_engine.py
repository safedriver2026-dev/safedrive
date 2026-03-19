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
    handlers=[logging.FileHandler("datalake/registro_sistema.log"), logging.StreamHandler()]
)

class MotorSeguranca:
    def __init__(self, persistencia=True):
        self.identificador = "MOTOR-AUTONOMO-ESTATISTICO"
        self.persistencia = persistencia
        self.banco = self._conectar_nuvem() if persistencia else None
        self.sessao = self._gerar_sessao_estavel()
        self.auditoria = {"bruta": 0, "confiavel": 0, "refinada": 0, "estatisticas": {}, "nuvem": {"total": 0, "diferencial": 0}}

    def _conectar_nuvem(self):
        config = os.environ.get('FIREBASE_JSON')
        if config and not firebase_admin._apps:
            try:
                cred = credentials.Certificate(json.loads(config))
                firebase_admin.initialize_app(cred)
                return firestore.client()
            except: return None
        return None

    def _gerar_sessao_estavel(self):
        s = requests.Session()
        tentativas = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=tentativas))
        return s

    def _limpar_texto(self, t):
        if pd.isna(t) or not isinstance(t, str): return ""
        return "".join([c for c in unicodedata.normalize('NFKD', t) if not unicodedata.combining(c)]).upper().strip()

    def _atribuir_peso(self, linha):
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
        dados['peso_risco'] = dados.apply(self._atribuir_peso, axis=1)
        
        perfis = {"Pedestre": ["CELULAR", "ONIBUS", "PEDESTRE"], "Motorista": ["VEICULO", "CARRO", "CARGA"], "Ciclista": ["BICI"], "Motociclista": ["MOTO"]}
        def classificar(r):
            t = self._limpar_texto(" ".join([str(v) for v in r.values if pd.api.types.is_scalar(v)]))
            m = [p for p, palavras in perfis.items() if any(w in t for w in palavras)]
            return m if m else ["Geral"]
        
        dados['perfil_usuario'] = dados.apply(classificar, axis=1)
        dados = dados.explode('perfil_usuario')
        
        def definir_turno(h):
            try:
                h = int(str(h).split(':')[0])
                if 0<=h<6: return 'Madrugada'
                if 6<=h<12: return 'Manha'
                if 12<=h<18: return 'Tarde'
                return 'Noite'
            except: return 'Noite'
        dados['turno_dia'] = dados['HORA_OCORRENCIA_BO'].apply(definir_turno)
        
        codificador = LabelEncoder()
        dados['turno_cod'] = codificador.fit_transform(dados['turno_dia'])
        X, y = dados[['LATITUDE', 'LONGITUDE', 'turno_cod']], dados['peso_risco']
        
        if len(X) >= 10:
            xt, xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05).fit(xt, yt)
            yp = modelo.predict(xv)
            mae, r2 = mean_absolute_error(yv, yp), r2_score(yv, yp)
        else: mae, r2 = 0.0, 1.0
        
        self.auditoria['estatisticas'] = {"erro": round(mae, 4), "r2": round(r2, 4), "acerto": round(max(0.0, 100.0 - ((mae/10)*100)), 2)}
        
        res = dados.groupby(['id_geometria', 'perfil_usuario', 'turno_dia']).agg({'peso_risco': ['mean', 'count'], 'LATITUDE': 'mean', 'LONGITUDE': 'mean'}).reset_index()
        res.columns = ['geometria', 'perfil', 'turno', 'media_peso', 'frequencia', 'lat', 'lon']
        res['nota_final'] = MinMaxScaler(feature_range=(0.5, 10.0)).fit_transform(res[['media_peso']]).round(1) if len(res)>1 else 5.0

        res.to_parquet('datalake/camada_ouro_refinada/mapa_auditavel.parquet', index=False)

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

    def _sincronizar_nuvem(self, dados):
        if not self.db or dados.empty: return
        referencia = 'datalake/camada_prata_confiavel/memoria_anterior.parquet'
        tabela_envio = dados.copy()
        if os.path.exists(referencia):
            try:
                antigo = pd.read_parquet(referencia)
                dados['assinatura'] = dados['geometria'] + dados['perfil'] + dados['turno'] + dados['nota_final'].astype(str)
                antigo['assinatura'] = antigo['geometria'] + antigo['perfil'] + antigo['turno'] + antigo['nota_final'].astype(str)
                tabela_envio = dados[~dados['assinatura'].isin(antigo['assinatura'])].copy()
                dados.drop(columns=['assinatura'], inplace=True)
            except: pass
            
        lote = self.banco.batch()
        contador = 0
        for _, r in tabela_envio.iterrows():
            ref_doc = self.banco.collection('malha_risco').document(f"{r['perfil'].lower()}_{r['geometria']}")
            lote.set(ref_doc, {
                "id_geometria": r['geometria'], "perfil": r['perfil'],
                f"pontuacao.{r['turno']}": {"nota": r['nota_final'], "atualizado": firestore.SERVER_TIMESTAMP}
            }, merge=True)
            contador += 1
            if contador >= 400:
                lote.commit(); lote = self.banco.batch(); contador = 0
        if contador > 0: lote.commit()
        self.auditoria['nuvem'] = {"total": len(dados), "diferencial": len(tabela_envio)}

    def processar(self):
        try:
            mestre = pd.DataFrame()
            caminho_meta, metadados = 'datalake/camada_bronze_bruta/controle_sincronia.json', {}
            if os.path.exists(caminho_meta):
                with open(caminho_meta, 'r') as f: metadados = json.load(f)
            
            ano_limite = datetime.now().year
            for ano in range(2022, ano_limite + 1):
                arquivo = f'datalake/camada_bronze_bruta/ssp_{ano}.parquet'
                url_fonte = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                try:
                    head = self.sessao.head(url_fonte, timeout=20)
                    tamanho_remoto = int(head.headers.get('Content-Length', 0))
                    if os.path.exists(arquivo) and metadados.get(str(ano)) == tamanho_remoto:
                        df_ano = pd.read_parquet(arquivo)
                    else:
                        resp = self.sessao.get(url_fonte, timeout=300)
                        df_ano = pd.read_excel(io.BytesIO(resp.content), dtype=str)
                        df_ano.columns = [self._limpar_texto(c) for c in df_ano.columns]
                        df_ano.to_parquet(arquivo, index=False); metadados[str(ano)] = tamanho_remoto
                except: df_ano = pd.read_parquet(arquivo) if os.path.exists(arquivo) else pd.DataFrame()
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
                    refinada.to_parquet('datalake/camada_prata_confiavel/memoria_anterior.parquet', index=False)
                    with open(caminho_meta, 'w') as f: json.dump(metadados, f)
                    self._enviar_notificacao(True)
        except Exception as e:
            self._enviar_notificacao(False, str(e))

    def _enviar_notificacao(self, sucesso, erro=None):
        url_webhook = os.environ.get('DISCORD_SUCESSO')
        if not url_webhook: return
        if sucesso:
            t, d = self.auditoria['nuvem']['total'], self.auditoria['nuvem']['diferencial']
            economia = ((t - d) / t * 100) if t > 0 else 0
            conteudo = {"embeds": [{"title": "✅ STATUS: MOTOR OPERACIONAL CONCLUÍDO", "color": 3066993, "fields": [
                {"name": "📈 MÉTRICAS PREDITIVAS", "value": f"Acurácia: {self.auditoria['estatisticas']['acerto']}% | $R^2$: {self.auditoria['estatisticas']['r2']}", "inline": False},
                {"name": "🏗️ ARQUITETURA DE DADOS", "value": f"Bronze (2022-2026): {self.auditoria['bruta']} | Refinada: {self.auditoria['refinada']}", "inline": True},
                {"name": "💰 EFICIÊNCIA NUVEM", "value": f"Escritas: {d} | Economia: **{economia:.1f}%**", "inline": True}]}]}
        else:
            conteudo = {"embeds": [{"title": "❌ STATUS: FALHA NO PROCESSAMENTO", "color": 15158332, "description": f"Erro: `{erro}`"}]}
        requests.post(url_webhook, json=conteudo)

if __name__ == "__main__":
    MotorSeguranca().processar()
