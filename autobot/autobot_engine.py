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
    handlers=[logging.FileHandler("datalake/diario_processamento.log"), logging.StreamHandler()]
)

class MotorSegurancaPublica:
    def __init__(self, persistencia=True):
        self.identificador = "MOTOR-AUTONOMO-VERSAO-FINAL"
        self.persistencia = persistencia
        self.banco = self._conectar_nuvem() if persistencia else None
        self.sessao = self._criar_sessao_resiliente()
        self.auditoria = {"bruta": 0, "confiavel": 0, "refinada": 0, "estatisticas": {}, "nuvem": {"total": 0, "diferencial": 0}}

    def _conectar_nuvem(self):
        chave = os.environ.get('FIREBASE_JSON')
        if chave and not firebase_admin._apps:
            try:
                credencial = credentials.Certificate(json.loads(chave))
                firebase_admin.initialize_app(credencial)
                return firestore.client()
            except: return None
        return None

    def _criar_sessao_resiliente(self):
        s = requests.Session()
        tentativas = Retry(total=5, backoff_factor=2, status_forcelist=[500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=tentativas))
        return s

    def _higienizar_texto(self, texto):
        if pd.isna(texto) or not isinstance(texto, str): return ""
        return "".join([c for c in unicodedata.normalize('NFKD', texto) if not unicodedata.combining(c)]).upper().strip()

    def _atribuir_peso_crime(self, linha):
        conteudo = " ".join([str(valor) for valor in linha.values if pd.api.types.is_scalar(valor) and pd.notnull(valor)])
        t = self._higienizar_texto(conteudo)
        if any(palavra in t for palavra in ["LATROCINIO", "SEQUESTRO"]): return 10.0
        if "ROUBO" in t and any(p in t for p in ["VEICULO", "CARRO", "MOTO", "AUTO"]): return 8.5
        if "ROUBO" in t and "CARGA" in t: return 8.0
        if "ROUBO" in t and any(p in t for p in ["CELULAR", "PEDESTRE", "TRANSEUNTE"]): return 7.5
        if "FURTO" in t and any(p in t for p in ["VEICULO", "CARRO", "MOTO"]): return 4.0
        return 3.0 if "FURTO" in t else 1.0

    def _gerar_camada_ouro(self, dados):
        dados['LATITUDE'] = pd.to_numeric(dados['LATITUDE'], errors='coerce')
        dados['LONGITUDE'] = pd.to_numeric(dados['LONGITUDE'], errors='coerce')
        dados = dados.dropna(subset=['LATITUDE', 'LONGITUDE']).copy()

        dados['id_geometria'] = dados.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 10), axis=1)
        dados['nota_perigo'] = dados.apply(self._atribuir_peso_crime, axis=1)
        
        categorias = {"Pedestre": ["CELULAR", "ONIBUS", "PEDESTRE"], "Motorista": ["VEICULO", "CARRO", "CARGA"], "Ciclista": ["BICI"], "Motociclista": ["MOTO"]}
        def categorizar(r):
            t = self._higienizar_texto(" ".join([str(v) for v in r.values if pd.api.types.is_scalar(v)]))
            m = [p for p, palavras in categorias.items() if any(w in t for w in palavras)]
            return m if m else ["Geral"]
        
        dados['perfis_alvo'] = dados.apply(categorizar, axis=1)
        dados = dados.explode('perfis_alvo')
        
        def definir_turno(h):
            try:
                h = int(str(h).split(':')[0])
                if 0<=h<6: return 'Madrugada'
                if 6<=h<12: return 'Manha'
                if 12<=h<18: return 'Tarde'
                return 'Noite'
            except: return 'Noite'
        dados['periodo_dia'] = dados['HORA_OCORRENCIA_BO'].apply(definir_turno)
        
        codificador = LabelEncoder()
        dados['turno_cod'] = codificador.fit_transform(dados['periodo_dia'])
        atributos, alvo = dados[['LATITUDE', 'LONGITUDE', 'turno_cod']], dados['nota_perigo']
        
        if len(atributos) >= 10:
            a_treino, a_teste, v_treino, v_teste = train_test_split(atributos, alvo, test_size=0.2, random_state=42)
            modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05).fit(a_treino, v_treino)
            previsoes = modelo.predict(a_teste)
            erro_medio = mean_absolute_error(v_teste, previsoes)
            precisao_r2 = r2_score(v_teste, previsoes)
        else: erro_medio, precisao_r2 = 0.0, 1.0
        
        self.auditoria['estatisticas'] = {"erro": round(erro_medio, 4), "r2": round(precisao_r2, 4), "acerto": round(max(0.0, 100.0 - ((erro_medio/10)*100)), 2)}
        
        consolidado = dados.groupby(['id_geometria', 'perfis_alvo', 'periodo_dia']).agg({'nota_perigo': ['mean', 'count'], 'LATITUDE': 'mean', 'LONGITUDE': 'mean'}).reset_index()
        consolidado.columns = ['geometria', 'perfil', 'turno', 'media_peso', 'ocorrencias', 'lat', 'lon']
        consolidado['nota_final'] = MinMaxScaler(feature_range=(0.5, 10.0)).fit_transform(consolidado[['media_peso']]).round(1) if len(consolidado)>1 else 5.0

        consolidado.to_parquet('datalake/camada_ouro_refinada/dados_mapa_auditaveis.parquet', index=False)

        dim_espacial = consolidado[['geometria', 'lat', 'lon']].drop_duplicates()
        dim_perfil = pd.DataFrame({'id_perfil': range(len(consolidado['perfil'].unique())), 'nome_perfil': consolidado['perfil'].unique()})
        dim_tempo = pd.DataFrame({'id_turno': range(len(consolidado['turno'].unique())), 'nome_turno': consolidado['turno'].unique()})
        
        fato = consolidado.merge(dim_perfil, left_on='perfil', right_on='nome_perfil').merge(dim_tempo, left_on='turno', right_on='nome_turno')
        fato = fato[['geometria', 'id_perfil', 'id_turno', 'nota_final', 'ocorrencias']]
        fato['data_processamento'] = datetime.now()
        
        dim_espacial.to_csv('datalake/camada_ouro_refinada/esquema_estrela/dim_localizacao.csv', index=False)
        dim_perfil.to_csv('datalake/camada_ouro_refinada/esquema_estrela/dim_perfil.csv', index=False)
        dim_tempo.to_csv('datalake/camada_ouro_refinada/esquema_estrela/dim_tempo.csv', index=False)
        fato.to_csv('datalake/camada_ouro_refinada/esquema_estrela/fato_risco.csv', index=False)
        
        return consolidado

    def _sincronizar_diferencial(self, dados):
        if not self.banco or dados.empty: return
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
            ref_doc = self.banco.collection('malha_risco_geografica').document(f"{r['perfil'].lower()}_{r['geometria']}")
            lote.set(ref_doc, {
                "id_geometria": r['geometria'], "perfil": r['perfil'],
                f"pontuacao.{r['turno']}": {"nota": r['nota_final'], "data_atualizacao": firestore.SERVER_TIMESTAMP}
            }, merge=True)
            contador += 1
            if contador >= 400:
                lote.commit(); lote = self.banco.batch(); contador = 0
        if contador > 0: lote.commit()
        self.auditoria['nuvem'] = {"total": len(dados), "diferencial": len(tabela_envio)}

    def processar_ciclo_completo(self):
        try:
            mestre = pd.DataFrame()
            caminho_metadados, metadados = 'datalake/camada_bronze_bruta/rastreio_arquivos.json', {}
            if os.path.exists(caminho_metadados):
                with open(caminho_metadados, 'r') as f: metadados = json.load(f)
            
            ano_corrente = datetime.now().year
            for ano in range(2022, ano_corrente + 1):
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
                        df_ano.columns = [self._higienizar_texto(c) for c in df_ano.columns]
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
                    self._sincronizar_diferencial(refinada)
                    refinada.to_parquet('datalake/camada_prata_confiavel/memoria_anterior.parquet', index=False)
                    with open(caminho_metadados, 'w') as f: json.dump(metadados, f)
                    self._enviar_notificacao(True)
        except Exception as e:
            self._enviar_notificacao(False, str(e))

    def _enviar_notificacao(self, sucesso, erro_msg=None):
        url_webhook = os.environ.get('DISCORD_SUCESSO')
        if not url_webhook: return
        if sucesso:
            t, d = self.auditoria['nuvem']['total'], self.auditoria['nuvem']['diferencial']
            economia = ((t - d) / t * 100) if t > 0 else 0
            conteudo = {"embeds": [{"title": "✅ STATUS: CICLO OPERACIONAL CONCLUÍDO", "color": 3066993, "fields": [
                {"name": "📉 MÉTRICAS ESTATÍSTICAS", "value": f"Acurácia: {self.auditoria['estatisticas']['acerto']}% | $R^2$: {self.auditoria['estatisticas']['r2']}", "inline": False},
                {"name": "🏗️ ARQUITETURA DE DADOS", "value": f"Bruta: {self.auditoria['bruta']} | Refinada: {self.auditoria['refinada']}", "inline": True},
                {"name": "💰 EFICIÊNCIA FINANCEIRA", "value": f"Alterações: {d} | Economia: **{economia:.1f}%**", "inline": True}]}]}
        else:
            conteudo = {"embeds": [{"title": "❌ STATUS: FALHA NO PROCESSAMENTO", "color": 15158332, "description": f"Log de Erro: `{erro_msg}`"}]}
        requests.post(url_webhook, json=conteudo)

if __name__ == "__main__":
    MotorSegurancaPublica().processar_ciclo_completo()
