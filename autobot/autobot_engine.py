import pandas as pd
import numpy as np
import os, requests, json, hashlib, traceback, io, gc
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import h3
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import holidays
import warnings
warnings.filterwarnings('ignore')

class SistemaInteligenciaSafeDriver:
    def __init__(self):
        self.identificador = "SAFEDRIVER-MOTOR-HIBRIDO"
        self.diretorios = {
            'bronze': 'datalake/camada_bronze_bruta',
            'prata': 'datalake/camada_prata_confiavel',
            'ouro': 'datalake/camada_ouro_refinada',
            'estrela': 'datalake/camada_ouro_refinada/esquema_estrela'
        }
        for d in self.diretorios.values(): os.makedirs(d, exist_ok=True)
        
        self.tokens = {"sucesso": os.environ.get('DISCORD_SUCESSO'), "erro": os.environ.get('DISCORD_ERRO')}
        self.ano_atual = datetime.now().year
        self.anos_alvo = list(range(2022, self.ano_atual + 1))
        self.url_ssp = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{}.xlsx"
        self.feriados_sp = holidays.country_holidays('BR', subdiv='SP')
        
        # MÉTRICAS DE EVIDÊNCIA E ASSERTIVIDADE
        self.telemetria = {
            "linhas_bronze": 0, "linhas_prata": 0, "linhas_fato": 0,
            "conversao_pct": 0.0, "ruido_eliminado_pct": 0.0, 
            "erro_absoluto_medio": 0.0, "erro_quadratico_medio": 0.0, "confianca_sistema_pct": 0.0
        }
        
        # IA NÍVEL 1: CLASSIFICADOR SEMÂNTICO (NLP)
        self.vetorizador_ia = TfidfVectorizer(max_features=500)
        self.classificador_ia = LogisticRegression(max_iter=1000)

    def _conectar_ssp(self):
        sessao = requests.Session()
        sessao.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=3)))
        return sessao

    def _log(self, mensagem):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{timestamp}] {mensagem}")

    def _notificar_operador(self, status_sucesso, detalhes=""):
        webhook = self.tokens["sucesso"] if status_sucesso else self.tokens["erro"]
        if not webhook: return
        cor = 3066993 if status_sucesso else 15158332
        titulo = "🛡️ RELATÓRIO DE INTELIGÊNCIA HÍBRIDA" if status_sucesso else "🚨 FALHA NO PROTOCOLO"
        
        conteudo = {
            "embeds": [{
                "title": titulo,
                "color": cor,
                "description": detalhes[:2000],
                "fields": [
                    {"name": "📊 FUNIL DE DADOS", "value": f"📥 Bruto: {self.telemetria['linhas_bronze']}\n🧹 Ruído IA: {self.telemetria['ruido_eliminado_pct']}%\n🥈 Prata: {self.telemetria['linhas_prata']}", "inline": True},
                    {"name": "🧠 MÉTRICAS HÍBRIDAS", "value": f"📉 Erro (MAE): {self.telemetria['erro_absoluto_medio']}\n✅ Confiança: {self.telemetria['confianca_sistema_pct']}%", "inline": True}
                ],
                "footer": {"text": f"SISTEMA SEGURO | CICLO HISTÓRICO 2022-{self.ano_atual}"}
            }]
        }
        try: requests.post(webhook, json=conteudo)
        except: pass

    def _baixar_dados_ssp(self, ano):
        url = self.url_ssp.format(ano)
        temp_file = f"{self.diretorios['bronze']}/fluxo_{ano}.xlsx"
        self._log(f"SOLICITANDO ACESSO AOS DADOS DE {ano}...")
        try:
            r = self._conectar_ssp().get(url, stream=True, timeout=600, verify=False)
            if r.status_code == 200:
                with open(temp_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
                
                planilha = pd.read_excel(temp_file, sheet_name=None)
                df_final = list(planilha.values())[-1]
                for nome, d in planilha.items():
                    if any(c in [str(x).upper() for x in d.columns] for x in ['NUM_BO', 'LATITUDE']):
                        df_final = d; break
                
                df_final.to_parquet(f"{self.diretorios['bronze']}/ssp_{ano}.parquet", index=False)
                del df_final, planilha; os.remove(temp_file); gc.collect()
                return True
            return False
        except: return False

    def _treinar_ia_limpeza(self):
        self._log("IA: INICIANDO TREINAMENTO DE CLASSIFICAÇÃO SEMÂNTICA...")
        base_treino = pd.DataFrame([
            {'t': 'ROUBO DE CARGA CAMINHAO', 'v': 1}, {'t': 'FURTO DE VEICULO', 'v': 1},
            {'t': 'LATROCINIO MORTE', 'v': 1}, {'t': 'ROUBO CELULAR', 'v': 1},
            {'t': 'VIOLENCIA DOMESTICA', 'v': 0}, {'t': 'BRIGA VIZINHO', 'v': 0},
            {'t': 'CALUNIA INJURIA', 'v': 0}, {'t': 'ESTUPRO', 'v': 0}
        ])
        X = self.vetorizador_ia.fit_transform(base_treino['t'])
        self.classificador_ia.fit(X, base_treino['v'])

    def _processar_camada_prata(self, df):
        self._log("IA: EXECUTANDO LIMPEZA E ENGENHARIA ESPACIAL...")
        df.columns = [str(c).upper().strip().replace(" ", "_") for c in df.columns]
        mapeamento = {'LATITUDE_Y': 'LATITUDE', 'LAT': 'LATITUDE', 'LONGITUDE_X': 'LONGITUDE', 'LON': 'LONGITUDE', 'DATA_OCORRENCIA_BO': 'DATA_OCORRENCIA', 'DATA_FATO': 'DATA_OCORRENCIA', 'NATUREZA_APURADA': 'RUBRICA', 'NATUREZA': 'RUBRICA'}
        df = df.rename(columns=mapeamento)

        # FILTRAGEM SEMÂNTICA IA (ELIMINAÇÃO DE RUÍDO)
        self._treinar_ia_limpeza()
        df['TEXTO_IA'] = df['RUBRICA'].fillna('') + ' ' + (df['DESCR_CONDUTA'].fillna('') if 'DESCR_CONDUTA' in df.columns else '')
        df['IA_RELEVANTE'] = self.classificador_ia.predict(self.vetorizador_ia.transform(df['TEXTO_IA']))
        
        total_antes = len(df)
        df = df[df['IA_RELEVANTE'] == 1].copy()
        self.telemetria['ruido_eliminado_pct'] = round(((total_antes - len(df))/total_antes)*100, 2) if total_antes > 0 else 0

        # TRATAMENTO GEOGRÁFICO
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
        df = df[(df['LATITUDE'] < -19.0) & (df['LATITUDE'] > -26.0) & (df['LONGITUDE'] < -44.0) & (df['LONGITUDE'] > -55.0)]
        df['ID_LOCALIZACAO'] = df.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 8), axis=1)

        # TEMPO
        df['DATA_OCORRENCIA'] = pd.to_datetime(df['DATA_OCORRENCIA'], errors='coerce')
        df = df.dropna(subset=['DATA_OCORRENCIA'])
        df['DATA_REF'] = df['DATA_OCORRENCIA'].dt.date
        df['DIA_SEMANA'] = df['DATA_OCORRENCIA'].dt.dayofweek
        df['MES'] = df['DATA_OCORRENCIA'].dt.month
        df['HORA'] = pd.to_numeric(df['HORA_OCORRENCIA_BO'].astype(str).str.extract(r'(^\d{1,2})')[0], errors='coerce').fillna(-1).astype(int) if 'HORA_OCORRENCIA_BO' in df.columns else -1
        
        # PESO DE GRAVIDADE (NEGÓCIOS)
        df['GRAVIDADE'] = df['RUBRICA'].apply(lambda x: 10 if any(c in str(x).upper() for c in ['LATROCINIO', 'HOMICIDIO', 'MORTE']) else (7 if 'ROUBO' in str(x).upper() else 3))
        
        self.telemetria['linhas_prata'] = len(df)
        df.to_parquet(f"{self.diretorios['prata']}/assinatura_anterior.parquet", index=False)
        return df

    def _processar_camada_ouro(self, df):
        self._log("IA HÍBRIDA: EXECUTANDO PROTOCOLO RADAR & BALANÇA...")
        
        X = df[['LATITUDE', 'LONGITUDE', 'DIA_SEMANA', 'MES', 'HORA']]
        y = df['GRAVIDADE']
        
        if len(X) > 100:
            X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # RADAR (LOCALIZAÇÃO E FREQUÊNCIA) - HÍBRIDO 1
            radar = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1).fit(X_treino, y_treino)
            radar_aux = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05).fit(X_treino, y_treino)
            
            # BALANÇA (SEVERIDADE) - HÍBRIDO 2
            balanca = CatBoostRegressor(iterations=100, learning_rate=0.05, silent=True).fit(X_treino, y_treino)
            
            # RESULTADO HÍBRIDO PONDERADO
            predicoes = (radar.predict(X_teste) * 0.3) + (radar_aux.predict(X_teste) * 0.3) + (balanca.predict(X_teste) * 0.4)
            
            self.telemetria['erro_absoluto_medio'] = round(float(np.mean(np.abs(y_teste - predicoes))), 4)
            self.telemetria['erro_quadratico_medio'] = round(float(np.sqrt(np.mean((y_teste - predicoes)**2))), 4)
            self.telemetria['confianca_sistema_pct'] = round(max(0, (1 - (self.telemetria['erro_absoluto_medio']/y.mean()))*100), 2)
            
            df['RISCO_CALCULADO'] = (radar.predict(X) * 0.3) + (radar_aux.predict(X) * 0.3) + (balanca.predict(X) * 0.4)
        else:
            df['RISCO_CALCULADO'] = df['GRAVIDADE']

        # ESQUEMA ESTRELA
        df['ID_TEMPO'] = pd.to_datetime(df['DATA_REF']).dt.strftime('%Y%m%d').astype(int)
        df[['DATA_REF', 'MES', 'DIA_SEMANA']].drop_duplicates().to_csv(f"{self.diretorios['estrela']}/dim_tempo.csv", index=False)
        df[['ID_LOCALIZACAO', 'LATITUDE', 'LONGITUDE']].groupby('ID_LOCALIZACAO').first().to_csv(f"{self.diretorios['estrela']}/dim_localizacao.csv", index=False)
        
        fato = df.groupby(['ID_TEMPO', 'HORA', 'ID_LOCALIZACAO']).agg({'RISCO_CALCULADO': 'mean', 'GRAVIDADE': 'count'}).reset_index()
        fato.rename(columns={'GRAVIDADE': 'QTD_OCORRENCIAS'}, inplace=True)
        fato.to_csv(f"{self.diretorios['estrela']}/fato_risco.csv", index=False)
        self.telemetria['linhas_fato'] = len(fato)
        
        # PERSISTÊNCIA DE EVIDÊNCIAS (AUDITORIA)
        audit = pd.DataFrame([{
            'DATA_EXECUCAO': datetime.now().isoformat(),
            'MAE': self.telemetria['erro_absoluto_medio'],
            'RMSE': self.telemetria['erro_quadratico_medio'],
            'CONFIANCA_PCT': self.telemetria['confianca_sistema_pct'],
            'REMOVIDOS_RUIDO_PCT': self.telemetria['ruido_eliminado_pct']
        }])
        audit.to_csv(f"{self.diretorios['estrela']}/fato_auditoria.csv", mode='a', index=False, header=not os.path.exists(f"{self.diretorios['estrela']}/fato_auditoria.csv"))

    def executar(self):
        try:
            self._log(f"INICIANDO PROTOCOLO DE TRANSMISSÃO DE DADOS (2022-{self.ano_atual})...")
            
            lista_dfs = []
            for ano in self.anos_alvo:
                if not os.path.exists(f"{self.diretorios['bronze']}/ssp_{ano}.parquet"):
                    self._baixar_dados_ssp(ano)
                if os.path.exists(f"{self.diretorios['bronze']}/ssp_{ano}.parquet"):
                    lista_dfs.append(pd.read_parquet(f"{self.diretorios['bronze']}/ssp_{ano}.parquet"))
            
            df_bronze = pd.concat(lista_dfs, ignore_index=True)
            self.telemetria['linhas_bronze'] = len(df_bronze)
            del lista_dfs; gc.collect()
            
            df_prata = self._processar_camada_prata(df_bronze)
            del df_bronze; gc.collect()
            
            self._processar_camada_ouro(df_prata)
            
            self.telemetria['conversao_pct'] = round((self.telemetria['linhas_prata'] / self.telemetria['linhas_bronze']) * 100, 2)
            self._notificar_operador(True, "SISTEMA SEGURO. PROCESSAMENTO HÍBRIDO CONCLUÍDO COM SUCESSO.")
            
        except Exception as e:
            self._notificar_operador(False, f"ERRO CRÍTICO NO SISTEMA: {str(e)}\n{traceback.format_exc()}")
            raise e

if __name__ == "__main__":
    SistemaInteligenciaSafeDriver().executar()
