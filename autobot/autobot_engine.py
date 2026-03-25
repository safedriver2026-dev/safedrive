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
from sklearn.metrics import mean_absolute_error
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
        
        self.telemetria = {
            "linhas_bronze": 0, "linhas_prata": 0, "linhas_fato": 0,
            "conversao_pct": 0.0, "ruido_ia_pct": 0.0, 
            "erro_absoluto_medio": 0.0, "confianca_sistema_pct": 0.0
        }
        
        self.vetorizador_ia = TfidfVectorizer(max_features=500)
        self.classificador_ia = LogisticRegression(max_iter=1000)

    def _log(self, m):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{ts}] {m}")

    def _notificar_operador(self, status, detalhes=""):
        webhook = self.tokens["sucesso"] if status else self.tokens["erro"]
        if not webhook: return
        cor = 3066993 if status else 15158332
        conteudo = {
            "embeds": [{
                "title": "🛡️ SAFEDRIVER - RELATÓRIO DE INTELIGÊNCIA HÍBRIDA",
                "color": cor,
                "description": detalhes[:2000],
                "fields": [
                    {"name": "📊 FUNIL", "value": f"📥 Bruto: {self.telemetria['linhas_bronze']}\n🧹 Ruído IA: {self.telemetria['ruido_ia_pct']}%\n🥈 Prata: {self.telemetria['linhas_prata']}", "inline": True},
                    {"name": "🧠 MÉTRICAS", "value": f"📉 Erro (MAE): {self.telemetria['erro_absoluto_medio']}\n✅ Confiança: {self.telemetria['confianca_sistema_pct']}%", "inline": True}
                ],
                "footer": {"text": f"SISTEMA SEGURO | CICLO HISTÓRICO 2022-{self.ano_atual}"}
            }]
        }
        try: requests.post(webhook, json=conteudo)
        except: pass

    def _baixar_dados_ssp(self, ano):
        url = self.url_ssp.format(ano)
        temp_file = f"{self.diretorios['bronze']}/fluxo_{ano}.xlsx"
        self._log(f"SOLICITANDO TRANSMISSÃO DE DADOS: {ano}...")
        try:
            sessao = requests.Session()
            sessao.mount('https://', HTTPAdapter(max_retries=Retry(total=3)))
            r = sessao.get(url, stream=True, timeout=600, verify=False)
            if r.status_code == 200:
                with open(temp_file, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
                
                planilha = pd.read_excel(temp_file, sheet_name=None)
                df_final = list(planilha.values())[-1]
                for nome, d in planilha.items():
                    cols_check = [str(x).upper() for x in d.columns]
                    if any(c in cols_check for c in ['NUM_BO', 'LATITUDE']):
                        df_final = d; break
                
                df_final.to_parquet(f"{self.diretorios['bronze']}/ssp_{ano}.parquet", index=False)
                os.remove(temp_file); gc.collect()
                return True
            return False
        except: return False

    def _processar_camada_prata(self, df):
        self._log("IA: EXECUTANDO FUSÃO DE EVIDÊNCIAS (RUBRICA + NATUREZA APURADA)...")
        df.columns = [str(c).upper().strip().replace(" ", "_") for c in df.columns]
        
        # MAPEAMENTO RESILIENTE
        mapeamento = {
            'LATITUDE': ['LATITUDE', 'LAT', 'LATITUDE_Y'],
            'LONGITUDE': ['LONGITUDE', 'LON', 'LONGITUDE_X'],
            'DATA_OCORRENCIA': ['DATA_OCORRENCIA', 'DATA_OCORRENCIA_BO', 'DATA_FATO']
        }
        for alvo, sinonimos in mapeamento.items():
            for col in df.columns:
                if col in sinonimos: df = df.rename(columns={col: alvo}); break

        # IDENTIFICAÇÃO DE CAMPOS CRIMINAIS DISTINTOS
        col_rubrica = next((c for c in df.columns if 'RUBRICA' in c), None)
        col_natureza = next((c for c in df.columns if 'NATUREZA' in c), None)
        col_conduta = next((c for c in df.columns if 'CONDUTA' in c), None)

        # FUSÃO SEMÂNTICA PARA IA NLP
        df['DESCRICAO_CONSOLIDADA'] = ""
        if col_rubrica: df['DESCRICAO_CONSOLIDADA'] += df[col_rubrica].fillna('') + " "
        if col_natureza: df['DESCRICAO_CONSOLIDADA'] += df[col_natureza].fillna('') + " "
        if col_conduta: df['DESCRICAO_CONSOLIDADA'] += df[col_conduta].fillna('')

        # TREINAMENTO DE CLASSIFICAÇÃO SEMÂNTICA
        base_ia = pd.DataFrame([
            {'t': 'ROUBO DE CARGA CAMINHAO', 'v': 1}, {'t': 'FURTO DE VEICULO', 'v': 1},
            {'t': 'LATROCINIO MORTE ROUBO', 'v': 1}, {'t': 'VIOLENCIA DOMESTICA AMEACA', 'v': 0},
            {'t': 'BRIGA VIZINHO', 'v': 0}, {'t': 'CALUNIA INJURIA', 'v': 0}
        ])
        self.classificador_ia.fit(self.vetorizador_ia.fit_transform(base_ia['t']), base_ia['v'])
        
        df['IA_RELEVANTE'] = self.classificador_ia.predict(self.vetorizador_ia.transform(df['DESCRICAO_CONSOLIDADA'].fillna('DESCONHECIDO')))
        
        total_antes = len(df)
        df = df[df['IA_RELEVANTE'] == 1].copy()
        self.telemetria['ruido_ia_pct'] = round(((total_antes - len(df))/total_antes)*100, 2) if total_antes > 0 else 0

        # ENGENHARIA ESPACIAL H3
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
        df = df[(df['LATITUDE'] < -19.0) & (df['LATITUDE'] > -26.0) & (df['LONGITUDE'] < -44.0) & (df['LONGITUDE'] > -55.0)]
        df['ID_LOCALIZACAO'] = df.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 8), axis=1)

        # ENGENHARIA TEMPORAL
        df['DATA_OCORRENCIA'] = pd.to_datetime(df['DATA_OCORRENCIA'], errors='coerce')
        df = df.dropna(subset=['DATA_OCORRENCIA'])
        df['DATA_REF'] = df['DATA_OCORRENCIA'].dt.date
        df['DIA_SEMANA'] = df['DATA_OCORRENCIA'].dt.dayofweek
        df['MES'] = df['DATA_OCORRENCIA'].dt.month
        df['HORA'] = pd.to_numeric(df['HORA_OCORRENCIA_BO'].astype(str).str.extract(r'(^\d{1,2})')[0], errors='coerce').fillna(-1).astype(int) if 'HORA_OCORRENCIA_BO' in df.columns else -1
        
        # ATRIBUIÇÃO DE GRAVIDADE POR EVIDÊNCIA
        def calcular_gravidade(texto):
            t = str(texto).upper()
            if any(x in t for x in ['LATROCINIO', 'HOMICIDIO', 'MORTE']): return 10
            if 'ROUBO' in t: return 7
            return 3
        
        df['GRAVIDADE'] = df['DESCRICAO_CONSOLIDADA'].apply(calcular_gravidade)
        
        self.telemetria['linhas_prata'] = len(df)
        df.to_parquet(f"{self.diretorios['prata']}/assinatura_anterior.parquet", index=False)
        return df

    def _processar_camada_ouro(self, df):
        self._log("IA HÍBRIDA: ANALISANDO ONDE (RADAR) E QUANTO (BALANÇA)...")
        X = df[['LATITUDE', 'LONGITUDE', 'DIA_SEMANA', 'MES', 'HORA']]
        y = df['GRAVIDADE']
        
        if len(X) > 100:
            X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # RADAR: LOCALIZAÇÃO E FREQUÊNCIA (LGBM + XGB)
            radar_a = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1).fit(X_treino, y_treino)
            radar_b = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05).fit(X_treino, y_treino)
            
            # BALANÇA: SEVERIDADE CRIMINAL (CATBOOST)
            balanca = CatBoostRegressor(iterations=100, learning_rate=0.05, silent=True).fit(X_treino, y_treino)
            
            predicoes = (radar_a.predict(X_teste) * 0.3) + (radar_b.predict(X_teste) * 0.3) + (balanca.predict(X_teste) * 0.4)
            
            self.telemetria['erro_absoluto_medio'] = round(float(np.mean(np.abs(y_teste - predicoes))), 4)
            self.telemetria['confianca_sistema_pct'] = round(max(0, (1 - (self.telemetria['erro_absoluto_medio']/y.mean()))*100), 2)
            
            df['RISCO_CALCULADO'] = (radar_a.predict(X) * 0.3) + (radar_b.predict(X) * 0.3) + (balanca.predict(X) * 0.4)
        else:
            df['RISCO_CALCULADO'] = df['GRAVIDADE']

        # GERAÇÃO DO ESQUEMA ESTRELA
        df['ID_TEMPO'] = pd.to_datetime(df['DATA_REF']).dt.strftime('%Y%m%d').astype(int)
        df[['DATA_REF', 'MES', 'DIA_SEMANA']].drop_duplicates().to_csv(f"{self.diretorios['estrela']}/dim_tempo.csv", index=False)
        df[['ID_LOCALIZACAO', 'LATITUDE', 'LONGITUDE']].groupby('ID_LOCALIZACAO').first().reset_index().to_csv(f"{self.diretorios['estrela']}/dim_localizacao.csv", index=False)
        
        fato = df.groupby(['ID_TEMPO', 'HORA', 'ID_LOCALIZACAO']).agg({'RISCO_CALCULADO': 'mean', 'GRAVIDADE': 'count'}).reset_index()
        fato.rename(columns={'GRAVIDADE': 'QTD_OCORRENCIAS'}, inplace=True)
        fato.to_csv(f"{self.dirs['estrela']}/fato_risco.csv", index=False)
        self.telemetria['linhas_fato'] = len(fato)
        
        # PERSISTÊNCIA DE EVIDÊNCIAS DE AUDITORIA
        audit = pd.DataFrame([{
            'DATA': datetime.now().isoformat(),
            'MAE': self.telemetria['erro_absoluto_medio'],
            'CONFIANCA_PCT': self.telemetria['confianca_sistema_pct'],
            'RUIDO_ELIMINADO_IA_PCT': self.telemetria['ruido_ia_pct']
        }])
        caminho_audit = f"{self.diretorios['estrela']}/fato_auditoria.csv"
        audit.to_csv(caminho_audit, mode='a', index=False, header=not os.path.exists(caminho_audit))

    def executar(self):
        try:
            self._log(f"EXECUTANDO PROTOCOLO HÍBRIDO (2022-{self.ano_atual})...")
            lista = []
            for a in self.anos_alvo:
                if not os.path.exists(f"{self.diretorios['bronze']}/ssp_{a}.parquet"): self._baixar_dados_ssp(a)
                if os.path.exists(f"{self.diretorios['bronze']}/ssp_{a}.parquet"): lista.append(pd.read_parquet(f"{self.diretorios['bronze']}/ssp_{a}.parquet"))
            
            df_bronze = pd.concat(lista, ignore_index=True)
            self.telemetria['linhas_bronze'] = len(df_bronze)
            del lista; gc.collect()
            
            df_prata = self._processar_camada_prata(df_bronze)
            del df_bronze; gc.collect()
            
            self._processar_camada_ouro(df_prata)
            
            self.telemetria['conversao_pct'] = round((self.telemetria['linhas_prata'] / self.telemetria['linhas_bronze']) * 100, 2)
            self._notificar_operador(True, "SISTEMA SEGURO. PROCESSAMENTO HÍBRIDO CONCLUÍDO.")
        except Exception as e:
            self._notificar_operador(False, f"ERRO NO MOTOR: {str(e)}\n{traceback.format_exc()}")
            raise e

if __name__ == "__main__":
    SistemaInteligenciaSafeDriver().executar()
