import pandas as pd
import numpy as np
import os, requests, json, hashlib, traceback, io
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import h3
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import holidays
import warnings
warnings.filterwarnings('ignore')

class MotorInteligenciaLakehouse:
    def __init__(self):
        self.identificador = "SAFE-DRIVER-ENTERPRISE-V2"
        
        self.dirs = {
            'bronze': 'datalake/camada_bronze_bruta',
            'prata': 'datalake/camada_prata_confiavel',
            'ouro': 'datalake/camada_ouro_refinada',
            'estrela': 'datalake/camada_ouro_refinada/esquema_estrela'
        }
        
        for d in self.dirs.values(): os.makedirs(d, exist_ok=True)
        
        self.tokens = {
            "sucesso": os.environ.get('DISCORD_SUCESSO'),
            "erro": os.environ.get('DISCORD_ERRO')
        }
        
        self.anos_alvo = [2025, 2026]
        self.url_ssp = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{}.xlsx"
        self.feriados_sp = holidays.country_holidays('BR', subdiv='SP')
        self.telemetria = {"linhas_bronze": 0, "linhas_prata": 0, "linhas_fato": 0}

    def _obter_sessao_resiliente(self):
        sessao = requests.Session()
        tentativas = Retry(total=5, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        adaptador = HTTPAdapter(max_retries=tentativas)
        sessao.mount('http://', adaptador)
        sessao.mount('https://', adaptador)
        sessao.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        })
        return sessao

    def _registrar_log(self, mensagem):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_path = f"{self.dirs['prata']}/registro_sistema.log"
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {mensagem}\n")
        print(mensagem)

    def _notificar_discord(self, sucesso, detalhes=""):
        url = self.tokens["sucesso"] if sucesso else self.tokens["erro"]
        if not url: return
        
        embed = {
            "title": "🛡️ SAFE DRIVER - LAKEHOUSE ATUALIZADO" if sucesso else "🚨 SAFE DRIVER - FALHA ESTRUTURAL",
            "color": 3066993 if sucesso else 15158332,
            "description": detalhes[:2000],
            "fields": [
                {"name": "📦 Eventos Brutos (Bronze)", "value": f"{self.telemetria['linhas_bronze']}", "inline": True},
                {"name": "🥈 Eventos Limpos (Prata)", "value": f"{self.telemetria['linhas_prata']}", "inline": True},
                {"name": "⭐ Malha de Risco (Ouro)", "value": f"{self.telemetria['linhas_fato']}", "inline": True}
            ],
            "footer": {"text": f"Inteligência Geocriminal | Execução: {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }
        try: requests.post(url, json={"embeds": [embed]})
        except: pass

    def _extrair_aba_correta_excel(self, conteudo_bytes):
        arquivo_excel = pd.read_excel(io.BytesIO(conteudo_bytes), sheet_name=None)
        for nome_aba, df_aba in arquivo_excel.items():
            colunas_limpas = [str(c).upper().strip().replace(" ", "_") for c in df_aba.columns]
            if any(chave in colunas_limpas for chave in ['NUM_BO', 'LATITUDE', 'RUBRICA']):
                self._registrar_log(f"✅ Scanner de dados: Ignorando capas. Tabela real na aba: '{nome_aba}'")
                return df_aba
        return list(arquivo_excel.values())[-1]

    def _baixar_e_converter_ssp(self, ano):
        url = self.url_ssp.format(ano)
        self._registrar_log(f"📥 Conectando à SSP-SP ({ano}) via rede resiliente...")
        sessao = self._obter_sessao_resiliente()
        
        try:
            resposta = sessao.get(url, timeout=180, verify=False)
            if resposta.status_code == 200:
                df_temp = self._extrair_aba_correta_excel(resposta.content)
                df_temp.to_parquet(f"{self.dirs['bronze']}/ssp_{ano}.parquet", index=False)
                return True
            return False
        except Exception as e:
            self._registrar_log(f"❌ Falha de rede no download de {ano}: {e}")
            return False

    def _auditar_todas_camadas(self):
        status = {"reconstruir_bronze": [], "reconstruir_prata": False, "reconstruir_ouro": False}
        self._registrar_log("🔍 Auditoria de Contratos de Dados (Schema Enforcement)...")

        for ano in self.anos_alvo:
            arq_bronze = f"{self.dirs['bronze']}/ssp_{ano}.parquet"
            if not os.path.exists(arq_bronze):
                status["reconstruir_bronze"].append(ano)
            else:
                try:
                    df = pd.read_parquet(arq_bronze)
                    cols = [str(c).upper() for c in df.columns]
                    if not any(c in cols for c in ['LATITUDE', 'LAT']) or not any(c in cols for c in ['DATA_OCORRENCIA_BO', 'DATA_OCORRENCIA']):
                        status["reconstruir_bronze"].append(ano)
                except:
                    status["reconstruir_bronze"].append(ano)

        arq_prata = f"{self.dirs['prata']}/assinatura_anterior.parquet"
        if not os.path.exists(arq_prata) or len(status["reconstruir_bronze"]) > 0:
            status["reconstruir_prata"] = True

        # Verifica se o novo esquema estrela expandido existe
        arquivos_estrela = ['dim_tempo.csv', 'dim_localizacao.csv', 'dim_perfil_crime.csv', 'dim_jurisdicao.csv', 'dim_ambiente.csv', 'fato_risco.csv']
        for arq in arquivos_estrela:
            if not os.path.exists(f"{self.dirs['estrela']}/{arq}"):
                status["reconstruir_ouro"] = True
                break
        
        if status["reconstruir_prata"]: 
            status["reconstruir_ouro"] = True

        return status

    def _processar_bronze(self, anos_para_baixar):
        dfs_para_concatenar = [] 
        for ano in self.anos_alvo:
            if ano in anos_para_baixar:
                self._baixar_e_converter_ssp(ano)
            arq_bronze = f"{self.dirs['bronze']}/ssp_{ano}.parquet"
            if os.path.exists(arq_bronze):
                dfs_para_concatenar.append(pd.read_parquet(arq_bronze))
                
        if not dfs_para_concatenar:
            raise ValueError("Falha: Base Bronze indisponível.")
            
        df_mestre = pd.concat(dfs_para_concatenar, ignore_index=True)
        self.telemetria['linhas_bronze'] = len(df_mestre)
        return df_mestre

    def _engenharia_prata(self, df):
        self._registrar_log("⚙️ Engenharia de Features (Espacial, Temporal, Operacional)...")
        df.columns = [str(c).upper().strip().replace(" ", "_") for c in df.columns]
        
        # Mapeamento do Dicionário de Segurança Pública
        renames = {
            'LATITUDE_Y': 'LATITUDE', 'LAT': 'LATITUDE', 'Y': 'LATITUDE',
            'LONGITUDE_X': 'LONGITUDE', 'LON': 'LONGITUDE', 'X': 'LONGITUDE',
            'DATA_OCORRENCIA_BO': 'DATA_OCORRENCIA', 'DATA_FATO': 'DATA_OCORRENCIA',
            'NATUREZA_APURADA': 'RUBRICA', 'NATUREZA': 'RUBRICA'
        }
        df = df.rename(columns=renames)

        # 1. Tratamento Espacial Estrito
        if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns:
            raise KeyError("Colunas de coordenadas (Latitude/Longitude) não encontradas.")
        
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
        df = df[(df['LATITUDE'] < -19.0) & (df['LATITUDE'] > -26.0) & (df['LONGITUDE'] < -44.0) & (df['LONGITUDE'] > -55.0)]
        df['ID_LOCALIZACAO'] = df.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 8), axis=1)

        # 2. Tratamento Temporal e Cronobiologia do Crime
        df['DATA_OCORRENCIA'] = pd.to_datetime(df['DATA_OCORRENCIA'], errors='coerce')
        df = df.dropna(subset=['DATA_OCORRENCIA'])
        df['DATA_REF'] = df['DATA_OCORRENCIA'].dt.date
        df['ANO'] = df['DATA_OCORRENCIA'].dt.year
        df['MES'] = df['DATA_OCORRENCIA'].dt.month
        df['DIA'] = df['DATA_OCORRENCIA'].dt.day
        df['DIA_SEMANA'] = df['DATA_OCORRENCIA'].dt.dayofweek
        
        if 'HORA_OCORRENCIA_BO' in df.columns:
            df['HORA'] = pd.to_numeric(df['HORA_OCORRENCIA_BO'].astype(str).str.extract(r'(^\d{1,2})')[0], errors='coerce').fillna(-1).astype(int)
        else:
            df['HORA'] = -1
            
        df['PERIODO'] = df['DESC_PERIODO'].fillna('IGNORADO') if 'DESC_PERIODO' in df.columns else 'IGNORADO'

        # 3. Tratamento de Atuária (O Quê e Como)
        df['RUBRICA'] = df['RUBRICA'].fillna('NAO_INFORMADO')
        df['CONDUTA'] = df['DESCR_CONDUTA'].fillna('NAO_INFORMADO') if 'DESCR_CONDUTA' in df.columns else 'NAO_INFORMADO'
        
        # 4. Tratamento Ambiental (Terreno)
        df['TIPO_LOCAL'] = df['DESCR_TIPOLOCAL'].fillna('NAO_INFORMADO') if 'DESCR_TIPOLOCAL' in df.columns else 'NAO_INFORMADO'
        
        # 5. Tratamento de Jurisdição Policial (Quem atende)
        df['MUNICIPIO'] = df['NOME_MUNICIPIO'].fillna('NAO_INFORMADO') if 'NOME_MUNICIPIO' in df.columns else 'NAO_INFORMADO'
        df['BTL'] = df['BTL'].fillna('IGNORADO') if 'BTL' in df.columns else 'IGNORADO'
        df['CIA'] = df['CIA'].fillna('IGNORADO') if 'CIA' in df.columns else 'IGNORADO'
        df['DELEGACIA'] = df['NOME_DELEGACIA'].fillna('IGNORADO') if 'NOME_DELEGACIA' in df.columns else 'IGNORADO'

        self.telemetria['linhas_prata'] = len(df)
        df.to_parquet(f"{self.dirs['prata']}/assinatura_anterior.parquet", index=False)
        return df

    def _modelagem_ouro(self, df):
        if df.empty: return
        self._registrar_log("🧠 IA Ensemble e Geração de Star Schema Multidimensional...")
        
        # Hash Keys para Modelagem Dimensional
        def gerar_id(*args):
            return hashlib.md5("".join([str(a) for a in args]).encode()).hexdigest()[:10]

        # Classificação de Perfil (App Delivery / Transportadoras)
        df['PERFIL_ALVO'] = df['CONDUTA'].apply(lambda x: "Motorista/Carga" if any(k in str(x).upper() for k in ["VEICULO", "CARGA", "TRANSPORTE"]) else ("Pedestre/Delivery" if any(k in str(x).upper() for k in ["CELULAR", "MOTO", "TRANSEUNTE"]) else "Geral"))
        
        df['ID_PERFIL'] = df.apply(lambda r: gerar_id(r['PERFIL_ALVO'], r['RUBRICA'], r['CONDUTA']), axis=1)
        df['ID_AMBIENTE'] = df.apply(lambda r: gerar_id(r['TIPO_LOCAL']), axis=1)
        df['ID_JURISDICAO'] = df.apply(lambda r: gerar_id(r['MUNICIPIO'], r['BTL'], r['CIA'], r['DELEGACIA']), axis=1)

        # Regras de Negócio de Risco Atuarial
        df['GRAVIDADE'] = df['RUBRICA'].apply(lambda x: 10 if any(c in str(x).upper() for c in ['LATROCINIO', 'HOMICIDIO']) else (7 if 'ROUBO' in str(x).upper() else 3))

        # Motor Machine Learning (Treino e Predição)
        X = df[['LATITUDE', 'LONGITUDE', 'DIA_SEMANA', 'MES', 'HORA']]
        y = df['GRAVIDADE']
        if len(X) > 100:
            m_xgb = xgb.XGBRegressor(n_estimators=60, max_depth=6, random_state=42).fit(X, y)
            m_rf = RandomForestRegressor(n_estimators=40, max_depth=6, random_state=42).fit(X, y)
            df['RISCO_CALCULADO'] = (m_xgb.predict(X) + m_rf.predict(X)) / 2
        else:
            df['RISCO_CALCULADO'] = df['GRAVIDADE']

        # === CONSTRUÇÃO DO STAR SCHEMA (POWER BI PRONTO) ===
        
        # 1. Dimensão Tempo
        dim_tempo = df[['DATA_REF', 'ANO', 'MES', 'DIA', 'DIA_SEMANA']].drop_duplicates().copy()
        dim_tempo['ID_TEMPO'] = pd.to_datetime(dim_tempo['DATA_REF']).dt.strftime('%Y%m%d').astype(int)
        dim_tempo['E_FERIADO'] = dim_tempo['DATA_REF'].apply(lambda d: 1 if d in self.feriados_sp else 0)
        dim_tempo['E_PAGAMENTO'] = dim_tempo['DIA'].apply(lambda d: 1 if d in [5, 6, 7, 20, 21] else 0)
        dim_tempo.to_csv(f"{self.dirs['estrela']}/dim_tempo.csv", index=False)

        # 2. Dimensão Localização Geográfica
        dim_loc = df[['ID_LOCALIZACAO', 'LATITUDE', 'LONGITUDE', 'MUNICIPIO']].groupby('ID_LOCALIZACAO').first().reset_index()
        dim_loc.to_csv(f"{self.dirs['estrela']}/dim_localizacao.csv", index=False)

        # 3. Dimensão Perfil Criminal (O Quê)
        dim_perfil = df[['ID_PERFIL', 'PERFIL_ALVO', 'RUBRICA', 'CONDUTA']].drop_duplicates()
        dim_perfil.to_csv(f"{self.dirs['estrela']}/dim_perfil_crime.csv", index=False)

        # 4. Dimensão Ambiente (Onde)
        dim_ambiente = df[['ID_AMBIENTE', 'TIPO_LOCAL']].drop_duplicates()
        dim_ambiente.to_csv(f"{self.dirs['estrela']}/dim_ambiente.csv", index=False)

        # 5. Dimensão Jurisdição PM/PC (Quem atende)
        dim_jur = df[['ID_JURISDICAO', 'MUNICIPIO', 'BTL', 'CIA', 'DELEGACIA']].drop_duplicates()
        dim_jur.to_csv(f"{self.dirs['estrela']}/dim_jurisdicao.csv", index=False)

        # 6. Tabela Fato Risco (A Alma do Negócio)
        df['ID_TEMPO'] = pd.to_datetime(df['DATA_REF']).dt.strftime('%Y%m%d').astype(int)
        fato = df.groupby(['ID_TEMPO', 'HORA', 'PERIODO', 'ID_LOCALIZACAO', 'ID_PERFIL', 'ID_AMBIENTE', 'ID_JURISDICAO']).agg({
            'RISCO_CALCULADO': 'mean',
            'GRAVIDADE': 'count'
        }).reset_index()
        fato.rename(columns={'GRAVIDADE': 'QTD_OCORRENCIAS'}, inplace=True)
        fato.to_csv(f"{self.dirs['estrela']}/fato_risco.csv", index=False)
        self.telemetria['linhas_fato'] = len(fato)

    def executar(self):
        try:
            self._registrar_log("🚀 INICIANDO ORQUESTRAÇÃO LAKEHOUSE MULTIDISCIPLINAR...")
            status_auditoria = self._auditar_todas_camadas()
            
            df_bronze = self._processar_bronze(status_auditoria["reconstruir_bronze"])
            
            if status_auditoria["reconstruir_prata"]:
                df_prata = self._engenharia_prata(df_bronze)
            else:
                df_prata = pd.read_parquet(f"{self.dirs['prata']}/assinatura_anterior.parquet")
                self.telemetria['linhas_prata'] = len(df_prata)
                
            if status_auditoria["reconstruir_ouro"]:
                self._modelagem_ouro(df_prata)
            else:
                fato = pd.read_csv(f"{self.dirs['estrela']}/fato_risco.csv")
                self.telemetria['linhas_fato'] = len(fato)

            self._registrar_log("✅ Lakehouse SafeDriver Finalizado. Dados analíticos disponíveis.")
            self._notificar_discord(True, "Star Schema 360º (Espacial, Atuarial, Operacional e IA) gerado com sucesso.")
        except Exception as e:
            erro_msg = f"Falha Crítica do Sistema: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self._registrar_log(f"❌ {erro_msg}")
            self._notificar_discord(False, erro_msg)
            raise e

if __name__ == "__main__":
    MotorInteligenciaLakehouse().executar()
