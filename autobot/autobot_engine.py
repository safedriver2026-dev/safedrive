import pandas as pd
import numpy as np
import os, requests, json, hashlib, traceback, io
from datetime import datetime
import h3
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import holidays
import warnings
warnings.filterwarnings('ignore')

class MotorInteligenciaLakehouse:
    def __init__(self):
        self.identificador = "SAFE-DRIVER-LAKEHOUSE"
        
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
            "title": "🛡️ SAFE DRIVER - DWH ATUALIZADO" if sucesso else "🚨 SAFE DRIVER - FALHA NO PIPELINE",
            "color": 3066993 if sucesso else 15158332,
            "description": detalhes[:2000], # Limite do Discord
            "fields": [
                {"name": "📦 Camada Bronze", "value": f"{self.telemetria['linhas_bronze']} registros", "inline": True},
                {"name": "🥈 Camada Prata", "value": f"{self.telemetria['linhas_prata']} registros", "inline": True},
                {"name": "⭐ Fato Risco (Ouro)", "value": f"{self.telemetria['linhas_fato']} agregações", "inline": True}
            ],
            "footer": {"text": f"Auditoria Multicamadas | Execução: {datetime.now().strftime('%Y-%m-%d %H:%M')}"}
        }
        try: requests.post(url, json={"embeds": [embed]})
        except: pass

    def _baixar_e_converter_ssp(self, ano):
        url = self.url_ssp.format(ano)
        self._registrar_log(f"📥 Baixando dados originais da SSP ({ano})...")
        
        # Prevenção B: Disfarce de Navegador para evitar bloqueios do Governo (Erro 403)
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
            'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet, application/excel'
        }
        
        try:
            resposta = requests.get(url, headers=headers, timeout=90, verify=False)
            if resposta.status_code == 200:
                # Prevenção A: Correção do erro de Buffer (BytesIO)
                df_temp = pd.read_excel(io.BytesIO(resposta.content))
                df_temp.to_parquet(f"{self.dirs['bronze']}/ssp_{ano}.parquet", index=False)
                self._registrar_log(f"✅ Download {ano} concluído e convertido para Parquet.")
                return True
            else:
                self._registrar_log(f"⚠️ SSP retornou Status Code {resposta.status_code} para {ano}")
                return False
        except Exception as e:
            self._registrar_log(f"❌ Erro no download de {ano}: {e}")
            return False

    def _auditar_todas_camadas(self):
        status = {"reconstruir_bronze": [], "reconstruir_prata": False, "reconstruir_ouro": False}
        self._registrar_log("🔍 Iniciando Auditoria Estrutural Completa...")

        for ano in self.anos_alvo:
            arq_bronze = f"{self.dirs['bronze']}/ssp_{ano}.parquet"
            if not os.path.exists(arq_bronze):
                status["reconstruir_bronze"].append(ano)
            else:
                try:
                    df = pd.read_parquet(arq_bronze)
                    cols = [str(c).upper() for c in df.columns]
                    if not any(c in cols for c in ['LATITUDE', 'LAT']) or not any(c in cols for c in ['DATA_OCORRENCIA', 'DATA_FATO', 'DATA']):
                        status["reconstruir_bronze"].append(ano)
                except:
                    status["reconstruir_bronze"].append(ano)

        arq_prata = f"{self.dirs['prata']}/assinatura_anterior.parquet"
        if not os.path.exists(arq_prata) or len(status["reconstruir_bronze"]) > 0:
            status["reconstruir_prata"] = True
        else:
            try:
                df_prata = pd.read_parquet(arq_prata)
                colunas_esperadas = ['DIA_SEMANA', 'ID_LOCALIZACAO', 'LATITUDE', 'LONGITUDE']
                if not all(col in df_prata.columns for col in colunas_esperadas):
                    status["reconstruir_prata"] = True
            except:
                status["reconstruir_prata"] = True

        arquivos_estrela = ['dim_tempo.csv', 'dim_localizacao.csv', 'dim_perfil.csv', 'fato_risco.csv']
        for arq in arquivos_estrela:
            if not os.path.exists(f"{self.dirs['estrela']}/{arq}"):
                status["reconstruir_ouro"] = True
                break
        
        if status["reconstruir_prata"]: 
            status["reconstruir_ouro"] = True

        return status

    def _processar_bronze(self, anos_para_baixar):
        dfs_para_concatenar = [] # Prevenção C: Evita memory leaks de DataFrames vazios
        
        for ano in self.anos_alvo:
            arq_bronze = f"{self.dirs['bronze']}/ssp_{ano}.parquet"
            if ano in anos_para_baixar:
                self._baixar_e_converter_ssp(ano)
            
            if os.path.exists(arq_bronze):
                dfs_para_concatenar.append(pd.read_parquet(arq_bronze))
                
        if not dfs_para_concatenar:
            raise ValueError("Falha crítica: Não foi possível obter ou carregar nenhum dado da camada Bronze.")
            
        df_mestre = pd.concat(dfs_para_concatenar, ignore_index=True)
        self.telemetria['linhas_bronze'] = len(df_mestre)
        return df_mestre

    def _engenharia_prata(self, df):
        if df.empty: raise ValueError("Base Bronze está vazia.")
        self._registrar_log("⚙️ Reconstruindo Camada Prata com novas features...")
        
        df.columns = [str(c).upper().strip().replace(" ", "_") for c in df.columns]
        
        rename_map = {}
        for col in df.columns:
            if col in ['LAT', 'LATITUDE_Y', 'Y']: rename_map[col] = 'LATITUDE'
            if col in ['LON', 'LONG', 'LONGITUDE_X', 'X']: rename_map[col] = 'LONGITUDE'
            if col in ['DATA', 'DATA_FATO', 'DT_OCORRENCIA']: rename_map[col] = 'DATA_OCORRENCIA'
            if col in ['RUBRICA', 'NATUREZA']: rename_map[col] = 'DESCRICAO'
        df = df.rename(columns=rename_map)

        # Prevenção D: Trava de Segurança para Schema Dinâmico da SSP
        if 'LATITUDE' not in df.columns or 'LONGITUDE' not in df.columns:
            msg_erro = f"Mudança drástica no arquivo da SSP. Colunas encontradas: {df.columns.tolist()}"
            self._registrar_log(f"❌ {msg_erro}")
            raise KeyError(msg_erro)

        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
        df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
        df = df[(df['LATITUDE'] < -19.0) & (df['LATITUDE'] > -26.0) & (df['LONGITUDE'] < -44.0) & (df['LONGITUDE'] > -55.0)]

        if 'DATA_OCORRENCIA' not in df.columns:
             raise KeyError(f"Coluna de Data não encontrada após padronização. Encontradas: {df.columns.tolist()}")

        df['DATA_OCORRENCIA'] = pd.to_datetime(df['DATA_OCORRENCIA'], errors='coerce')
        df = df.dropna(subset=['DATA_OCORRENCIA'])
        
        df['DATA_REF'] = df['DATA_OCORRENCIA'].dt.date
        df['ANO'] = df['DATA_OCORRENCIA'].dt.year
        df['MES'] = df['DATA_OCORRENCIA'].dt.month
        df['DIA'] = df['DATA_OCORRENCIA'].dt.day
        df['DIA_SEMANA'] = df['DATA_OCORRENCIA'].dt.dayofweek
        
        df['ID_LOCALIZACAO'] = df.apply(lambda r: h3.latlng_to_cell(r['LATITUDE'], r['LONGITUDE'], 8), axis=1)
        
        # Trava de segurança para Descrição (pode vir como nulo do governo)
        if 'DESCRICAO' not in df.columns:
            df['DESCRICAO'] = 'NATUREZA_NAO_INFORMADA'
        else:
            df['DESCRICAO'] = df['DESCRICAO'].fillna('NATUREZA_NAO_INFORMADA')
        
        self.telemetria['linhas_prata'] = len(df)
        df.to_parquet(f"{self.dirs['prata']}/assinatura_anterior.parquet", index=False)
        return df

    def _modelagem_ouro(self, df):
        if df.empty: return
        self._registrar_log("🧠 Aplicando IA e construindo Esquema Estrela (Ouro)...")
        
        df['PERFIL_ALVO'] = df['DESCRICAO'].apply(lambda x: "Motorista/Carga" if any(k in str(x).upper() for k in ["VEICULO", "CARRO", "CARGA", "CAMINHAO"]) else ("Pedestre/Delivery" if any(k in str(x).upper() for k in ["CELULAR", "MOTO", "PEDESTRE"]) else "Geral"))
        df['ID_PERFIL'] = df['PERFIL_ALVO'].apply(lambda x: hashlib.md5(x.encode()).hexdigest()[:8])
        df['GRAVIDADE'] = df['DESCRICAO'].apply(lambda x: 10 if isinstance(x, str) and any(c in x.upper() for c in ['LATROCINIO', 'HOMICIDIO', 'MORTE']) else (7 if 'ROUBO' in str(x).upper() else 3))

        X = df[['LATITUDE', 'LONGITUDE', 'DIA_SEMANA', 'MES']]
        y = df['GRAVIDADE']
        if len(X) > 100:
            m_xgb = xgb.XGBRegressor(n_estimators=50, max_depth=5, random_state=42).fit(X, y)
            m_rf = RandomForestRegressor(n_estimators=30, max_depth=5, random_state=42).fit(X, y)
            df['RISCO_CALCULADO'] = (m_xgb.predict(X) + m_rf.predict(X)) / 2
        else:
            df['RISCO_CALCULADO'] = df['GRAVIDADE']

        dim_tempo = df[['DATA_REF', 'ANO', 'MES', 'DIA', 'DIA_SEMANA']].drop_duplicates().copy()
        dim_tempo['ID_TEMPO'] = pd.to_datetime(dim_tempo['DATA_REF']).dt.strftime('%Y%m%d').astype(int)
        dim_tempo['E_FERIADO'] = dim_tempo['DATA_REF'].apply(lambda d: 1 if d in self.feriados_sp else 0)
        dim_tempo['E_PAGAMENTO'] = dim_tempo['DIA'].apply(lambda d: 1 if d == 5 or d == 20 else 0) 
        dim_tempo.to_csv(f"{self.dirs['estrela']}/dim_tempo.csv", index=False)

        dim_loc = df[['ID_LOCALIZACAO', 'LATITUDE', 'LONGITUDE']].groupby('ID_LOCALIZACAO').mean().reset_index()
        dim_loc.to_csv(f"{self.dirs['estrela']}/dim_localizacao.csv", index=False)

        dim_perfil = df[['ID_PERFIL', 'PERFIL_ALVO']].drop_duplicates()
        dim_perfil.to_csv(f"{self.dirs['estrela']}/dim_perfil.csv", index=False)

        df['ID_TEMPO'] = pd.to_datetime(df['DATA_REF']).dt.strftime('%Y%m%d').astype(int)
        fato = df.groupby(['ID_TEMPO', 'ID_LOCALIZACAO', 'ID_PERFIL']).agg({
            'RISCO_CALCULADO': 'mean',
            'GRAVIDADE': 'count'
        }).reset_index()
        fato.rename(columns={'GRAVIDADE': 'QTD_OCORRENCIAS'}, inplace=True)
        fato.to_csv(f"{self.dirs['estrela']}/fato_risco.csv", index=False)
        self.telemetria['linhas_fato'] = len(fato)

        df_consolidado = pd.merge(fato, dim_loc, on='ID_LOCALIZACAO')
        df_consolidado.to_parquet(f"{self.dirs['ouro']}/mapa_auditavel.parquet", index=False)

    def executar(self):
        try:
            self._registrar_log("🚀 INICIANDO ORQUESTRAÇÃO LAKEHOUSE...")
            status_auditoria = self._auditar_todas_camadas()
            
            df_bronze = self._processar_bronze(status_auditoria["reconstruir_bronze"])
            
            if status_auditoria["reconstruir_prata"]:
                df_prata = self._engenharia_prata(df_bronze)
            else:
                self._registrar_log("✅ Camada Prata está atualizada. Carregando dados...")
                df_prata = pd.read_parquet(f"{self.dirs['prata']}/assinatura_anterior.parquet")
                self.telemetria['linhas_prata'] = len(df_prata)
                
            if status_auditoria["reconstruir_ouro"]:
                self._modelagem_ouro(df_prata)
            else:
                self._registrar_log("✅ Esquema Estrela (Ouro) está atualizado.")
                fato = pd.read_csv(f"{self.dirs['estrela']}/fato_risco.csv")
                self.telemetria['linhas_fato'] = len(fato)

            self._registrar_log("✅ Pipeline finalizado com sucesso.")
            self._notificar_discord(True, "Processamento Medallion e Machine Learning concluídos.")
        except Exception as e:
            erro_msg = f"Falha crítica na orquestração: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            self._registrar_log(f"❌ {erro_msg}")
            self._notificar_discord(False, erro_msg)
            raise e

if __name__ == "__main__":
    MotorInteligenciaLakehouse().executar()
