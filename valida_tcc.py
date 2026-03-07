import pandas as pd
import numpy as np
import unicodedata
import os
import glob
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# Constantes espelhadas do seu config.py
LIMITES_SP = {'lat': (-25.5, -19.5), 'lon': (-53.5, -44.0)}

class AuditoriaSazonal:
    def __init__(self):
        # Definição dos meses conforme solicitado
        self.mes_controle = 8  # Agosto: Estabilidade total
        self.mes_estresse = 11 # Novembro: Pico de instabilidade (Feriados)

    def _higienizar_texto(self, texto_bruto):
        """Replica identicamente a lógica de normalização do seu MotorSafeDriver"""
        if pd.isna(texto_bruto) or not isinstance(texto_bruto, str): 
            return str(texto_bruto) if not pd.isna(texto_bruto) else ""
        texto_normalizado = unicodedata.normalize('NFKD', texto_bruto)
        return "".join([c for c in texto_normalizado if not unicodedata.combining(c)]).upper().strip()

    def _tratar_massa_dados(self, df):
        """Fluxo defensivo: Higieniza primeiro, mapeia depois, calcula por último"""
        # 1. Higienização total de cabeçalho (Remove espaços e caracteres invisíveis)
        df.columns = [self._higienizar_texto(c) for c in df.columns]

        # 2. Mapeamento de correção flexível (SSP vira e mexe muda os nomes)
        mapeamento = {
            'NUMERO_BO': 'NUM_BO', 'N_BO': 'NUM_BO', 
            'LAT': 'LATITUDE', 'LATITUD': 'LATITUDE',
            'LON': 'LONGITUDE', 'LONGITUD': 'LONGITUDE', 
            'DATA_FATO': 'DATA_OCORRENCIA_BO', 
            'HORA_FATO': 'HORA_OCORRENCIA_BO', 
            'DT_OCORRENCIA': 'DATA_OCORRENCIA_BO'
        }
        df.rename(columns=mapeamento, inplace=True)

        # 3. Conversão Numérica (Força float e trata vírgulas)
        for col in ['LATITUDE', 'LONGITUDE']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors="coerce")
            else:
                df[col] = np.nan # Garante a existência da coluna para não dar KeyError

        # 4. Máscara Geográfica SafeDriver
        mascara_geo = (
            df['LATITUDE'].notna() & df['LONGITUDE'].notna() &
            (df['LATITUDE'] != 0) & (df['LONGITUDE'] != 0) &
            df['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1]) &
            df['LONGITUDE'].between(LIMITES_SP['lon'][0], LIMITES_SP['lon'][1])
        )
        df = df[mascara_geo].copy()

        # 5. Normalização Temporal
        col_data = 'DATA_OCORRENCIA_BO' if 'DATA_OCORRENCIA_BO' in df.columns else 'DATA'
        if col_data in df.columns:
            df['DT'] = pd.to_datetime(df[col_data], errors='coerce')
        
        col_hora = 'HORA_OCORRENCIA_BO' if 'HORA_OCORRENCIA_BO' in df.columns else 'HORA'
        if col_hora in df.columns:
            df['H'] = pd.to_numeric(df[col_hora].astype(str).str.split(':').str[0], errors='coerce').fillna(0)
        else:
            df['H'] = 0

        # Target e Pesos
        if 'PESO_REAL' not in df.columns: df['PESO_REAL'] = 1.0
        if 'PESO_HEURISTICA' not in df.columns: df['PESO_HEURISTICA'] = 1.0
        
        return df.dropna(subset=['DT', 'LATITUDE', 'LONGITUDE'])

    def executar(self):
        print("Iniciando Comparativo Cego: Controle vs Estresse...")
        
        # Localização via glob recursivo
        arquivos = {
            '23': glob.glob('**/ssp_2023.parquet', recursive=True)[0],
            '24': glob.glob('**/ssp_2024.parquet', recursive=True)[0],
            '25': glob.glob('**/ssp_2025.parquet', recursive=True)[0]
        }

        # Carga e tratamento com lógica espelhada do SafeDriver
        df_treino = self._tratar_massa_dados(pd.concat([pd.read_parquet(arquivos['23']), pd.read_parquet(arquivos['24'])]))
        df_teste = self._tratar_massa_dados(pd.read_parquet(arquivos['25']))

        # --- MODELAGEM (SEM INDUÇÃO) ---
        feats_base = ['LATITUDE', 'LONGITUDE', 'H']
        
        # 1. XGBoost Solo
        m_xgb = XGBRegressor(n_estimators=50).fit(df_treino[feats_base], df_treino['PESO_REAL'])
        
        # 2. Camada Prophet
        df_p = df_treino.groupby(df_treino['DT'].dt.date).size().reset_index(name='y').rename(columns={'DT':'ds'})
        m_p = Prophet(yearly_seasonality=True).fit(df_p)
        forecast = m_p.predict(m_p.make_future_dataframe(periods=365))
        f_map = dict(zip(forecast['ds'].dt.date, (forecast['yhat'] / (forecast['yhat'].mean() + 1e-6))))

        # 3. Ensemble (Híbrido)
        df_treino['f_p'] = df_treino['DT'].dt.date.map(f_map).fillna(1.0)
        m_ens = XGBRegressor(n_estimators=50).fit(df_treino[feats_base + ['f_p']], df_treino['PESO_REAL'])

        # --- VALIDAÇÃO POR CENÁRIO ---
        df_ago = df_teste[df_teste['DT'].dt.month == self.mes_controle]
        df_nov = df_teste[df_teste['DT'].dt.month == self.mes_estresse]

        def medir(df_sub):
            if df_sub.empty: return {"H": 0.0, "X": 0.0, "E": 0.0}
            df_sub = df_sub.copy()
            df_sub['f_p'] = df_sub['DT'].dt.date.map(f_map).fillna(1.0)
            y = df_sub['PESO_REAL']
            return {
                "H": mean_absolute_error(y, df_sub['PESO_HEURISTICA']),
                "X": mean_absolute_error(y, m_xgb.predict(df_sub[feats_base])),
                "E": mean_absolute_error(y, m_ens.predict(df_sub[feats_base + ['f_p']]))
            }

        res_ago = medir(df_ago)
        res_nov = medir(df_nov)

        # Relatório Final
        formula_mae = r"$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$"
        
        relatorio = f"""RELATORIO DE AUDITORIA SAZONAL - SAFEDRIVER 2025
--------------------------------------------------
METRICA UTILIZADA:
{formula_mae}

CENARIO A: CONTROLE (AGOSTO - SEM FERIADOS)
- MAE Heuristica: {res_ago['H']:.6f}
- MAE XGBoost Solo: {res_ago['X']:.6f}
- MAE Ensemble Hibrido: {res_ago['E']:.6f}
Vencedor Agosto: {min(res_ago, key=res_ago.get)}

CENARIO B: ESTRESSE (NOVEMBRO - PICO SAZONAL)
- MAE Heuristica: {res_nov['H']:.6f}
- MAE XGBoost Solo: {res_nov['X']:.6f}
- MAE Ensemble Hibrido: {res_nov['E']:.6f}
Vencedor Novembro: {min(res_nov, key=res_nov.get)}

ANALISE COMPARATIVA (STRESSE):
Impacto do Ensemble vs Heuristica: {((res_nov['H'] - res_nov['E'])/res_nov['H'])*100:+.2f}%
Impacto do Ensemble vs XGB Solo: {((res_nov['X'] - res_nov['E'])/res_nov['X'])*100:+.2f}%
--------------------------------------------------
"""
        with open("validacao_motores_2025.txt", "w", encoding="utf-8") as f:
            f.write(relatorio)
        print("Auditoria finalizada com sucesso.")

if __name__ == "__main__":
    AuditoriaSazonal().executar()
