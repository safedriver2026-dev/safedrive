import pandas as pd
import numpy as np
import unicodedata
import os
import glob
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# Constantes de limites geograficos do seu config.py
LIMITES_SP = {'lat': [-24.5, -23.0], 'lon': [-47.5, -45.0]} 

class AuditoriaSazonal:
    def __init__(self):
        self.mes_controle = 8  # Agosto
        self.mes_estresse = 11 # Novembro

    def _higienizar_texto(self, texto_bruto):
        if pd.isna(texto_bruto) or not isinstance(texto_bruto, str): 
            return str(texto_bruto) if not pd.isna(texto_bruto) else ""
        texto_normalizado = unicodedata.normalize('NFKD', texto_bruto)
        return "".join([c for c in texto_normalizado if not unicodedata.combining(c)]).upper().strip()

    def _tratar_massa_dados(self, df):
        df.columns = [self._higienizar_texto(c) for c in df.columns]

        mapeamento = {
            'NUMERO_BO': 'NUM_BO', 'N_BO': 'NUM_BO', 'LAT': 'LATITUDE', 
            'LON': 'LONGITUDE', 'DATA_FATO': 'DATA_OCORRENCIA_BO', 
            'HORA_FATO': 'HORA_OCORRENCIA_BO'
        }
        df.rename(columns=mapeamento, inplace=True)

        for col in ['LATITUDE', 'LONGITUDE']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors="coerce")

        mascara_geo = (
            df['LATITUDE'].notna() & df['LONGITUDE'].notna() &
            (df['LATITUDE'] != 0) & (df['LONGITUDE'] != 0) &
            df['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1]) &
            df['LONGITUDE'].between(LIMITES_SP['lon'][0], LIMITES_SP['lon'][1])
        )
        df = df[mascara_geo].copy()

        if 'DATA_OCORRENCIA_BO' in df.columns:
            df['DT'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        
        if 'HORA_OCORRENCIA_BO' in df.columns:
            df['H'] = pd.to_numeric(df['HORA_OCORRENCIA_BO'].astype(str).str.split(':').str[0], errors='coerce').fillna(0)
        
        if 'PESO_REAL' not in df.columns: df['PESO_REAL'] = 1.0
        if 'PESO_HEURISTICA' not in df.columns: df['PESO_HEURISTICA'] = 1.0
        
        return df.dropna(subset=['DT', 'LATITUDE', 'LONGITUDE'])

    def executar(self):
        print("Iniciando Comparativo: Controle (Ago) vs Estresse (Nov)...")
        
        arquivos = {
            '23': glob.glob('**/ssp_2023.parquet', recursive=True)[0],
            '24': glob.glob('**/ssp_2024.parquet', recursive=True)[0],
            '25': glob.glob('**/ssp_2025.parquet', recursive=True)[0]
        }

        df_treino = self._tratar_massa_dados(pd.concat([pd.read_parquet(arquivos['23']), pd.read_parquet(arquivos['24'])]))
        df_teste = self._tratar_massa_dados(pd.read_parquet(arquivos['25']))

        feats_base = ['LATITUDE', 'LONGITUDE', 'H']
        m_xgb = XGBRegressor(n_estimators=50).fit(df_treino[feats_base], df_treino['PESO_REAL'])
        
        df_p = df_treino.groupby(df_treino['DT'].dt.date).size().reset_index(name='y').rename(columns={'DT':'ds'})
        m_p = Prophet(yearly_seasonality=True).fit(df_p)
        forecast = m_p.predict(m_p.make_future_dataframe(periods=365))
        f_map = dict(zip(forecast['ds'].dt.date, (forecast['yhat'] / (forecast['yhat'].mean() + 1e-6))))

        df_treino['f_p'] = df_treino['DT'].dt.date.map(f_map).fillna(1.0)
        m_ens = XGBRegressor(n_estimators=50).fit(df_treino[feats_base + ['f_p']], df_treino['PESO_REAL'])

        df_ago = df_teste[df_teste['DT'].dt.month == self.mes_controle]
        df_nov = df_teste[df_teste['DT'].dt.month == self.mes_estresse]

        def calcular_mae(df_sub):
            if df_sub.empty: return {"H": 0.0, "X": 0.0, "E": 0.0}
            df_sub = df_sub.copy() # Evita SettingWithCopyWarning
            df_sub['f_p'] = df_sub['DT'].dt.date.map(f_map).fillna(1.0)
            y = df_sub['PESO_REAL']
            return {
                "H": mean_absolute_error(y, df_sub['PESO_HEURISTICA']),
                "X": mean_absolute_error(y, m_xgb.predict(df_sub[feats_base])),
                "E": mean_absolute_error(y, m_ens.predict(df_sub[feats_base + ['f_p']]))
            }

        res_ago = calcular_mae(df_ago)
        res_nov = calcular_mae(df_nov)

        # Escapando chaves para evitar f-string SyntaxError
        formula_latex = r"$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$"

        relatorio = f"""RELATORIO DE AUDITORIA SAZONAL - SAFEDRIVER
--------------------------------------------------
MES DE CONTROLE: AGOSTO (ESTABILIDADE)
- MAE Heuristica: {res_ago['H']:.6f}
- MAE XGBoost Solo: {res_ago['X']:.6f}
- MAE Ensemble Hibrido: {res_ago['E']:.6f}
Veredito: {min(res_ago, key=res_ago.get)}

MES DE STRESSE: NOVEMBRO (SAZONALIDADE)
- MAE Heuristica: {res_nov['H']:.6f}
- MAE XGBoost Solo: {res_nov['X']:.6f}
- MAE Ensemble Hibrido: {res_nov['E']:.6f}
Veredito: {min(res_nov, key=res_nov.get)}

ANÁLISE DE PERFORMANCE:
{formula_latex}
Diferenca Ensemble vs Heuristica em Stress: {((res_nov['H'] - res_nov['E'])/res_nov['H'])*100:+.2f}%
--------------------------------------------------
"""
        with open("validacao_motores_2025.txt", "w", encoding="utf-8") as f:
            f.write(relatorio)
        print("Auditoria finalizada.")

if __name__ == "__main__":
    AuditoriaSazonal().executar()
