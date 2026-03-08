import pandas as pd
import numpy as np
import unicodedata
import os
import glob
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error


LIMITES_SP = {'lat': (-25.5, -19.5), 'lon': (-53.5, -44.0)}
CATALOGO_CRIMES = {
    'FURTO DE VEICULO': 1.0, 'ROUBO DE VEICULO': 2.5, 'ROUBO DE CARGA': 2.5,
    'FURTO DE CARGA': 1.0, 'LATROCINIO': 5.0, 'EXTORSAO MEDIANTE SEQUESTRO': 5.0,
    'ROUBO A TRANSEUNTE': 2.5, 'FURTO DE CELULAR': 1.0
}

class AuditoriaSazonal:
    def __init__(self):
        self.mes_controle = 8  
        self.mes_estresse = 11 

    def _higienizar(self, texto):
        if pd.isna(texto) or not isinstance(texto, str): 
            return str(texto) if not pd.isna(texto) else ""
        return "".join([c for c in unicodedata.normalize('NFKD', texto) if not unicodedata.combining(c)]).upper().strip()

    def _tratar_dados(self, df):
      
        df.columns = [self._higienizar(c) for c in df.columns]
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'].astype(str).str.replace(',', '.'), errors="coerce")
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'].astype(str).str.replace(',', '.'), errors="coerce")
        
    
        mask = (df['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1])) & \
               (df['LONGITUDE'].between(LIMITES_SP['lon'][0], LIMITES_SP['lon'][1]))
        df = df[mask].copy()

      
        df['DT'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        df['H'] = pd.to_numeric(df['HORA_OCORRENCIA_BO'].astype(str).str.split(':').str[0], errors='coerce').fillna(0)
        
       
        df['PESO_REAL'] = df['NATUREZA_APURADA'].map(CATALOGO_CRIMES).fillna(1.0)
        
      
        df['PESO_HEURISTICA'] = df['PESO_REAL'].mean() 
        
        return df.dropna(subset=['DT', 'LATITUDE', 'LONGITUDE'])

    def executar_auditoria(self):
        print("Carregando parquets para auditoria cega...")
        f23 = glob.glob('**/ssp_2023.parquet', recursive=True)[0]
        f24 = glob.glob('**/ssp_2024.parquet', recursive=True)[0]
        f25 = glob.glob('**/ssp_2025.parquet', recursive=True)[0]

        df_train = self._tratar_dados(pd.concat([pd.read_parquet(f23), pd.read_parquet(f24)]))
        df_test = self._tratar_dados(pd.read_parquet(f25))

     
        feats = ['LATITUDE', 'LONGITUDE', 'H']
        m_xgb = XGBRegressor(n_estimators=50).fit(df_train[feats], df_train['PESO_REAL'])
        
     
        df_p = df_train.groupby(df_train['DT'].dt.date).size().reset_index(name='y').rename(columns={'DT':'ds'})
        m_p = Prophet(yearly_seasonality=True).fit(df_p)
        fcst = m_p.predict(m_p.make_future_dataframe(periods=365))
        f_map = dict(zip(fcst['ds'].dt.date, (fcst['yhat'] / (fcst['yhat'].mean() + 1e-6))))

        
        df_train['f_p'] = df_train['DT'].dt.date.map(f_map).fillna(1.0)
        m_ens = XGBRegressor(n_estimators=50).fit(df_train[feats + ['f_p']], df_train['PESO_REAL'])

   
        df_ago = df_test[df_test['DT'].dt.month == self.mes_controle].copy()
        df_nov = df_test[df_test['DT'].dt.month == self.mes_estresse].copy()

        def avaliar(df_sub):
            if df_sub.empty: return {"Heurística": 0, "XGB_Solo": 0, "Ensemble": 0}
            df_sub['f_p'] = df_sub['DT'].dt.date.map(f_map).fillna(1.0)
            y = df_sub['PESO_REAL']
            return {
                "Heuristica": mean_absolute_error(y, df_sub['PESO_HEURISTICA']),
                "XGB_Solo": mean_absolute_error(y, m_xgb.predict(df_sub[feats])),
                "Ensemble": mean_absolute_error(y, m_ens.predict(df_sub[feats + ['f_p']]))
            }

        res_ago = avaliar(df_ago)
        res_nov = avaliar(df_nov)

    
        formula_mae = r"$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$"
        
        relatorio = f"""RELATORIO TECNICO DE AUDITORIA
--------------------------------------------------
METRICA: {formula_mae}

CENARIO 01: AGOSTO 
- MAE Heuristica: {res_ago['Heuristica']:.6f}
- MAE XGBoost: {res_ago['XGB_Solo']:.6f}
- MAE Ensemble Hibrido: {res_ago['Ensemble']:.6f}
Resultado: O motor com menor erro foi {min(res_ago, key=res_ago.get)}.

CENARIO 02: NOVEMBRO
- MAE Heuristica: {res_nov['Heuristica']:.6f}
- MAE XGBoost: {res_nov['XGB_Solo']:.6f}
- MAE Ensemble Hibrido: {res_nov['Ensemble']:.6f}
Resultado: O motor com menor erro foi {min(res_nov, key=res_nov.get)}.

DIFERENÇA NO CENÁRIO DE ESTRESSE:
Ensemble vs Heuristica: {((res_nov['Heuristica'] - res_nov['Ensemble'])/res_nov['Heuristica'])*100:+.2f}%
Ensemble vs XGB Solo: {((res_nov['XGB_Solo'] - res_nov['Ensemble'])/res_nov['XGB_Solo'])*100:+.2f}%
--------------------------------------------------
"""
        with open("validacao_motores_2025.txt", "w", encoding="utf-8") as f:
            f.write(relatorio)

if __name__ == "__main__":
    AuditoriaSazonal().executar_auditoria()
