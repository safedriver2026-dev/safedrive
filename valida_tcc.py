import pandas as pd
import numpy as np
import unicodedata
import os
import glob
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

LIMITES_SP = {'lat': [-24.5, -23.0], 'lon': [-47.5, -45.0]} 

class AuditoriaSazonal:
    def _higienizar(self, texto):
        if pd.isna(texto) or not isinstance(texto, str): return str(texto) if not pd.isna(texto) else ""
        return "".join([c for c in unicodedata.normalize('NFKD', texto) if not unicodedata.combining(c)]).upper().strip()

    def _tratar_dados(self, df):
        df.columns = [self._higienizar(c) for c in df.columns]
        mapa = {'LAT': 'LATITUDE', 'LON': 'LONGITUDE', 'DATA_OCORRENCIA_BO': 'DATA', 'DT_OCORRENCIA': 'DATA'}
        df.rename(columns=mapa, inplace=True)

        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'].astype(str).str.replace(',', '.'), errors="coerce")
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'].astype(str).str.replace(',', '.'), errors="coerce")
        
        mask = (df['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1])) & \
               (df['LONGITUDE'].between(LIMITES_SP['lon'][0], LIMITES_SP['lon'][1]))
        df = df[mask].copy()

        df['DT'] = pd.to_datetime(df['DATA'], errors='coerce')
        df['H'] = pd.to_numeric(df['HORA'].astype(str).str.split(':').str[0], errors='coerce').fillna(0)
       
        if 'PESO_REAL' not in df.columns: df['PESO_REAL'] = 1.0
        if 'PESO_HEURISTICA' not in df.columns: df['PESO_HEURISTICA'] = 1.0
        
        return df.dropna(subset=['DT', 'LATITUDE', 'LONGITUDE'])

    def executar(self):
        print("Iniciando Comparativo: Estresse (Nov) vs Controle (Ago)...")
        
      
        f23 = glob.glob('**/ssp_2023.parquet', recursive=True)[0]
        f24 = glob.glob('**/ssp_2024.parquet', recursive=True)[0]
        f25 = glob.glob('**/ssp_2025.parquet', recursive=True)[0]

        df_train = self._tratar_dados(pd.concat([pd.read_parquet(f23), pd.read_parquet(f24)]))
        df_test = self._tratar_dados(pd.read_parquet(f25))

   
        feats_base = ['LATITUDE', 'LONGITUDE', 'H']
        model_xgb = XGBRegressor(n_estimators=50).fit(df_train[feats_base], df_train['PESO_REAL'])
        
     
        df_p = df_train.groupby(df_train['DT'].dt.date).size().reset_index(name='y').rename(columns={'DT':'ds'})
        m_prophet = Prophet(yearly_seasonality=True).fit(df_p)
        forecast = m_prophet.predict(m_prophet.make_future_dataframe(periods=365))
        f_map = dict(zip(forecast['ds'].dt.date, (forecast['yhat'] / (forecast['yhat'].mean() + 1e-6))))

       
        df_train['f_p'] = df_train['DT'].dt.date.map(f_map).fillna(1.0)
        model_ens = XGBRegressor(n_estimators=50).fit(df_train[feats_base + ['f_p']], df_train['PESO_REAL'])

   
        df_controle = df_test[df_test['DT'].dt.month == 8] # Agosto
        df_estresse = df_test[df_test['DT'].dt.month == 11] # Novembro

        def medir(df_sub):
            if df_sub.empty: return {"MAE_H": 0, "MAE_X": 0, "MAE_E": 0}
            df_sub['f_p'] = df_sub['DT'].dt.date.map(f_map).fillna(1.0)
            y = df_sub['PESO_REAL']
            return {
                "MAE_H": mean_absolute_error(y, df_sub['PESO_HEURISTICA']),
                "MAE_X": mean_absolute_error(y, model_xgb.predict(df_sub[feats_base])),
                "MAE_E": mean_absolute_error(y, model_ens.predict(df_sub[feats_base + ['f_p']]))
            }

        res_c = medir(df_controle)
        res_e = medir(df_estresse)

       
        relatorio = f"""RELATORIO DE VALIDACAO DE PROJETO
--------------------------------------------------
CENARIO A: AGOSTO - SEM FERIADOS
- MAE Heuristica: {res_c['MAE_H']:.6f}
- MAE XGBoost Solo: {res_c['MAE_X']:.6f}
- MAE Ensemble Hibrido: {res_c['MAE_E']:.6f}
Vencedor Agosto: {min(res_c, key=res_c.get).replace('MAE_', '')}

CENARIO B: NOVEMBRO - MUITOS FERIADOS
- MAE Heuristica: {res_e['MAE_H']:.6f}
- MAE XGBoost Solo: {res_e['MAE_X']:.6f}
- MAE Ensemble Hibrido: {res_e['MAE_E']:.6f}
Vencedor Novembro: {min(res_e, key=res_e.get).replace('MAE_', '')}

ANALISE DE IMPACTO:
A inclusao do fator temporal (Prophet) alterou o erro em:
- No Mes de Controle: {((res_c['MAE_E'] - res_c['MAE_X'])/res_c['MAE_X'])*100:+.2f}%
- No Mes de Estresse: {((res_e['MAE_E'] - res_e['MAE_X'])/res_e['MAE_X'])*100:+.2f}%
--------------------------------------------------
"""
        with open("validacao_motores_2025.txt", "w", encoding="utf-8") as f:
            f.write(relatorio)
        print("Auditoria concluída com sucesso.")

if __name__ == "__main__":
    AuditoriaSazonal().executar()
