import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import os


BASE_DIR = "safedrive/datalake/raw/"
DATAS_CRITICAS = [
    '2025-01-01', '2025-03-03', '2025-03-04', '2025-04-18', 
    '2025-05-01', '2025-09-07', '2025-11-20', '2025-12-25'
]

def executar_validacao():
    paths = {
        '23': os.path.join(BASE_DIR, 'ssp_2023.parquet'),
        '24': os.path.join(BASE_DIR, 'ssp_2024.parquet'),
        '25': os.path.join(BASE_DIR, 'ssp_2025.parquet')
    }

   
    df_treino = pd.concat([pd.read_parquet(paths['23']), pd.read_parquet(paths['24'])])
    df_teste = pd.read_parquet(paths['25'])

    def transformar(df):
        df['LAT'] = df['LATITUDE'].fillna(df['LATITUDE'].median())
        df['LON'] = df['LONGITUDE'].fillna(df['LONGITUDE'].median())
        df['H'] = pd.to_numeric(df['HORA_OCORRENCIA_BO'].str.split(':').str[0], errors='coerce').fillna(0)
        df['DT'] = pd.to_datetime(df['DATA']).dt.date
        return df

    df_treino = transformar(df_treino)
    df_teste = transformar(df_teste)

   

    # 2. Motor Prophet 
    df_p = df_treino.groupby('DT').size().reset_index(name='y').rename(columns={'DT':'ds'})
    m = Prophet(yearly_seasonality=True).fit(df_p)
    forecast = m.predict(m.make_future_dataframe(periods=365))
    f_map = dict(zip(forecast['ds'].dt.date, (forecast['yhat'] / forecast['yhat'].mean())))

    # 3. Motor XGBoost Solo
    feats_s = ['LAT', 'LON', 'H']
    xgb_s = XGBRegressor(n_estimators=100).fit(df_treino[feats_s], df_treino['PESO_REAL'])
    
    # 4. Motor Ensemble 
    df_treino['f_p'] = df_treino['DT'].map(f_map).fillna(1.0)
    feats_e = ['LAT', 'LON', 'H', 'f_p']
    xgb_e = XGBRegressor(n_estimators=100).fit(df_treino[feats_e], df_treino['PESO_REAL'])

    # Aplicacao em 2025
    df_teste['f_p'] = df_teste['DT'].map(f_map).fillna(1.0)
    df_teste['is_critico'] = df_teste['DT'].astype(str).isin(DATAS_CRITICAS)

    def compilar(subset):
        y = subset['PESO_REAL']
        res = {
            'Heuristica': mean_absolute_error(y, subset['PESO_HEURISTICA']),
            'XGB_Solo': mean_absolute_error(y, xgb_s.predict(subset[feats_s])),
            'Ensemble': mean_absolute_error(y, xgb_e.predict(subset[feats_e]))
        }
        res['Melhor_Motor'] = min(res, key=res.get)
        return res

    res_normal = compilar(df_teste[~df_teste['is_critico']])
    res_critico = compilar(df_teste[df_teste['is_critico']])

    # Output tecnico sem inducao
    relatorio = f"""RESULTADOS DA VALIDACAO TECNICA 
--------------------------------------------------
CENARIO: DIAS COMUNS
- MAE Heuristica: {res_normal['Heuristica']:.4f}
- MAE XGBoost Solo: {res_normal['XGB_Solo']:.4f}
- MAE Ensemble: {res_normal['Ensemble']:.4f}
Motor com menor erro: {res_normal['Melhor_Motor']}

CENARIO: FERIADOS E DATAS CRITICAS
- MAE Heuristica: {res_critico['Heuristica']:.4f}
- MAE XGBoost Solo: {res_critico['XGB_Solo']:.4f}
- MAE Ensemble: {res_critico['Ensemble']:.4f}
Motor com menor erro: {res_critico['Melhor_Motor']}
--------------------------------------------------
"""
    with open("validacao_motores_2025.txt", "w", encoding="utf-8") as f:
        f.write(relatorio)

if __name__ == "__main__":
    executar_validacao()
