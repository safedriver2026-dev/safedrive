import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os


FERIADOS_2025 = [
    '2025-01-01', '2025-03-03', '2025-03-04', '2025-04-18', 
    '2025-04-21', '2025-05-01', '2025-06-19', '2025-09-07', 
    '2025-10-12', '2025-11-02', '2025-11-15', '2025-11-20', '2025-12-25'
]

def benchmark_final():
    print("⏳ Carregando dados e iniciando motores de inferência...")
    train = pd.concat([pd.read_parquet('ssp_2023.parquet'), pd.read_parquet('ssp_2024.parquet')])
    test = pd.read_parquet('ssp_2025.parquet')

    def prep(df):
        df['LAT'] = df['LATITUDE'].fillna(df['LATITUDE'].median())
        df['LON'] = df['LONGITUDE'].fillna(df['LONGITUDE'].median())
        df['H'] = pd.to_numeric(df['HORA_OCORRENCIA_BO'].str.split(':').str[0], errors='coerce').fillna(0)
        df['DT'] = pd.to_datetime(df['DATA']).dt.date
        return df

    train, test = prep(train), prep(test)

  
    df_p = train.groupby('DT').size().reset_index(name='y').rename(columns={'DT':'ds'})
    m = Prophet(yearly_seasonality=True).fit(df_p)
    forecast = m.predict(m.make_future_dataframe(periods=365))
    forecast['fator_p'] = forecast['yhat'] / forecast['yhat'].mean()
    f_map = dict(zip(forecast['ds'].dt.date, forecast['fator_p']))

    -
    feat_solo = ['LAT', 'LON', 'H']
    xgb_solo = XGBRegressor(n_estimators=100).fit(train[feat_solo], train['PESO_REAL'])
    

    train['fator_p'] = train['DT'].map(f_map).fillna(1.0)
    feat_ens = ['LAT', 'LON', 'H', 'fator_p']
    xgb_ens = XGBRegressor(n_estimators=100).fit(train[feat_ens], train['PESO_REAL'])

   
    test['fator_p'] = test['DT'].map(f_map).fillna(1.0)
    test['eh_feriado'] = test['DT'].astype(str).isin(FERIADOS_2025)

    def avaliar(df, label):
        y = df['PESO_REAL']
        p_h = df['PESO_HEURISTICA']
        p_s = xgb_solo.predict(df[feat_solo])
        p_e = xgb_ens.predict(df[feat_ens])
        
        return {
            "Label": label,
            "Heuristica": mean_absolute_error(y, p_h),
            "XGB_Solo": mean_absolute_error(y, p_s),
            "Ensemble": mean_absolute_error(y, p_e)
        }

    res_norm = avaliar(test[~test['eh_feriado']], "DIAS NORMAIS")
    res_fer = avaliar(test[test['eh_feriado']], "FERIADOS")

   
    relatorio = "=== RELATÓRIO DE PERFORMANCE ===\n\n"
    for r in [res_norm, res_fer]:
        relatorio += f"CENÁRIO: {r['Label']}\n"
        relatorio += f"- MAE Heurística: {r['Heuristica']:.4f}\n"
        relatorio += f"- MAE XGBoost Solo:   {r['XGB_Solo']:.4f}\n"
        relatorio += f"- MAE XGBoost + Prophet:   {r['Ensemble']:.4f}\n"
        ganho = ((r['XGB_Solo'] - r['Ensemble']) / r['XGB_Solo']) * 100
        relatorio += f"-> Ganho do Ensemble sobre XGB Solo: {ganho:.2f}%\n\n"

    with open("resultado_benchmarking.txt", "w", encoding="utf-8") as f:
        f.write(relatorio)
    print(relatorio)

if __name__ == "__main__":
    benchmark_final()
