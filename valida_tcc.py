import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import os
import glob

DATAS_CRITICAS = [
    '2025-01-01', '2025-03-03', '2025-03-04', '2025-04-18', 
    '2025-05-01', '2025-09-07', '2025-11-20', '2025-12-25'
]

def localizar_ativo(nome):
    match = glob.glob(f"**/{nome}", recursive=True)
    if not match:
        raise FileNotFoundError(f"Ativo {nome} nao encontrado.")
    return match[0]

def rodar_experimento():
    print("Localizando massas de dados...")
    try:
        f23 = localizar_ativo('ssp_2023.parquet')
        f24 = localizar_ativo('ssp_2024.parquet')
        f25 = localizar_ativo('ssp_2025.parquet')
        df_train = pd.concat([pd.read_parquet(f23), pd.read_parquet(f24)])
        df_test = pd.read_parquet(f25)
    except Exception as e:
        print(f"Erro na carga: {e}")
        return

    def pipeline_tratamento(df):
    
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
        
        df['LAT'] = df['LATITUDE'].fillna(df['LATITUDE'].median())
        df['LON'] = df['LONGITUDE'].fillna(df['LONGITUDE'].median())
        df['H'] = pd.to_numeric(df['HORA_OCORRENCIA_BO'].str.split(':').str[0], errors='coerce').fillna(0)
        df['DT'] = pd.to_datetime(df['DATA']).dt.date
        return df

    print("Tratando dados e convertendo tipos...")
    df_train = pipeline_tratamento(df_train)
    df_test = pipeline_tratamento(df_test)

    # Prophet
    print("Processando serie temporal...")
    df_p = df_train.groupby('DT').size().reset_index(name='y').rename(columns={'DT':'ds'})
    m_p = Prophet(yearly_seasonality=True).fit(df_p)
    fcst = m_p.predict(m_p.make_future_dataframe(periods=365))
    f_map = dict(zip(fcst['ds'].dt.date, (fcst['yhat'] / fcst['yhat'].mean())))

    #  XGBoost Solo
    print("Processando motor espacial solo...")
    feats_s = ['LAT', 'LON', 'H']
    xgb_s = XGBRegressor(n_estimators=100).fit(df_train[feats_s], df_train['PESO_REAL'])
    
    # Ensemble
    print("Processando motor hibrido...")
    df_train['f_p'] = df_train['DT'].map(f_map).fillna(1.0)
    feats_e = ['LAT', 'LON', 'H', 'f_p']
    xgb_e = XGBRegressor(n_estimators=100).fit(df_train[feats_e], df_train['PESO_REAL'])

    df_test['f_p'] = df_test['DT'].map(f_map).fillna(1.0)
    df_test['is_critico'] = df_test['DT'].astype(str).isin(DATAS_CRITICAS)

    def extrair_performance(subset):
        y = subset['PESO_REAL']
        return {
            'Heuristica': mean_absolute_error(y, subset['PESO_HEURISTICA']),
            'XGB_Solo': mean_absolute_error(y, xgb_s.predict(subset[feats_s])),
            'Ensemble': mean_absolute_error(y, xgb_e.predict(subset[feats_e]))
        }

    res_normal = extrair_performance(df_test[~df_test['is_critico']])
    res_critico = extrair_performance(df_test[df_test['is_critico']])

    def vencedor(d): return min(d, key=d.get)

    relatorio = f"""RELATORIO DE PERFORMANCE
--------------------------------------------------
CENARIO: ROTINA
- Heuristica: {res_normal['Heuristica']:.4f}
- XGBoost Solo: {res_normal['XGB_Solo']:.4f}
- Ensemble Hibrido: {res_normal['Ensemble']:.4f}
Melhor: {vencedor(res_normal)}

CENARIO: ANOMALIAS (FERIADOS)
- Heuristica: {res_critico['Heuristica']:.4f}
- XGBoost Solo: {res_critico['XGB_Solo']:.4f}
- Ensemble Hibrido: {res_critico['Ensemble']:.4f}
Melhor: {vencedor(res_critico)}
--------------------------------------------------
"""
    with open("validacao_motores_2025.txt", "w", encoding="utf-8") as f:
        f.write(relatorio)
    print("Relatorio gerado.")

if __name__ == "__main__":
    rodar_experimento()
