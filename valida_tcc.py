import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
import os
import glob

# Datas criticas para teste de desvio sazonal
DATAS_CRITICAS = [
    '2025-01-01', '2025-03-03', '2025-03-04', '2025-04-18', 
    '2025-05-01', '2025-09-07', '2025-11-20', '2025-12-25'
]

def localizar_ativo(nome):
    """Busca o arquivo no repositorio independente da estrutura de pastas."""
    match = glob.glob(f"**/{nome}", recursive=True)
    if not match:
        raise FileNotFoundError(f"Ativo {nome} nao encontrado no volume.")
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
        print(f"Erro na montagem do dataset: {e}")
        return

    def pipeline_tratamento(df):
        df['LAT'] = df['LATITUDE'].fillna(df['LATITUDE'].median())
        df['LON'] = df['LONGITUDE'].fillna(df['LONGITUDE'].median())
        df['H'] = pd.to_numeric(df['HORA_OCORRENCIA_BO'].str.split(':').str[0], errors='coerce').fillna(0)
        df['DT'] = pd.to_datetime(df['DATA']).dt.date
        return df

    df_train = pipeline_tratamento(df_train)
    df_test = pipeline_tratamento(df_test)

    # 1. Componente Temporal Prophet
    print("Processando serie temporal...")
    df_p = df_train.groupby('DT').size().reset_index(name='y').rename(columns={'DT':'ds'})
    m_p = Prophet(yearly_seasonality=True).fit(df_p)
    fcst = m_p.predict(m_p.make_future_dataframe(periods=365))
    f_map = dict(zip(fcst['ds'].dt.date, (fcst['yhat'] / fcst['yhat'].mean())))

    # 2. XGBoost
    print("Processando modelo espacial solo...")
    feats_s = ['LAT', 'LON', 'H']
    xgb_s = XGBRegressor(n_estimators=100).fit(df_train[feats_s], df_train['PESO_REAL'])
    
    # 3. Estrategia Hibrida
    print("Processando modelo hibrido...")
    df_train['f_p'] = df_train['DT'].map(f_map).fillna(1.0)
    feats_e = ['LAT', 'LON', 'H', 'f_p']
    xgb_e = XGBRegressor(n_estimators=100).fit(df_train[feats_e], df_train['PESO_REAL'])

    # Aplicacao em 2025
    df_test['f_p'] = df_test['DT'].map(f_map).fillna(1.0)
    df_test['is_critico'] = df_test['DT'].astype(str).isin(DATAS_CRITICAS)

    def extrair_performance(subset):
        y = subset['PESO_REAL']
        metrics = {
            'Heuristica': mean_absolute_error(y, subset['PESO_HEURISTICA']),
            'XGB_Solo': mean_absolute_error(y, xgb_s.predict(subset[feats_s])),
            'Ensemble': mean_absolute_error(y, xgb_e.predict(subset[feats_e]))
        }
        metrics['Vencedor'] = min(metrics, key=metrics.get)
        return metrics

    res_normal = extrair_performance(df_test[~df_test['is_critico']])
    res_critico = extrair_performance(df_test[df_test['is_critico']])

   
    relatorio = f"""RELATORIO DE PERFORMANCE TECNICA 
--------------------------------------------------
CENARIO: DIAS COMUNS
- MAE Heuristica: {res_normal['Heuristica']:.4f}
- MAE XGBoost Solo: {res_normal['XGB_Solo']:.4f}
- MAE Ensemble Hibrido: {res_normal['Ensemble']:.4f}
Motor mais eficiente: {res_normal['Vencedor']}

CENARIO: FERIADOS/DATAS CRITICAS
- MAE Heuristica: {res_critico['Heuristica']:.4f}
- MAE XGBoost Solo: {res_critico['XGB_Solo']:.4f}
- MAE Ensemble Hibrido: {res_critico['Ensemble']:.4f}
Motor mais eficiente: {res_critico['Vencedor']}

NOTAS DE DESEMPENHO:
Ganho Ensemble vs Heuristica (Critico): {((res_critico['Heuristica'] - res_critico['Ensemble'])/res_critico['Heuristica'])*100:.2f}%
Ganho Ensemble vs XGB Solo (Critico): {((res_critico['XGB_Solo'] - res_critico['Ensemble'])/res_critico['XGB_Solo'])*100:.2f}%
--------------------------------------------------
"""
    with open("validacao_motores_2025.txt", "w", encoding="utf-8") as f:
        f.write(relatorio)
    print("Processo concluido. Relatorio gerado.")

if __name__ == "__main__":
    rodar_experimento()
