import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def testar_ensemble_total(p23, p24, p25):
    # Carga e Setup
    train = pd.concat([pd.read_parquet(p23), pd.read_parquet(p24)])
    test = pd.read_parquet(p25)

    def prep(df):
        df['LAT'] = df['LATITUDE'].fillna(df['LATITUDE'].median())
        df['LON'] = df['LONGITUDE'].fillna(df['LONGITUDE'].median())
        df['H'] = pd.to_numeric(df['HORA_OCORRENCIA_BO'].str.split(':').str[0], errors='coerce').fillna(0)
        return df

    train, test = prep(train), prep(test)

    # Prophet
    df_p = train.groupby('DATA').size().reset_index(name='y').rename(columns={'DATA':'ds'})
    m = Prophet(yearly_seasonality=True).fit(df_p)
    forecast = m.predict(m.make_future_dataframe(periods=365))
    forecast['fator_p'] = forecast['yhat'] / forecast['yhat'].mean()
    f_map = dict(zip(forecast['ds'].dt.date, forecast['fator_p']))

    # XGBoost 
    # XGB Solo (Sem Prophet)
    feat_solo = ['LAT', 'LON', 'H']
    xgb_solo = XGBRegressor().fit(train[feat_solo], train['PESO_REAL'])
    
    # XGB Ensemble com Prophet
    train['fator_p'] = train['DATA'].map(f_map).fillna(1.0)
    feat_ens = ['LAT', 'LON', 'H', 'fator_p']
    xgb_ens = XGBRegressor().fit(train[feat_ens], train['PESO_REAL'])

    # Predições em 2025
    y = test['PESO_REAL']
    test['fator_p'] = test['DATA'].map(f_map).fillna(1.0)
    
    p_h = test['PESO_HEURISTICA']
    p_solo = xgb_solo.predict(test[feat_solo])
    p_ens = xgb_ens.predict(test[feat_ens])

    # Métricas
    def m_res(pred): return {"MAE": mean_absolute_error(y, pred), "RMSE": np.sqrt(mean_squared_error(y, pred))}

    res = {"Heurística": m_res(p_h), "XGB Solo": m_res(p_solo), "Ensemble": m_res(p_ens)}
    
    # Lift (Top 10%)
    t10 = int(len(test) * 0.1)
    lift_final = y.iloc[p_ens.argsort()[-t10:]].sum() / y.iloc[p_h.argsort()[-t10:]].sum()

    print("\n--- RESULTADO DE PERFORMANCE ---")
    for k, v in res.items(): print(f"{k}: MAE={v['MAE']:.4f} | RMSE={v['RMSE']:.4f}")
    print(f"\n🚀 GANHO DO ENSEMBLE SOBRE XGB SOLO: {((res['XGB Solo']['MAE'] - res['Ensemble']['MAE'])/res['XGB Solo']['MAE'])*100:.2f}%")
    print(f" LIFT FINAL (IA vs REGRA): {lift_final:.2f}x")

# testar_ensemble_total('ssp_23.parquet', 'ssp_24.parquet', 'ssp_25.parquet')
