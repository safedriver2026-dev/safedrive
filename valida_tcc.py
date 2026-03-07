import pandas as pd
import numpy as np
import unicodedata
import os
import glob
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# Simulando constantes do seu config.py caso o import falhe no CI
LIMITES_SP = {'lat': [-24.5, -23.0], 'lon': [-47.5, -45.0]} 

class ValidadorTecnico:
    def __init__(self):
        self.datas_criticas = [
            '2025-01-01', '2025-03-03', '2025-03-04', '2025-04-18', 
            '2025-05-01', '2025-09-07', '2025-11-20', '2025-12-25'
        ]

    def _higienizar_texto(self, texto_bruto):
        if pd.isna(texto_bruto) or not isinstance(texto_bruto, str): 
            return str(texto_bruto) if not pd.isna(texto_bruto) else ""
        texto_normalizado = unicodedata.normalize('NFKD', texto_bruto)
        return "".join([c for c in texto_normalizado if not unicodedata.combining(c)]).upper().strip()

    def _pipeline_tratamento_oficial(self, df):
        #
        mapeamento = {'NUMERO_BO': 'NUM_BO', 'N_BO': 'NUM_BO', 'LAT': 'LATITUDE', 'LON': 'LONGITUDE'}
        df.rename(columns=mapeamento, inplace=True)
        df.columns = [self._higienizar_texto(c) for c in df.columns]

     
        df['LATITUDE'] = pd.to_numeric(df['LATITUDE'].astype(str).str.replace(',', '.'), errors="coerce")
        df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'].astype(str).str.replace(',', '.'), errors="coerce")
        
      
        mascara_geo = (
            df['LATITUDE'].notna() & df['LONGITUDE'].notna() &
            (df['LATITUDE'] != 0) & (df['LONGITUDE'] != 0) &
            df['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1]) &
            df['LONGITUDE'].between(LIMITES_SP['lon'][0], LIMITES_SP['lon'][1])
        )
        df = df[mascara_geo].copy()

      
        col_data = 'DATA_OCORRENCIA_BO' if 'DATA_OCORRENCIA_BO' in df.columns else 'DATA'
        df['DT'] = pd.to_datetime(df[col_data], errors='coerce').dt.date
        df['H'] = pd.to_numeric(df['HORA_OCORRENCIA_BO'].astype(str).str.split(':').str[0], errors='coerce').fillna(0)
        
       
        df['LAT'] = df['LATITUDE']
        df['LON'] = df['LONGITUDE']
        
        return df.dropna(subset=['DT', 'LAT', 'LON'])

    def localizar_arquivo(self, nome):
        match = glob.glob(f"**/{nome}", recursive=True)
        if not match: raise FileNotFoundError(f"Ativo {nome} não localizado.")
        return match[0]

    def rodar_teste_stress(self):
        print("Iniciando auditoria técnica replicando pipeline SafeDriver...")
        
        
        f23 = self.localizar_arquivo('ssp_2023.parquet')
        f24 = self.localizar_arquivo('ssp_2024.parquet')
        f25 = self.localizar_arquivo('ssp_2025.parquet')
        
        df_train = pd.concat([pd.read_parquet(f23), pd.read_parquet(f24)])
        df_test = pd.read_parquet(f25)

 
        df_train = self._pipeline_tratamento_oficial(df_train)
        df_test = self._pipeline_tratamento_oficial(df_test)

     
        df_p = df_train.groupby('DT').size().reset_index(name='y').rename(columns={'DT':'ds'})
        m_p = Prophet(yearly_seasonality=True).fit(df_p)
        fcst = m_p.predict(m_p.make_future_dataframe(periods=365))
        f_map = dict(zip(fcst['ds'].dt.date, (fcst['yhat'] / fcst['yhat'].mean())))

  
        feats_s = ['LAT', 'LON', 'H']
        xgb_solo = XGBRegressor(n_estimators=100).fit(df_train[feats_s], df_train['PESO_REAL'])
        
        df_train['f_p'] = df_train['DT'].map(f_map).fillna(1.0)
        feats_e = feats_s + ['f_p']
        xgb_ens = XGBRegressor(n_estimators=100).fit(df_train[feats_e], df_train['PESO_REAL'])

       
        df_test['f_p'] = df_test['DT'].map(f_map).fillna(1.0)
        df_test['is_critico'] = df_test['DT'].astype(str).isin(self.datas_criticas)

        def calcular(subset):
            y = subset['PESO_REAL']
            p_heu = subset['PESO_HEURISTICA'] if 'PESO_HEURISTICA' in subset.columns else np.zeros(len(subset))
            return {
                'Heuristica': mean_absolute_error(y, p_heu),
                'XGB_Solo': mean_absolute_error(y, xgb_solo.predict(subset[feats_s])),
                'Ensemble': mean_absolute_error(y, xgb_ens.predict(subset[feats_e]))
            }

        res_n = calcular(df_test[~df_test['is_critico']])
        res_c = calcular(df_test[df_test['is_critico']])

       
        relatorio = f"""RELATORIO TECNICO DE PERFORMANCE 
--------------------------------------------------
CENARIO 1: DIAS NORMAIS
- Heuristica: {res_n['Heuristica']:.4f}
- XGBoost Solo: {res_n['XGB_Solo']:.4f}
- Ensemble Hibrido: {res_n['Ensemble']:.4f}
Veredito: {min(res_n, key=res_n.get)}

CENARIO 2: FERIADOS
- Heuristica: {res_c['Heuristica']:.4f}
- XGBoost Solo: {res_c['XGB_Solo']:.4f}
- Ensemble Hibrido: {res_c['Ensemble']:.4f}
Veredito: {min(res_c, key=res_c.get)}
--------------------------------------------------
"""
        with open("validacao_motores_2025.txt", "w", encoding="utf-8") as f:
            f.write(relatorio)
        print("Auditoria concluída com sucesso.")

if __name__ == "__main__":
    ValidadorTecnico().rodar_teste_stress()
