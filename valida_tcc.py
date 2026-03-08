import pandas as pd
import numpy as np
import unicodedata
import os
import glob
from xgboost import XGBRegressor
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

# Configurações extraídas do seu MotorSafeDriver
LIMITES_SP = {'lat': (-25.5, -19.5), 'lon': (-53.5, -44.0)}
CATALOGO_CRIMES = {
    'FURTO DE VEICULO': 1.0, 'ROUBO DE VEICULO': 2.5, 'ROUBO DE CARGA': 2.5,
    'FURTO DE CARGA': 1.0, 'LATROCINIO': 5.0, 'EXTORSAO MEDIANTE SEQUESTRO': 5.0,
    'ROUBO A TRANSEUNTE': 2.5, 'FURTO DE CELULAR': 1.0
}

class AuditoriaSafeDriver:
    def __init__(self):
        self.mes_controle = 8  # Agosto (Estabilidade)
        self.mes_estresse = 11 # Novembro (Alta Sazonalidade)

    def _higienizar_texto(self, texto_bruto):
        """Replica exatamente o seu método oficial"""
        if pd.isna(texto_bruto) or not isinstance(texto_bruto, str): 
            return str(texto_bruto) if not pd.isna(texto_bruto) else ""
        texto_normalizado = unicodedata.normalize('NFKD', texto_bruto)
        return "".join([c for c in texto_normalizado if not unicodedata.combining(c)]).upper().strip()

    def _tratar_massa_dados(self, df):
        """Replica a lógica de limpeza do seu pipeline principal"""
        # 1. Higienização de colunas
        df.columns = [self._higienizar_texto(c) for c in df.columns]

        # 2. Mapeamento de correção oficial do seu MotorSafeDriver
        mapeamento = {
            'NUMERO_BO': 'NUM_BO', 'N_BO': 'NUM_BO', 
            'LAT': 'LATITUDE', 'LON': 'LONGITUDE', 
            'DATA_FATO': 'DATA_OCORRENCIA_BO', 
            'HORA_FATO': 'HORA_OCORRENCIA_BO'
        }
        df.rename(columns=mapeamento, inplace=True)

        # 3. Conversão Numérica (Trusted Layer)
        for col in ['LATITUDE', 'LONGITUDE']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors="coerce")

        # 4. Máscara Geográfica (Trusted Layer)
        mascara_geo = (
            df['LATITUDE'].notna() & df['LONGITUDE'].notna() &
            (df['LATITUDE'] != 0) & (df['LONGITUDE'] != 0) &
            df['LATITUDE'].between(LIMITES_SP['lat'][0], LIMITES_SP['lat'][1]) &
            df['LONGITUDE'].between(LIMITES_SP['lon'][0], LIMITES_SP['lon'][1])
        )
        df = df[mascara_geo].copy()

        # 5. Normalização de Tempo e Alvo
        df['DT'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        df['H'] = pd.to_numeric(df['HORA_OCORRENCIA_BO'].astype(str).str.split(':').str[0], errors='coerce').fillna(0)
        
        # Define o Target (Peso Real)
        df['PESO_REAL'] = df['NATUREZA_APURADA'].map(CATALOGO_CRIMES).fillna(1.0)
        
        # Heurística simples (Baseline de média global)
        df['PESO_HEURISTICA'] = df['PESO_REAL'].mean()

        return df.dropna(subset=['DT', 'LATITUDE', 'LONGITUDE'])

    def executar_teste(self):
        print("Buscando parquets...")
        f23 = glob.glob('**/ssp_2023.parquet', recursive=True)[0]
        f24 = glob.glob('**/ssp_2024.parquet', recursive=True)[0]
        f25 = glob.glob('**/ssp_2025.parquet', recursive=True)[0]

        df_train = self._tratar_massa_dados(pd.concat([pd.read_parquet(f23), pd.read_parquet(f24)]))
        df_test = self._tratar_massa_dados(pd.read_parquet(f25))

        # --- MODELOS ---
        feats = ['LATITUDE', 'LONGITUDE', 'H']
        
        # 1. Solo
        m_xgb = XGBRegressor(n_estimators=50).fit(df_train[feats], df_train['PESO_REAL'])
        
        # 2. Prophet (Camada Sazonal)
        df_p = df_train.groupby(df_train['DT'].dt.date).size().reset_index(name='y').rename(columns={'DT':'ds'})
        m_prophet = Prophet(yearly_seasonality=True).fit(df_p)
        fcst = m_prophet.predict(m_prophet.make_future_dataframe(periods=365))
        f_map = dict(zip(fcst['ds'].dt.date, (fcst['yhat'] / (fcst['yhat'].mean() + 1e-6))))

        # 3. Ensemble (Híbrido)
        df_train['f_p'] = df_train['DT'].dt.date.map(f_map).fillna(1.0)
        m_ens = XGBRegressor(n_estimators=50).fit(df_train[feats + ['f_p']], df_train['PESO_REAL'])

        # --- VALIDAÇÃO ---
        df_ago = df_test[df_test['DT'].dt.month == self.mes_controle].copy()
        df_nov = df_test[df_test['DT'].dt.month == self.mes_estresse].copy()

        def medir(df_sub):
            if df_sub.empty: return {"H": 0.0, "X": 0.0, "E": 0.0}
            df_sub['f_p'] = df_sub['DT'].dt.date.map(f_map).fillna(1.0)
            y = df_sub['PESO_REAL']
            return {
                "H": mean_absolute_error(y, df_sub['PESO_HEURISTICA']),
                "X": mean_absolute_error(y, m_xgb.predict(df_sub[feats])),
                "E": mean_absolute_error(y, m_ens.predict(df_sub[feats + ['f_p']]))
            }

        res_ago = medir(df_ago)
        res_nov = medir(df_nov)

        # Relatório Final
        formula_mae = r"$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$"
        
        relatorio = f"""RELATORIO DE PERFORMANCE
--------------------------------------------------
METRICA DE ERRO: {formula_mae}

CENARIO 01: AGOSTO (CONTROLE - ESTABILIDADE)
- MAE Heuristica: {res_ago['H']:.6f}
- MAE XGBoost Solo: {res_ago['X']:.6f}
- MAE Ensemble Hibrido: {res_ago['E']:.6f}
Veredito: {min(res_ago, key=res_ago.get)}

CENARIO 02: NOVEMBRO (ESTRESSE - SAZONALIDADE)
- MAE Heuristica: {res_nov['H']:.6f}
- MAE XGBoost Solo: {res_nov['X']:.6f}
- MAE Ensemble Hibrido: {res_nov['E']:.6f}
Veredito: {min(res_nov, key=res_nov.get)}

IMPACTO DO COMPONENTE TEMPORAL (NOVEMBRO):
Melhoria vs XGBoost Solo: {((res_nov['X'] - res_nov['E'])/res_nov['X'])*100:+.2f}%
--------------------------------------------------
"""
        with open("validacao_motores_2025.txt", "w", encoding="utf-8") as f:
            f.write(relatorio)

if __name__ == "__main__":
    AuditoriaSafeDriver().executar_teste()
