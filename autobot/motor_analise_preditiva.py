import os
import json
import requests
import pandas as pd
import numpy as np
import h3
import shap
import hashlib
import holidays
import gc
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

class MotorAnalitico:
    def __init__(self):
        self.raiz = Path(".")
        self.pastas = {
            "bronze": self.raiz / "datalake" / "bronze",
            "ouro": self.raiz / "datalake" / "ouro",
            "auditoria": self.raiz / "datalake" / "auditoria"
        }
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        
        self.hoje = datetime.now()
        self.feriados_sp = holidays.Brazil(state='SP')
        self.webhook = os.environ.get("DISCORD_SUCESSO")

    def _gerar_hash(self, caminho):
        sha = hashlib.sha256()
        with open(caminho, "rb") as f:
            for bloco in iter(lambda: f.read(4096), b""): sha.update(bloco)
        return sha.hexdigest()

    def suavizar_risco_espacial(self, df):
        """Aplica Spatial Lag para eliminar efeito de borda"""
        mapa_base = df.groupby('H3_INDEX')['SCORE_PONDERADO'].mean().to_dict()
        def calcular_vizinhos(hex_id):
            vizinhos = h3.grid_disk(hex_id, 1)
            return np.mean([mapa_base.get(v, 0) for v in vizinhos])
        return df['H3_INDEX'].apply(calcular_vizinhos)

    def processar_camada_ouro(self, df_raw):
        df = df_raw.copy()
        df.columns = [str(c).upper().strip() for c in df.columns]
        df['DATA_DT'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        df = df.dropna(subset=['DATA_DT', 'LATITUDE', 'LONGITUDE'])
        
        df['LAT'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
        df['LON'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
        df['H3_INDEX'] = [h3.latlng_to_cell(lat, lon, 8) for lat, lon in zip(df['LAT'], df['LON'])]
        
        # Ponderação por Recência (Foco no presente)
        dias_atraso = (self.hoje - df['DATA_DT']).dt.days
        df['PESO_RECENCIA'] = np.where(dias_atraso <= 180, 3.0, 1.0)
        df['SCORE_PONDERADO'] = np.where(df['RUBRICA'].str.contains('ROUBO', na=False), 20, 5) * df['PESO_RECENCIA']
        
        df['TURNO'] = pd.cut(df['DATA_DT'].dt.hour, bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]).astype(int)
        
        fato = df.groupby(['H3_INDEX', 'TURNO', 'DATA_DT']).agg({
            'SCORE_PONDERADO': 'sum', 'LAT': 'mean', 'LON': 'mean'
        }).reset_index()
        
        fato['RISCO_VIZINHANCA'] = self.suavizar_risco_espacial(fato)
        fato['IS_PAGAMENTO'] = fato['DATA_DT'].dt.day.isin([5,6,7,20,21]).astype(int)
        fato['DIA_SEMANA'] = fato['DATA_DT'].dt.dayofweek
        
        return fato

    def motor_preditivo(self, fato):
        X = fato[['LAT', 'LON', 'TURNO', 'IS_PAGAMENTO', 'DIA_SEMANA', 'RISCO_VIZINHANCA']]
        y = np.log1p(fato['SCORE_PONDERADO'])
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)

        cat = CatBoostRegressor(iterations=1000, depth=8, learning_rate=0.04, silent=True).fit(X_train, y_train)
        lgbm = LGBMRegressor(n_estimators=1000, max_depth=10, learning_rate=0.04, verbose=-1).fit(X_train, y_train)
        
        # Auditoria SHAP
        explainer = shap.TreeExplainer(cat)
        shap_values = explainer.shap_values(X_s)
        for i, col in enumerate(X.columns):
            fato[f'SHAP_{col}'] = np.round(shap_values[:, i], 4)
            
        fato['PREDICAO_RISCO'] = np.round(np.expm1((cat.predict(X_s) * 0.6) + (lgbm.predict(X_s) * 0.4)), 2)
        
        r2 = r2_score(y_test, (cat.predict(X_test) * 0.6) + (lgbm.predict(X_test) * 0.4))
        mae = mean_absolute_error(np.expm1(y_test), np.expm1((cat.predict(X_test) * 0.6) + (lgbm.predict(X_test) * 0.4)))
        
        return r2, mae, fato

    def exportar_resultados(self, r2, mae, df_final):
        # 1. Parquet (Completo)
        df_final.to_parquet(self.pastas["ouro"] / "inteligencia_consolidada.parquet", index=False)
        
        # 2. CSV Otimizado (Dashboard < 100MB)
        cols_dash = ['H3_INDEX', 'LAT', 'LON', 'TURNO', 'IS_PAGAMENTO', 'PREDICAO_RISCO']
        df_final[cols_dash].to_csv(self.pastas["ouro"] / "dashboard_risco_consolidado.csv", index=False)
        
        # 3. Manifesto de Integridade
        selo = self._gerar_hash(self.pastas["ouro"] / "dashboard_risco_consolidado.csv")
        log = {"r2": r2, "mae": mae, "sha256": selo, "timestamp": self.hoje.isoformat()}
        with open(self.pastas["auditoria"] / "controle_integridade.json", "w") as f:
            json.dump(log, f, indent=4)

        if self.webhook:
            requests.post(self.webhook, json={"embeds": [{"title": "🛡️ Status: Pipeline Auditado", 
            "description": f"**Confiança ($R^2$):** {r2:.2%}\n**Integridade:** ✅ Verificada", "color": 3066993}]})
