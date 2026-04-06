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

class MotorAnaliseSafeDriver:
    def __init__(self):
        self.raiz = Path(".")
        self.pastas = {
            "bronze": self.raiz / "datalake" / "bronze",
            "ouro": self.raiz / "datalake" / "ouro",
            "auditoria": self.raiz / "datalake" / "auditoria"
        }
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        self.hoje = datetime.now()
        self.webhook = os.environ.get("DISCORD_SUCESSO")

    def _gerar_assinatura(self, caminho):
        sha = hashlib.sha256()
        with open(caminho, "rb") as f:
            for bloco in iter(lambda: f.read(4096), b""): sha.update(bloco)
        return sha.hexdigest()

    def processar_pipeline_comparativo(self, df_raw):
        df = df_raw.copy()
        del df_raw
        gc.collect()

        df.columns = [str(c).upper().strip() for c in df.columns]
        df['DATA_DT'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        df.dropna(subset=['DATA_DT', 'LATITUDE', 'LONGITUDE'], inplace=True)

        df['LAT'] = pd.to_numeric(df['LATITUDE']).astype(np.float32)
        df['LON'] = pd.to_numeric(df['LONGITUDE']).astype(np.float32)
        df['H3_INDEX'] = [h3.latlng_to_cell(lat, lon, 8) for lat, lon in zip(df['LAT'], df['LON'])]
        
        dias_atraso = (self.hoje - df['DATA_DT']).dt.days
        df['PESO_RECENCIA'] = np.where(dias_atraso <= 180, 3.0, 1.0).astype(np.float32)
        df['RISCO_REAL'] = (np.where(df['RUBRICA'].str.contains('ROUBO', na=False), 20, 5) * df['PESO_RECENCIA']).astype(np.float32)
        df['TURNO'] = pd.cut(df['DATA_DT'].dt.hour, bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]).astype(np.int8)

        fato = df.groupby(['H3_INDEX', 'TURNO', 'DATA_DT']).agg({
            'RISCO_REAL': 'sum', 'LAT': 'mean', 'LON': 'mean'
        }).reset_index()
        
        del df
        gc.collect()

        fato['IS_PAGAMENTO'] = fato['DATA_DT'].dt.day.isin([5,6,7,20,21]).astype(np.int8)
        fato['DIA_SEMANA'] = fato['DATA_DT'].dt.dayofweek.astype(np.int8)
        
        X = fato[['LAT', 'LON', 'TURNO', 'IS_PAGAMENTO', 'DIA_SEMANA']]
        y = np.log1p(fato['RISCO_REAL'])
        
        X_s = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)

        cat = CatBoostRegressor(iterations=800, depth=8, learning_rate=0.05, silent=True).fit(X_train, y_train)
        lgbm = LGBMRegressor(n_estimators=800, max_depth=10, learning_rate=0.05, verbose=-1).fit(X_train, y_train)

        fato['RISCO_PREDITO'] = np.round(np.expm1((cat.predict(X_s) * 0.6) + (lgbm.predict(X_s) * 0.4)), 2).astype(np.float32)
        fato['DESVIO_ABS'] = np.abs(fato['RISCO_REAL'] - fato['RISCO_PREDITO']).astype(np.float32)
        
        fato.to_parquet(self.pastas["ouro"] / "fato_risco_comparativo.parquet", index=False)
        
        cols_comparativo = ['H3_INDEX', 'DATA_DT', 'TURNO', 'RISCO_REAL', 'RISCO_PREDITO', 'DESVIO_ABS']
        caminho_csv = self.pastas["ouro"] / "dashboard_comparativo_real_ia.csv"
        fato[cols_comparativo].to_csv(caminho_csv, index=False)

        r2 = r2_score(y_test, (cat.predict(X_test) * 0.6) + (lgbm.predict(X_test) * 0.4))
        selo = self._gerar_assinatura(caminho_csv)
        
        with open(self.pastas["auditoria"] / "controle_integridade.json", "w") as f:
            json.dump({"r2": float(r2), "sha256": selo, "timestamp": self.hoje.isoformat()}, f, indent=4)

        if self.webhook:
            requests.post(self.webhook, json={"embeds": [{"title": "🛡️ Status: Auditado", "description": f"**R²:** {r2:.2%}", "color": 3066993}]})

        return r2

if __name__ == "__main__":
    pass
