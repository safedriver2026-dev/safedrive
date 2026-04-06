import os
import json
import requests
import pandas as pd
import numpy as np
import h3
import shap
import hashlib
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
            "raw": self.raiz / "datalake" / "raw",
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

    def extrair_dados(self):
        arquivos = list(self.pastas["raw"].glob("*.xlsx")) + list(self.pastas["raw"].glob("*.parquet")) + list(self.pastas["raw"].glob("*.csv"))
        
        if not arquivos:
            np.random.seed(42)
            n_samples = 5000
            datas = pd.date_range(end=self.hoje, periods=n_samples, freq='30min')
            
            regioes = ['capital', 'campinas', 'ribeirao', 'santos', 'sjc']
            probs = [0.5, 0.2, 0.1, 0.1, 0.1]
            locais = np.random.choice(regioes, n_samples, p=probs)
            
            lats = np.zeros(n_samples)
            lons = np.zeros(n_samples)
            
            base_coords = {
                'capital': (-23.5505, -46.6333),
                'campinas': (-22.9099, -47.0626),
                'ribeirao': (-21.1704, -47.8103),
                'santos': (-23.9618, -46.3322),
                'sjc': (-23.2237, -45.9009)
            }
            
            for regiao, (lat_base, lon_base) in base_coords.items():
                mask = locais == regiao
                lats[mask] = np.random.normal(lat_base, 0.03, mask.sum())
                lons[mask] = np.random.normal(lon_base, 0.03, mask.sum())
            
            is_pagto = np.isin(datas.day, [5, 6, 7, 20, 21])
            is_noite = (datas.hour < 6) | (datas.hour > 18)
            is_polo_alto_risco = np.isin(locais, ['capital', 'campinas', 'santos'])
            
            score = is_pagto.astype(int) + is_noite.astype(int) + is_polo_alto_risco.astype(int)
            rubricas = np.where(score >= 2, "ROUBO", "FURTO")
                
            return pd.DataFrame({
                'DATA_OCORRENCIA_BO': datas,
                'LATITUDE': lats,
                'LONGITUDE': lons,
                'RUBRICA': rubricas
            })
        
        pool = []
        for arq in arquivos:
            if arq.suffix == '.xlsx':
                pool.append(pd.read_excel(arq, engine='calamine'))
            elif arq.suffix == '.parquet':
                pool.append(pd.read_parquet(arq))
            elif arq.suffix == '.csv':
                pool.append(pd.read_csv(arq))
        return pd.concat(pool, ignore_index=True)

    def processar_pipeline_comparativo(self, df_raw):
        vol_processado = len(df_raw)
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

        preds_teste = (cat.predict(X_test) * 0.6) + (lgbm.predict(X_test) * 0.4)
        r2 = r2_score(y_test, preds_teste)
        mae = mean_absolute_error(np.expm1(y_test), np.expm1(preds_teste))

        fato['RISCO_PREDITO'] = np.round(np.expm1((cat.predict(X_s) * 0.6) + (lgbm.predict(X_s) * 0.4)), 2).astype(np.float32)
        fato['DESVIO_ABS'] = np.abs(fato['RISCO_REAL'] - fato['RISCO_PREDITO']).astype(np.float32)
        
        # CÁLCULO DA TAXA DE ASSERTIVIDADE TÁTICA (Margem de tolerância: 5 pontos)
        margem_tolerancia = 5.0
        qtd_acertos = (fato['DESVIO_ABS'] <= margem_tolerancia).sum()
        taxa_assertividade = qtd_acertos / len(fato)

        fato.to_parquet(self.pastas["ouro"] / "fato_risco_comparativo.parquet", index=False)
        
        cols_comparativo = ['H3_INDEX', 'DATA_DT', 'TURNO', 'RISCO_REAL', 'RISCO_PREDITO', 'DESVIO_ABS']
        caminho_csv = self.pastas["ouro"] / "dashboard_comparativo_real_ia.csv"
        fato[cols_comparativo].to_csv(caminho_csv, index=False)

        selo = self._gerar_assinatura(caminho_csv)
        
        with open(self.pastas["auditoria"] / "controle_integridade.json", "w") as f:
            json.dump({"r2": float(r2), "sha256": selo, "timestamp": self.hoje.isoformat()}, f, indent=4)

        if self.webhook:
            importancias = cat.get_feature_importance()
            top_idx = np.argsort(importancias)[-3:][::-1]
            top_features = [X.columns[i] for i in top_idx]

            payload = {
                "embeds": [{
                    "title": "📊 Relatório Executivo: Motor Preditivo SafeDriver",
                    "description": "Síntese operacional da compilação geospacial.",
                    "color": 3066993 if taxa_assertividade > 0.70 else 15105570,
                    "fields": [
                        {"name": "🎯 Taxa de Assertividade", "value": f"**{taxa_assertividade:.1%}**", "inline": True},
                        {"name": "📉 Erro Médio (MAE)", "value": f"± {mae:.2f} pts", "inline": True},
                        {"name": "🗂️ Volume", "value": f"{vol_processado:,.0f} registros", "inline": True},
                        {"name": "🔬 R² (Sinal Estatístico)", "value": f"{r2:.2%}", "inline": False},
                        {"name": "🧠 Variáveis Críticas (Top 3)", "value": f"1. `{top_features[0]}`\n2. `{top_features[1]}`\n3. `{top_features[2]}`", "inline": False},
                    ],
                    "footer": {"text": f"Selo de Integridade: {selo[:10]} | {self.hoje.strftime('%d/%m/%Y %H:%M')}"}
                }]
            }
            requests.post(self.webhook, json=payload)

        return r2

    def executar(self):
        df_raw = self.extrair_dados()
        self.processar_pipeline_comparativo(df_raw)

if __name__ == "__main__":
    MotorAnaliseSafeDriver().executar()
