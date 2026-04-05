import os
import json
import time
import requests
import pandas as pd
import numpy as np
import h3
import folium
import uvicorn
import shap
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from pathlib import Path
from geopy.geocoders import Nominatim
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime

app = FastAPI(title="SafeDriver API")
CACHE_OURO_CONSOLIDADO = None

def notificar_discord(sucesso: bool, mensagem: str):
    webhook = os.environ.get("DISCORD_SUCESSO") if sucesso else os.environ.get("DISCORD_ERRO")
    if webhook:
        try: requests.post(webhook, json={"content": mensagem}, timeout=5)
        except: pass

class SafeDriverEngine:
    def __init__(self):
        self.ano_atual = datetime.now().year
        self.anos = list(range(2022, self.ano_atual + 1))
        self.raiz = Path(".")
        self.camadas = {
            "bronze": self.raiz / "datalake" / "bronze",
            "prata": self.raiz / "datalake" / "prata",
            "ouro": self.raiz / "datalake" / "ouro"
        }
        for p in self.camadas.values(): p.mkdir(parents=True, exist_ok=True)
        self.geolocator = Nominatim(user_agent="safedriver_pro_v14")

    def ingestao_bronze(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        arq_pq = self.camadas["bronze"] / f"bruto_{ano}.parquet"
        if arq_pq.exists(): return pd.read_parquet(arq_pq)
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            xls = pd.ExcelFile(r.content)
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                df.columns = [str(c).upper().strip() for c in df.columns]
                if 'LATITUDE' in df.columns:
                    df.to_parquet(arq_pq, index=False)
                    return df
            return None
        except: return None

    def refinamento_prata(self, df, ano):
        df.columns = [str(c).lower() for c in df.columns]
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
        df.to_parquet(self.camadas["prata"] / f"prata_{ano}.parquet", index=False)
        return df

    def inteligencia_ouro(self, df_total):
        df_total['h3_index'] = df_total.apply(lambda x: h3.latlng_to_cell(float(x['latitude']), float(x['longitude']), 9), axis=1)
        analise = df_total.groupby(['h3_index', 'desc_periodo']).size().reset_index(name='crimes')
        analise['lat'] = analise['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        analise['lon'] = analise['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        analise['periodo_cat'] = analise['desc_periodo'].astype('category').cat.codes

        X = analise[['lat', 'lon', 'periodo_cat']]
        y = analise['crimes']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        m_lgb = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        m_cat = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)
        m_knn = KNeighborsRegressor(n_neighbors=5, weights='distance')

        m_lgb.fit(X_train, y_train); m_cat.fit(X_train, y_train); m_knn.fit(X_train, y_train)
        preds = (m_lgb.predict(X_test)*0.4) + (m_cat.predict(X_test)*0.4) + (m_knn.predict(X_test)*0.2)
        
        metricas = {"MAE": round(mean_absolute_error(y_test, preds), 2), "R2": round(r2_score(y_test, preds), 4), "Atualizacao": datetime.now().isoformat()}
        with open(self.camadas["ouro"] / "metricas.json", "w") as f: json.dump(metricas, f)

        analise['score_risco'] = (((m_lgb.predict(X)*0.4 + m_cat.predict(X)*0.4 + m_knn.predict(X)*0.2) - y.min()) / (y.max() - y.min()) * 100).round(2)
        analise.to_csv(self.camadas["ouro"] / "base_looker.csv", index=False)
        
        explainer = shap.TreeExplainer(m_lgb)
        shap_v = explainer.shap_values(X)
        plt.figure(); shap.summary_plot(shap_v, X, show=False)
        plt.savefig(self.camadas["ouro"] / "ia_audit.png", bbox_inches='tight'); plt.close()

        return df_total.merge(analise[['h3_index', 'score_risco']], on='h3_index', how='left')

    def rodar(self):
        try:
            lista = []
            for a in self.anos:
                b = self.ingestao_bronze(a)
                if b is not None: lista.append(self.refinamento_prata(b, a))
            if lista:
                df_ouro = self.inteligencia_ouro(pd.concat(lista, ignore_index=True))
                df_ouro.to_parquet(self.camadas["ouro"] / "ouro_final.parquet")
                notificar_discord(True, f"✅ SafeDriver: Ciclo {self.ano_atual} finalizado com sucesso.")
        except Exception as e:
            notificar_discord(False, f"🚨 Erro no SafeDriver: {str(e)}")

if __name__ == "__main__":
    SafeDriverEngine().rodar()
