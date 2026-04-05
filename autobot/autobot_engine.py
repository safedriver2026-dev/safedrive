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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

app = FastAPI(title="SafeDriver API")
CACHE_OURO_CONSOLIDADO = None

def notificar_discord(sucesso: bool, mensagem: str):
    webhook_url = os.environ.get("DISCORD_SUCESSO") if sucesso else os.environ.get("DISCORD_ERRO")
    if webhook_url:
        try:
            requests.post(webhook_url, json={"content": mensagem}, timeout=5)
        except:
            pass

class AutobotPipeline:
    def __init__(self, ano_atual):
        self.ano_atual = ano_atual
        self.anos_historico = list(range(2022, self.ano_atual + 1))
        self.raiz = Path(".")
        self.bronze = self.raiz / "datalake" / "bronze"
        self.prata = self.raiz / "datalake" / "prata"
        self.ouro = self.raiz / "datalake" / "ouro"
        self.arquivo_cache_geo = self.raiz / "geo_cache.json"
        self.geolocator = Nominatim(user_agent="safedriver_bot_v12")
        self.colunas_verificacao = ['LATITUDE', 'LONGITUDE', 'RUBRICA', 'NOME_MUNICIPIO', 'BAIRRO', 'LOGRADOURO', 'DESC_PERIODO']

    def preparar_ambiente(self):
        for diretorio in [self.bronze, self.prata, self.ouro]:
            diretorio.mkdir(parents=True, exist_ok=True)

    def ingerir_bronze(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        arquivo_temp = self.bronze / f"download_{ano}.xlsx"
        arquivo_parquet = self.bronze / f"bruto_{ano}.parquet"
        if arquivo_parquet.exists(): return pd.read_parquet(arquivo_parquet)
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(arquivo_temp, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): f.write(chunk)
            xls = pd.ExcelFile(arquivo_temp)
            df_final = pd.DataFrame()
            for sheet in xls.sheet_names:
                previa = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=50)
                linha_header = -1
                for i, linha in previa.iterrows():
                    valores = [str(v).strip().upper() for v in linha.values if pd.notna(v)]
                    if len(set(self.colunas_verificacao).intersection(set(valores))) >= 4:
                        linha_header = i
                        break
                if linha_header != -1:
                    df = pd.read_excel(xls, sheet_name=sheet, header=linha_header)
                    df.columns = [str(c).strip().upper() for c in df.columns]
                    cols = [c for c in self.colunas_verificacao if c in df.columns]
                    df_final = df[cols].copy()
                    break
            arquivo_temp.unlink()
            if not df_final.empty: df_final.to_parquet(arquivo_parquet, index=False)
            return df_final if not df_final.empty else None
        except: return None

    def refinar_prata(self, df_bronze, ano):
        df = df_bronze.copy()
        df.columns = [str(c).lower() for c in df.columns]
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        df_limpo = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
        df_limpo.to_parquet(self.prata / f"prata_{ano}.parquet", index=False)
        return df_limpo

    def consolidar_ouro(self, df_historico):
        df_historico['h3_index'] = df_historico.apply(
            lambda x: h3.latlng_to_cell(float(x['latitude']), float(x['longitude']), 9), axis=1
        )
        analise = df_historico.groupby(['h3_index', 'desc_periodo']).size().reset_index(name='crimes')
        analise['lat'] = analise['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        analise['lon'] = analise['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        analise['periodo_cat'] = analise['desc_periodo'].astype('category').cat.codes

        X = analise[['lat', 'lon', 'periodo_cat']]
        y = analise['crimes']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1, random_state=42)
        model_cat = CatBoostRegressor(n_estimators=100, learning_rate=0.05, verbose=0, random_state=42)
        model_knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
        
        model_lgb.fit(X_train, y_train)
        model_cat.fit(X_train, y_train)
        model_knn.fit(X_train, y_train)

        p_lgb = model_lgb.predict(X_test)
        p_cat = model_cat.predict(X_test)
        p_knn = model_knn.predict(X_test)
        
        pred_final = (p_lgb * 0.4) + (p_cat * 0.4) + (p_knn * 0.2)

        metricas = {
            "MAE": round(mean_absolute_error(y_test, pred_final), 2),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, pred_final)), 2),
            "R2": round(r2_score(y_test, pred_final), 4),
            "Ultima_Atualizacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.ouro / "metricas_modelo.json", "w") as f:
            json.dump(metricas, f, indent=4)

        analise['score_bruto'] = (model_lgb.predict(X) * 0.4) + (model_cat.predict(X) * 0.4) + (model_knn.predict(X) * 0.2)
        analise['score_risco'] = ((analise['score_bruto'] - analise['score_bruto'].min()) / 
                                 (analise['score_bruto'].max() - analise['score_bruto'].min()) * 100).round(2)

        explainer = shap.TreeExplainer(model_lgb)
        shap_values = explainer.shap_values(X)
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(self.ouro / "explica_ia_shap.png", bbox_inches='tight')
        plt.close()

        df_final = df_historico.merge(analise[['h3_index', 'score_risco']], on='h3_index', how='left')
        df_final.to_parquet(self.ouro / "ouro_consolidado.parquet", index=False)
        return df_final

    def renderizar_mapa(self, df_ouro):
        df_resumo = df_ouro.drop_duplicates(subset=['h3_index']).copy()
        mapa = folium.Map(location=[-23.5505, -46.6333], zoom_start=11, tiles='CartoDB dark_matter')
        for _, row in df_resumo.iterrows():
            score = row.get('score_risco', 0)
            cor = '#00FF00' if score < 5 else '#FF0000'
            lat, lon = h3.cell_to_latlng(row['h3_index'])
            folium.CircleMarker(location=[lat, lon], radius=5, color=cor, fill=True, fill_opacity=0.6).add_to(mapa)
        mapa.save(str(self.ouro / "mapa_estrategico.html"))

    def executar_pipeline(self):
        try:
            self.preparar_ambiente()
            colecao_prata = []
            for ano in self.anos_historico:
                bruto = self.ingerir_bronze(ano)
                if bruto is not None: colecao_prata.append(self.refinar_prata(bruto, ano))
            if colecao_prata:
                df_total = pd.concat(colecao_prata, ignore_index=True)
                df_ouro = self.consolidar_ouro(df_total)
                self.renderizar_mapa(df_ouro)
                global CACHE_OURO_CONSOLIDADO
                CACHE_OURO_CONSOLIDADO = df_ouro.drop_duplicates(subset=['h3_index'])[['h3_index', 'score_risco']].set_index('h3_index').to_dict('index')
            notificar_discord(sucesso=True, mensagem=f"✅ SafeDriver Atualizado: {self.ano_atual}")
        except Exception as e:
            notificar_discord(sucesso=False, mensagem=f"🚨 Falha no Pipeline: {str(e)}")
            raise e

@app.get("/risco/{lat}/{lon}")
def consultar(lat: float, lon: float):
    if not CACHE_OURO_CONSOLIDADO: raise HTTPException(status_code=503)
    h3_idx = h3.latlng_to_cell(lat, lon, 9)
    info = CACHE_OURO_CONSOLIDADO.get(h3_idx, {"score_risco": 0.0})
    return {"h3": h3_idx, "risco": info['score_risco'], "status": "SEGURO" if info['score_risco'] < 5 else "ALERTA"}

if __name__ == "__main__":
    ano = int(os.environ.get("ANO_PROCESSAMENTO", datetime.now().year))
    AutobotPipeline(ano).executar_pipeline()
