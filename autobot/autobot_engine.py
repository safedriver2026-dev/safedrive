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
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
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
        self.geolocator = Nominatim(user_agent="safedriver_bot_v11")
        self.colunas_verificacao = ['LATITUDE', 'LONGITUDE', 'RUBRICA', 'NOME_MUNICIPIO', 'BAIRRO', 'LOGRADOURO', 'DESC_PERIODO']

    def preparar_ambiente(self):
        for diretorio in [self.bronze, self.prata, self.ouro]:
            diretorio.mkdir(parents=True, exist_ok=True)

    def ingerir_bronze(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        arquivo_temp = self.bronze / f"download_{ano}.xlsx"
        arquivo_parquet = self.bronze / f"bruto_{ano}.parquet"
        if arquivo_parquet.exists():
            return pd.read_parquet(arquivo_parquet)
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(arquivo_temp, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            xls = pd.ExcelFile(arquivo_temp)
            df_final = pd.DataFrame()
            
            for sheet in xls.sheet_names:
                previa = pd.read_excel(xls, sheet_name=sheet, header=None, nrows=50)
                linha_header = -1
                for i, linha in previa.iterrows():
                    valores_linha = [str(v).strip().upper() for v in linha.values if pd.notna(v)]
                    matches = len(set(self.colunas_verificacao).intersection(set(valores_linha)))
                    if matches >= 4:
                        linha_header = i
                        break
                
                if linha_header != -1:
                    df = pd.read_excel(xls, sheet_name=sheet, header=linha_header)
                    df.columns = [str(c).strip().upper() for c in df.columns]
                    colunas_existentes = [c for c in self.colunas_verificacao if c in df.columns]
                    if 'LATITUDE' in colunas_existentes and 'LONGITUDE' in colunas_existentes:
                        df_final = df[colunas_existentes].copy()
                        break
            
            arquivo_temp.unlink()
            if df_final.empty: return None
            df_final.to_parquet(arquivo_parquet, index=False)
            return df_final
        except:
            return None

    def carregar_cache(self):
        if self.arquivo_cache_geo.exists():
            with open(self.arquivo_cache_geo, 'r') as f: return json.load(f)
        return {}

    def salvar_cache(self, cache):
        with open(self.arquivo_cache_geo, 'w') as f: json.dump(cache, f)

    def refinar_prata(self, df_bronze, ano):
        df = df_bronze.copy()
        df.columns = [str(c).lower() for c in df.columns]
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        
        precisa_limpeza = (df['latitude'] == 0) | (df['latitude'].isna())
        df_sujo = df[precisa_limpeza].copy()
        
        if not df_sujo.empty:
            cache = self.carregar_cache()
            recuperados = 0
            for idx, row in df_sujo.iterrows():
                if recuperados >= 30: break
                endereco = f"{row.get('logradouro', '')}, {row.get('bairro', '')}, {row.get('nome_municipio', '')}, SP"
                if endereco in cache:
                    df.at[idx, 'latitude'] = cache[endereco]['lat']
                    df.at[idx, 'longitude'] = cache[endereco]['lon']
                else:
                    try:
                        time.sleep(1.1)
                        loc = self.geolocator.geocode(endereco, timeout=5)
                        if loc:
                            cache[endereco] = {'lat': loc.latitude, 'lon': loc.longitude}
                            df.at[idx, 'latitude'] = loc.latitude
                            df.at[idx, 'longitude'] = loc.longitude
                            recuperados += 1
                    except: continue
            self.salvar_cache(cache)
            
        df_limpo = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
        df_limpo.to_parquet(self.prata / f"prata_{ano}.parquet", index=False)
        return df_limpo

    def consolidar_ouro(self, df_historico):
        df_historico['h3_index'] = df_historico.apply(
            lambda x: h3.latlng_to_cell(float(x['latitude']), float(x['longitude']), 9), axis=1
        )
        
        if 'desc_periodo' not in df_historico.columns:
            df_historico['desc_periodo'] = 'Desconhecido'
            
        analise = df_historico.groupby(['h3_index', 'desc_periodo']).size().reset_index(name='crimes')
        analise['lat'] = analise['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        analise['lon'] = analise['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        analise['periodo_cat'] = analise['desc_periodo'].astype('category').cat.codes

        X = analise[['lat', 'lon', 'periodo_cat']]
        y = analise['crimes']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
        model_cat = CatBoostRegressor(n_estimators=100, learning_rate=0.05, depth=6, verbose=0, random_state=42)
        
        model_xgb.fit(X_train, y_train)
        model_cat.fit(X_train, y_train)

        pred_xgb_test = model_xgb.predict(X_test)
        pred_cat_test = model_cat.predict(X_test)
        pred_ensemble_test = (pred_xgb_test * 0.5) + (pred_cat_test * 0.5)

        mae = mean_absolute_error(y_test, pred_ensemble_test)
        rmse = np.sqrt(mean_squared_error(y_test, pred_ensemble_test))
        r2 = r2_score(y_test, pred_ensemble_test)

        metricas = {
            "MAE_Erro_Medio_Absoluto": round(mae, 2),
            "RMSE_Raiz_Erro_Quadratico": round(rmse, 2),
            "R2_Coeficiente_Determinacao": round(r2, 4),
            "Ultima_Atualizacao": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(self.ouro / "metricas_modelo.json", "w") as f:
            json.dump(metricas, f, indent=4)

        p_xgb = model_xgb.predict(X)
        p_cat = model_cat.predict(X)
        analise['score_bruto'] = (p_xgb * 0.5) + (p_cat * 0.5)
        
        analise['score_risco'] = ((analise['score_bruto'] - analise['score_bruto'].min()) / 
                                 (analise['score_bruto'].max() - analise['score_bruto'].min()) * 100).round(2)

        explainer = shap.TreeExplainer(model_xgb)
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
            folium.CircleMarker(
                location=[lat, lon], radius=5, color=cor, fill=True, fill_opacity=0.6,
                tooltip=f"H3: {row['h3_index']} | Risco: {score}%"
            ).add_to(mapa)
        mapa.save(str(self.ouro / "mapa_estrategico.html"))

    def executar_pipeline(self):
        try:
            self.preparar_ambiente()
            colecao_prata = []
            for ano in self.anos_historico:
                dados_brutos = self.ingerir_bronze(ano)
                if dados_brutos is not None:
                    colecao_prata.append(self.refinar_prata(dados_brutos, ano))
            if colecao_prata:
                df_total = pd.concat(colecao_prata, ignore_index=True)
                df_ouro = self.consolidar_ouro(df_total)
                self.renderizar_mapa(df_ouro)
                global CACHE_OURO_CONSOLIDADO
                CACHE_OURO_CONSOLIDADO = df_ouro.drop_duplicates(subset=['h3_index'])[['h3_index', 'score_risco']].set_index('h3_index').to_dict('index')
            
            notificar_discord(sucesso=True, mensagem=f"✅ Pipeline SafeDriver finalizado com sucesso! Processado até o ano: {self.ano_atual}")
        except Exception as e:
            notificar_discord(sucesso=False, mensagem=f"🚨 ERRO CRÍTICO no Pipeline SafeDriver: {str(e)}")
            raise e

@app.get("/")
def status():
    return {"status": "Operacional", "poligonos_h3": len(CACHE_OURO_CONSOLIDADO) if CACHE_OURO_CONSOLIDADO else 0}

@app.get("/risco/{lat}/{lon}")
def consultar_risco(lat: float, lon: float):
    if not CACHE_OURO_CONSOLIDADO:
        raise HTTPException(status_code=503)
    h3_idx = h3.latlng_to_cell(lat, lon, 9)
    info = CACHE_OURO_CONSOLIDADO.get(h3_idx, {"score_risco": 0.0})
    return {
        "h3_index": h3_idx,
        "score_risco": info['score_risco'],
        "classificacao": "SEGURO" if info['score_risco'] < 5 else "ALERTA"
    }

if __name__ == "__main__":
    ano = int(os.environ.get("ANO_PROCESSAMENTO", datetime.now().year))
    pipeline = AutobotPipeline(ano)
    pipeline.executar_pipeline()
    if os.getenv("GITHUB_ACTIONS") != "true":
        uvicorn.run(app, host="0.0.0.0", port=8000)
