import os
import json
import time
import requests
import pandas as pd
import numpy as np
import h3
import folium
import uvicorn
from fastapi import FastAPI, HTTPException
from pathlib import Path
from datetime import datetime
from geopy.geocoders import Nominatim
from lightgbm import LGBMRegressor

app = FastAPI(title="SafeDriver API")
BASE_OURO_CACHE = None

class AutobotPipeline:
    def __init__(self, ano_alvo):
        self.ano_alvo = ano_alvo
        self.inicio = datetime.now()
        self.raiz = Path(".")
        self.bronze = self.raiz / "datalake" / "bronze"
        self.prata = self.raiz / "datalake" / "prata"
        self.ouro = self.raiz / "datalake" / "ouro"
        self.arquivo_cache_geo = self.raiz / "geo_cache.json"
        self.geolocator = Nominatim(user_agent="safedriver_bot_v3")

    def preparar_ambiente(self):
        for diretorio in [self.bronze, self.prata, self.ouro]:
            diretorio.mkdir(parents=True, exist_ok=True)

    def ingerir_bronze(self):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{self.ano_alvo}.xlsx"
        arquivo_temp = self.bronze / f"download_{self.ano_alvo}.xlsx"
        arquivo_parquet = self.bronze / f"bruto_{self.ano_alvo}.parquet"

        if arquivo_parquet.exists():
            return pd.read_parquet(arquivo_parquet)

        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(arquivo_temp, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            colunas_alvo = ['NOME_MUNICIPIO', 'BAIRRO', 'LOGRADOURO', 'NUMERO_LOGRADOURO', 'LATITUDE', 'LONGITUDE', 'RUBRICA', 'DATA_OCORRENCIA_BO']
            df = pd.read_excel(arquivo_temp, usecols=lambda c: c in colunas_alvo)
            df.to_parquet(arquivo_parquet, index=False)
            arquivo_temp.unlink()
            return df
        except Exception:
            return None

    def carregar_cache(self):
        if self.arquivo_cache_geo.exists():
            with open(self.arquivo_cache_geo, 'r') as f:
                return json.load(f)
        return {}

    def salvar_cache(self, cache):
        with open(self.arquivo_cache_geo, 'w') as f:
            json.dump(cache, f)

    def refinar_prata(self, df_bronze):
        df = df_bronze.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        
        nulos = (df['latitude'] == 0) | (df['latitude'].isna())
        df_nulos = df[nulos].copy()
        
        if not df_nulos.empty:
            cache = self.carregar_cache()
            recuperados = 0
            limite_api = 100
            
            for idx, row in df_nulos.iterrows():
                if recuperados >= limite_api: break
                
                endereco = f"{row.get('logradouro', '')}, {row.get('numero_logradouro', '')}, {row.get('bairro', '')}, {row.get('nome_municipio', '')}, SP"
                
                if endereco in cache:
                    df.at[idx, 'latitude'] = cache[endereco]['lat']
                    df.at[idx, 'longitude'] = cache[endereco]['lon']
                else:
                    try:
                        time.sleep(1.5)
                        loc = self.geolocator.geocode(endereco, timeout=5)
                        if loc:
                            cache[endereco] = {'lat': loc.latitude, 'lon': loc.longitude}
                            df.at[idx, 'latitude'] = loc.latitude
                            df.at[idx, 'longitude'] = loc.longitude
                        recuperados += 1
                    except: continue
            
            self.salvar_cache(cache)

        arquivo_prata = self.prata / f"prata_{self.ano_alvo}.parquet"
        df_limpo = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
        df_limpo.to_parquet(arquivo_prata, index=False)
        return df_limpo

    def consolidar_ouro(self, df_prata):
        df_prata['h3_index'] = df_prata.apply(lambda x: h3.latlng_to_cell(float(x['latitude']), float(x['longitude']), 9), axis=1)
        
        dados = df_prata.groupby('h3_index').size().reset_index(name='crimes')
        dados['lat'] = dados['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        dados['lon'] = dados['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        
        X = dados[['lat', 'lon']]
        y = dados['crimes']
        
        modelo = LGBMRegressor(n_estimators=100, learning_rate=0.05, verbose=-1)
        modelo.fit(X, y)
        
        preds = modelo.predict(X)
        dados['score_risco'] = ((preds - preds.min()) / (preds.max() - preds.min()) * 100).round(2)
        
        df_final = df_prata.merge(dados[['h3_index', 'score_risco']], on='h3_index', how='left')
        
        arquivo_ouro = self.ouro / f"ouro_{self.ano_alvo}.parquet"
        df_final.to_parquet(arquivo_ouro, index=False)
        return df_final

    def renderizar_mapa(self, df_ouro):
        df_mapa = df_ouro.drop_duplicates(subset=['h3_index']).copy()
        mapa = folium.Map(location=[-23.5505, -46.6333], zoom_start=11, tiles='CartoDB dark_matter')
        grupo = folium.FeatureGroup(name="Risco Preditivo")

        for _, row in df_mapa.iterrows():
            score = row.get('score_risco', 0)
            cor = '#00FF00' if score < 5 else '#FF0000'
            
            try:
                lat, lon = h3.cell_to_latlng(row['h3_index'])
                folium.CircleMarker(
                    location=[lat, lon], radius=6, color=cor, fill=True, fill_opacity=0.7,
                    tooltip=f"H3: {row['h3_index']} | Risco: {score}%"
                ).add_to(grupo)
            except: continue
            
        grupo.add_to(mapa)
        caminho_mapa = self.ouro / "mapa_estrategico.html"
        mapa.save(str(caminho_mapa))

    def executar_pipeline(self):
        self.preparar_ambiente()
        df_bronze = self.ingerir_bronze()
        if df_bronze is not None:
            df_prata = self.refinar_prata(df_bronze)
            df_ouro = self.consolidar_ouro(df_prata)
            self.renderizar_mapa(df_ouro)
            
            global BASE_OURO_CACHE
            BASE_OURO_CACHE = df_ouro.drop_duplicates(subset=['h3_index'])[['h3_index', 'score_risco']].set_index('h3_index').to_dict('index')

@app.get("/")
def status_api():
    return {"status": "Online", "servico": "SafeDriver Risco Preditivo", "registros_ativos": len(BASE_OURO_CACHE) if BASE_OURO_CACHE else 0}

@app.get("/risco/{lat}/{lon}")
def consultar_risco(lat: float, lon: float):
    if not BASE_OURO_CACHE:
        raise HTTPException(status_code=503, detail="Banco Ouro não carregado.")
        
    try:
        h3_idx = h3.latlng_to_cell(lat, lon, 9)
        dados_hex = BASE_OURO_CACHE.get(h3_idx, {"score_risco": 0.0})
        status = "SEGURO" if dados_hex['score_risco'] < 5 else "ALERTA"
        
        return {
            "coordenadas": {"lat": lat, "lon": lon},
            "h3_index": h3_idx,
            "score_risco": dados_hex['score_risco'],
            "status": status
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    ano_atual = int(os.getenv("ANO_PROCESSAMENTO", 2026))
    bot = AutobotPipeline(ano_atual)
    bot.executar_pipeline()
    
    if os.getenv("GITHUB_ACTIONS") != "true":
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
