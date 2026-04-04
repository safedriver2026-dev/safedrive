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
from geopy.geocoders import Nominatim
from lightgbm import LGBMRegressor

app = FastAPI(title="SafeDriver API")
CACHE_OURO_CONSOLIDADO = None

class AutobotPipeline:
    def __init__(self, ano_atual):
        self.ano_atual = ano_atual
        self.anos_historico = list(range(2022, self.ano_atual + 1))
        self.raiz = Path(".")
        self.bronze = self.raiz / "datalake" / "bronze"
        self.prata = self.raiz / "datalake" / "prata"
        self.ouro = self.raiz / "datalake" / "ouro"
        self.arquivo_cache_geo = self.raiz / "geo_cache.json"
        self.geolocator = Nominatim(user_agent="safedriver_bot_v6")
        self.colunas_verificacao = ['LATITUDE', 'LONGITUDE', 'RUBRICA', 'NOME_MUNICIPIO', 'BAIRRO', 'LOGRADOURO']

    def preparar_ambiente(self):
        for diretorio in [self.bronze, self.prata, self.ouro]:
            diretorio.mkdir(parents=True, exist_ok=True)

    def descobrir_cabecalho_robusto(self, caminho_arquivo):
        previa = pd.read_excel(caminho_arquivo, header=None, nrows=50)
        for i, linha in previa.iterrows():
            valores_linha = [str(v).strip().upper() for v in linha.values if pd.notna(v)]
            matches = len(set(self.colunas_verificacao).intersection(set(valores_linha)))
            if matches >= 4:
                return i
        return 0

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
            linha_header = self.descobrir_cabecalho_robusto(arquivo_temp)
            df = pd.read_excel(arquivo_temp, header=linha_header)
            df.columns = [str(c).strip().upper() for c in df.columns]
            colunas_existentes = [c for c in self.colunas_verificacao if c in df.columns]
            df_final = df[colunas_existentes].copy()
            df_final.to_parquet(arquivo_parquet, index=False)
            arquivo_temp.unlink()
            return df_final
        except:
            return None

    def carregar_cache(self):
        if self.arquivo_cache_geo.exists():
            with open(self.arquivo_cache_geo, 'r') as f:
                return json.load(f)
        return {}

    def salvar_cache(self, cache):
        with open(self.arquivo_cache_geo, 'w') as f:
            json.dump(cache, f)

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
        analise = df_historico.groupby('h3_index').size().reset_index(name='contagem_crimes')
        analise['lat'] = analise['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        analise['lon'] = analise['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        modelo = LGBMRegressor(n_estimators=100, learning_rate=0.08, verbose=-1)
        modelo.fit(analise[['lat', 'lon']], analise['contagem_crimes'])
        preds = modelo.predict(analise[['lat', 'lon']])
        analise['score_risco'] = ((preds - preds.min()) / (preds.max() - preds.min()) * 100).round(2)
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
    ano = int(os.environ.get("ANO_PROCESSAMENTO", 2026))
    pipeline = AutobotPipeline(ano)
    pipeline.executar_pipeline()
    if os.getenv("GITHUB_ACTIONS") != "true":
        uvicorn.run(app, host="0.0.0.0", port=8000)
