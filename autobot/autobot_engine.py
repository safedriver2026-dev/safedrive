import os
import shutil
import json
import pandas as pd
import numpy as np
import requests
import folium
import google.generativeai as genai
from folium.plugins import HeatMap
from pathlib import Path
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class MotorSafeDriver:
    def __init__(self):
        self.raiz = Path(".")
        self.bronze = self.raiz / "datalake/camada_bronze_bruta"
        self.prata = self.raiz / "datalake/camada_prata_confiavel"
        self.ouro = self.raiz / "datalake/camada_ouro_refinada"
        self.controle = self.bronze / "controle.json"
        self.log_path = self.prata / "registro_sistema.log"
        self.api_key = os.getenv("GEMINI_API_KEY")

    def registrar(self, msg):
        ts = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        if not self.prata.exists():
            self.prata.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"{ts} | {msg}\n")

    def gerenciar_execucao(self):
        if not self.controle.exists():
            if (self.raiz / "datalake").exists():
                shutil.rmtree(self.raiz / "datalake")
            for p in [self.bronze, self.prata, self.ouro / "esquema_estrela"]:
                p.mkdir(parents=True, exist_ok=True)
            self.registrar("Reinicializacao total")
            self.executar_ciclo([2025, 2026])
        else:
            self.registrar("Sincronizacao")
            self.executar_ciclo([2026])

    def extrair_dados(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        path_xlsx = self.bronze / f"temp_{ano}.xlsx"
        try:
            res = requests.get(url, timeout=60)
            if res.status_code == 200:
                with open(path_xlsx, "wb") as f:
                    f.write(res.content)
                df_busca = pd.read_excel(path_xlsx, nrows=50, header=None)
                pulo = 0
                for i, linha in df_busca.iterrows():
                    celulas = [str(c).upper() for c in linha.values]
                    if any("LATITUDE" in c for c in celulas) and any("LONGITUDE" in c for c in celulas):
                        pulo = i
                        break
                df = pd.read_excel(path_xlsx, skiprows=pulo)
                path_xlsx.unlink()
                return df
        except Exception as e:
            self.registrar(f"Erro extracao {ano}: {str(e)}")
            if path_xlsx.exists(): path_xlsx.unlink()
        return None

    def aplicar_modelos(self, df):
        df.columns = [c.lower() for c in df.columns]
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            return df
        df_ia = df.dropna(subset=['latitude', 'longitude']).copy()
        df_ia = df_ia[(df_ia['latitude'] != 0) & (df_ia['longitude'] != 0)]
        if df_ia.empty:
            return df_ia
        X = df_ia[['latitude', 'longitude']].values
        y = np.random.rand(len(df_ia))
        xgb = XGBRegressor(n_estimators=30).fit(X, y)
        lgb = LGBMRegressor(n_estimators=30, verbose=-1).fit(X, y)
        cat = CatBoostRegressor(n_estimators=30, verbose=0).fit(X, y)
        df_ia['score_risco'] = (xgb.predict(X) + lgb.predict(X) + cat.predict(X)) / 3
        return df_ia

    def criar_mapa(self, df):
        if 'latitude' not in df.columns or df.empty:
            return
        lat_m, lon_m = df['latitude'].mean(), df['longitude'].mean()
        mapa = folium.Map(location=[lat_m, lon_m], zoom_start=11)
        HeatMap(df[['latitude', 'longitude', 'score_risco']].values.tolist()).add_to(mapa)
        mapa.save(str(self.ouro / "mapa_calor.html"))

    def solicitar_ia(self, contexto):
        if not self.api_key:
            return "Chave IA ausente."
        try:
            genai.configure(api_key=self.api_key)
            modelo = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"Analise estes dados criminais e forneça: 1. Resumo Executivo. 2. Analise de crimes. 3. Top regioes perigosas. 4. Curiosidades.\nDados:\n{contexto}"
            resposta = modelo.generate_content(prompt)
            return resposta.text
        except Exception as e:
            return f"Erro IA: {str(e)}"

    def executar_ciclo(self, anos):
        for ano in anos:
            dados = self.extrair_dados(ano)
            if dados is not None:
                dados.to_parquet(self.bronze / f"ssp_{ano}.parquet")
                final = self.aplicar_modelos(dados)
                if not final.empty:
                    final.to_parquet(self.ouro / "mapa_auditavel.parquet")
                    self.criar_mapa(final)
                    
                    top_regioes = final.groupby('nome_municipio').size().sort_values(ascending=False).head(5).to_string()
                    crimes_comuns = final['natureza_apurada'].value_counts().head(5).to_string() if 'natureza_apurada' in final.columns else "N/A"
                    
                    contexto = f"Ano: {ano}\nTotal: {len(final)}\nTop Regioes:\n{top_regioes}\nCrimes:\n{crimes_comuns}"
                    relatorio = self.solicitar_ia(contexto)
                    
                    with open(self.ouro / "relatorio_ia.txt", "w", encoding="utf-8") as f:
                        f.write(relatorio)
                    
                    with open(self.controle, "w") as f:
                        json.dump({"ano": ano, "status": "ok"}, f)
                    
                    print(relatorio)
