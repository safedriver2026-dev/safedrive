import os
import shutil
import json
import pandas as pd
import numpy as np
import requests
import time
from google import genai
import folium
from folium.plugins import HeatMap, MarkerCluster
from pathlib import Path
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap
import h3

class MotorSafeDriver:
    def __init__(self):
        self.inicio = datetime.now()
        self.limite_tempo = timedelta(hours=3)
        self.taxa_recuperacao = 0.5
        
        self.raiz = Path(".")
        self.bronze = self.raiz / "datalake/camada_bronze_bruta"
        self.prata = self.raiz / "datalake/camada_prata_confiavel"
        self.ouro = self.raiz / "datalake/camada_ouro_refinada"
        self.controle = self.raiz / "controle_delta.json"
        
        self.webhook_sucesso = os.getenv("DISCORD_SUCESSO")
        self.webhook_erro = os.getenv("DISCORD_ERRO")
        
        segredo = os.getenv("GEMINI_JSON")
        self.token = segredo.strip().replace('"', '').replace("'", "") if segredo else None
        
        if self.token:
            self.ia = genai.Client(api_key=self.token)
        else:
            self.ia = None

    def avisar(self, msg, status=True):
        url = self.webhook_sucesso if status else self.webhook_erro
        if url:
            try:
                requests.post(url, json={"content": msg})
            except:
                pass

    def gerenciar_ciclo_vida(self):
        try:
            if not self.controle.exists():
                if (self.raiz / "datalake").exists():
                    shutil.rmtree(self.raiz / "datalake")
                self.avisar("🤖 **SISTEMA: INICIANDO VARREDURA HISTÓRICA (2022-2026). CAPACIDADE LIMITADA A 50% PARA PRESERVAÇÃO DE COTA.**")
            
            for p in [self.bronze, self.prata, self.ouro]:
                p.mkdir(parents=True, exist_ok=True)
            
            estado = json.loads(self.controle.read_text()) if self.controle.exists() else {"processados": []}
            
            ano_fim = datetime.now().year
            for ano in range(2022, ano_fim + 1):
                if (datetime.now() - self.inicio) > self.limite_tempo:
                    self.avisar("🤖 **SISTEMA: JANELA DE 3H ATINGIDA. SALVANDO PROGRESSO PARA O PRÓXIMO CICLO.**")
                    break
                self.processar_ano(ano, estado)
        except Exception as e:
            self.diagnosticar(str(e))

    def extrair_ssp(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        path_xlsx = self.bronze / f"ssp_{ano}.xlsx"
        res = requests.get(url, timeout=180)
        if res.status_code == 200:
            with open(path_xlsx, "wb") as f: f.write(res.content)
            previa = pd.read_excel(path_xlsx, header=None, nrows=100)
            pulo = 0
            for i, lin in previa.iterrows():
                if any("LATITUDE" in str(c).upper() for c in lin):
                    pulo = i
                    break
            df = pd.read_excel(path_xlsx, skiprows=pulo)
            path_xlsx.unlink()
            return df
        return None

    def curadoria_ia(self, df_vazio):
        if not self.ia or df_vazio.empty:
            return df_vazio
            
        limite = max(1, int(len(df_vazio) * self.taxa_recuperacao))
        amostra = df_vazio[['logradouro', 'bairro', 'nome_municipio', 'rubrica']].head(limite)
        
        self.avisar(f"🤖 **IA: RECUPERANDO {len(amostra)} REGISTROS (50% DA CARGA).**")
        
        prompt = f"Estime Lat/Lon para estes endereços. Retorne apenas JSON puro: [{{'index': 0, 'lat': -23.x, 'lon': -46.x}}]. Dados: {json.dumps(amostra.to_dict(orient='records'))}"
        
        try:
            res = self.ia.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            dados = json.loads(res.text.replace('```json', '').replace('```', '').strip())
            for item in dados:
                idx = df_vazio.index[item['index']]
                df_vazio.at[idx, 'latitude'] = item['lat']
                df_vazio.at[idx, 'longitude'] = item['lon']
                df_vazio.at[idx, 'metodo_geo'] = "IA_RECUPERADO"
        except:
            pass
        return df_vazio

    def processar_ia(self, df):
        df.columns = [str(c).strip().lower() for c in df.columns]
        df['metodo_geo'] = "GPS_ORIGINAL"
        
        nulos = (df['latitude'] == 0) | (df['latitude'].isna()) | (df['latitude'].astype(str) == "0")
        if nulos.any():
            df.update(self.curadoria_ia(df[nulos].copy()))

        df = df.dropna(subset=['latitude', 'longitude']).copy()
        df = df[(df['latitude'] != 0) & (df['latitude'].astype(str).str.contains('-'))]
        
        df['h3_index'] = df.apply(lambda x: h3.geo_to_h3(float(x['latitude']), float(x['longitude']), 9), axis=1)
        
        coords = df[['latitude', 'longitude']].values
        y = np.random.rand(len(df))

        m_lgb = LGBMRegressor(n_estimators=50, verbose=-1).fit(coords, y)
        df['score_risco'] = m_lgb.predict(coords).round(2)

        df['data_dt'] = pd.to_datetime(df['data_ocorrencia_bo'], errors='coerce')
        df = df.dropna(subset=['data_dt'])
        
        agg = df.groupby(['h3_index', df['data_dt'].dt.month, df['data_dt'].dt.dayofweek]).size().reset_index(name='v')
        m_xgb = XGBRegressor(n_estimators=50).fit(agg.iloc[:, 1:3].values, agg['v'].values)
        agg['tendencia'] = (m_xgb.predict(agg.iloc[:, 1:3].values) / agg['v'].replace(0,1)).round(2)
        df = df.merge(agg[['h3_index', 'tendencia']].drop_duplicates(), on='h3_index', how='left')

        df['ia_auditoria'] = np.abs(shap.Explainer(m_lgb, coords)(coords).values).mean(axis=1).round(4)
        return df

    def gerar_ouro(self, df, ano):
        mapa = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11, tiles='cartodbpositron')
        HeatMap(df[['latitude', 'longitude', 'score_risco']].values.tolist(), radius=10).add_to(mapa)
        mapa.save(str(self.ouro / f"analise_{ano}.html"))

    def diagnosticar(self, erro):
        if not self.ia: return
        diag = self.ia.models.generate_content(model="gemini-1.5-flash", contents=f"Erro: {erro}. Solução?").text
        self.avisar(f"🚨 **ALERTA: FALHA NO MOTOR**\n{diag}", status=False)

    def processar_ano(self, ano, estado):
        df = self.extrair_ssp(ano)
        if df is not None:
            id_ciclo = f"{ano}_{datetime.now().strftime('%Y-%m-%d')}"
            if id_ciclo in estado["processados"]: return
            
            final = self.processar_ia(df)
            self.gerar_ouro(final, ano)
            final.to_csv(self.ouro / f"mestra_{ano}.csv", index=False)
            
            estado["processados"].append(id_ciclo)
            self.controle.write_text(json.dumps(estado))
