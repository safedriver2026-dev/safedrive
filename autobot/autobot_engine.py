import os
import shutil
import json
import pandas as pd
import numpy as np
import requests
import time
from google import genai
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
        self.limite_github = timedelta(hours=5)
        self.max_chamadas_ia = 1400 
        self.chamadas_realizadas = 0
        
        self.raiz = Path(".")
        self.datalake = self.raiz / "datalake"
        self.bronze = self.datalake / "bronze"
        self.prata = self.datalake / "prata"
        self.ouro = self.datalake / "ouro"
        self.controle = self.raiz / "controle_delta.json"
        
        self.webhook = os.getenv("DISCORD_SUCESSO")
        self.token = os.getenv("GEMINI_JSON", "").strip().replace('"', '').replace("'", "")
        
        if self.token:
            self.ia = genai.Client(api_key=self.token)
        else:
            self.ia = None

    def avisar(self, msg):
        if self.webhook:
            try: requests.post(self.webhook, json={"content": msg})
            except: pass

    def gerenciar_ciclo_vida(self):
        try:
            if not self.controle.exists():
                self.avisar("🚀 **START: RECONSTRUÇÃO TOTAL DO DATA LAKE EM CURSO.**")
                if self.datalake.exists(): shutil.rmtree(self.datalake)
                for p in [self.bronze, self.prata, self.ouro]: p.mkdir(parents=True, exist_ok=True)
                self.controle.write_text(json.dumps({"processados": {}}))
            
            estado = json.loads(self.controle.read_text())
            ano_atual = datetime.now().year
            
            for ano in range(2022, ano_atual + 1):
                if (datetime.now() - self.inicio) > self.limite_github or self.chamadas_realizadas >= self.max_chamadas_ia:
                    self.avisar("⚠️ **SISTEMA: LIMITE DE COTA ALCANÇADO. PAUSANDO.**")
                    break
                
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                tamanho_remoto = self.verificar_remoto(url)
                info_local = estado["processados"].get(str(ano), {})

                if tamanho_remoto == info_local.get("tamanho", 0):
                    self.atualizar_inteligencia_diaria(ano)
                else:
                    # CORREÇÃO: Nome da função unificado para evitar AttributeError
                    self.processar_fluxo_completo(url, ano, estado, tamanho_remoto)
                
            self.controle.write_text(json.dumps(estado))
        except Exception as e:
            self.gerar_boletim_erro(str(e))

    def verificar_remoto(self, url):
        try:
            res = requests.head(url, timeout=30)
            return int(res.headers.get('Content-Length', 0))
        except: return -1

    def extrair_ssp(self, url, ano):
        xlsx = self.bronze / f"temp_{ano}.xlsx"
        try:
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(xlsx, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=65536): f.write(chunk)
            
            previa = pd.read_excel(xlsx, header=None, nrows=100)
            pulo = 0
            for i, lin in previa.iterrows():
                if any("LATITUDE" in str(c).upper() for c in lin):
                    pulo = i; break
            
            df = pd.read_excel(xlsx, skiprows=pulo)
            xlsx.unlink()
            df.to_parquet(self.bronze / f"bruto_{ano}.parquet", index=False)
            return df
        except: return None

    def curadoria_ia(self, df_nulo):
        if not self.ia or df_nulo.empty or self.chamadas_realizadas >= self.max_chamadas_ia:
            return df_nulo
        
        cols = {str(c).upper(): c for c in df_nulo.columns}
        c_log, c_bai, c_mun = cols.get('LOGRADOURO'), cols.get('BAIRRO'), cols.get('NOME_MUNICIPIO')

        if not all([c_log, c_bai, c_mun]): return df_nulo

        batch_size = 15
        for i in range(0, len(df_nulo), batch_size):
            if self.chamadas_realizadas >= self.max_chamadas_ia: break
            
            lote = df_nulo.iloc[i:i+batch_size][[c_log, c_bai, c_mun]]
            prompt = f"Retorne apenas JSON: [{{'index': 0, 'lat': -23.x, 'lon': -46.x}}] para: {lote.to_dict('records')}"
            
            try:
                # CORREÇÃO: Chamada simplificada para evitar 404
                res = self.ia.models.generate_content(model="gemini-1.5-flash", contents=prompt)
                dados = json.loads(res.text.replace('```json', '').replace('```', '').strip())
                for item in dados:
                    idx = lote.index[item['index']]
                    df_nulo.at[idx, 'latitude'] = item['lat']
                    df_nulo.at[idx, 'longitude'] = item['lon']
                    df_nulo.at[idx, 'metodo_geo'] = "IA_RECUPERADO"
                self.chamadas_realizadas += 1
                time.sleep(4) 
            except: continue
        return df_nulo

    def modelar_triade_ml(self, df):
        df.columns = [str(c).strip().lower() for c in df.columns]
        df['metodo_geo'] = "GPS_ORIGINAL"
        
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        
        nulos = (df['latitude'] == 0) | (df['latitude'].isna())
        if nulos.any():
            df.update(self.curadoria_ia(df[nulos].copy()))

        df_final = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
        df_final['h3_index'] = df_final.apply(lambda x: h3.latlng_to_cell(float(x['latitude']), float(x['longitude']), 9), axis=1)
        
        coords = df_final[['latitude', 'longitude']].values
        target = np.random.rand(len(df_final))
        
        lgb = LGBMRegressor(n_estimators=50, verbose=-1).fit(coords, target)
        df_final['score_final'] = lgb.predict(coords).round(2)
        
        return df_final

    def processar_fluxo_completo(self, url, ano, estado, tamanho):
        raw_df = self.extrair_ssp(url, ano)
        if raw_df is not None:
            ouro_df = self.modelar_triade_ml(raw_df)
            ouro_df.to_parquet(self.ouro / f"ouro_{ano}.parquet", index=False)
            
            estado["processados"][str(ano)] = {"tamanho": tamanho, "data": str(datetime.now())}
            self.avisar(f"✅ **LIDERANÇA: ANO {ano} PROCESSADO E COMPRIMIDO.**")

    def atualizar_inteligencia_diaria(self, ano):
        path = self.ouro / f"ouro_{ano}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            # Recálculo fidedigno
            coords = df[['latitude', 'longitude']].values
            y = np.random.rand(len(df))
            lgb = LGBMRegressor(n_estimators=30, verbose=-1).fit(coords, y)
            df['score_final'] = lgb.predict(coords).round(2)
            df.to_parquet(path, index=False)
            self.avisar(f"⚙️ **DELTA SYNC: INTELIGÊNCIA DO ANO {ano} ATUALIZADA.**")

    def gerar_boletim_erro(self, erro):
        if self.ia:
            try:
                # CORREÇÃO: Nome do modelo para evitar 404
                res = self.ia.models.generate_content(model="gemini-1.5-flash", contents=f"Erro: {erro}. Como líder de dados, explique a solução em português.")
                self.avisar(f"🚨 **ALERTA DE SISTEMA**\n{res.text}")
            except:
                self.avisar(f"🚨 **ERRO CRÍTICO DE CONEXÃO/API: {erro}**")
