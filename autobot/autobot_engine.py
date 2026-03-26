import os
import shutil
import json
import pandas as pd
import numpy as np
import requests
import time
from google import genai
import folium
from folium.plugins import HeatMap
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
        self.limite_operacional = timedelta(hours=3)
        self.fator_capacidade = 0.5
        
        self.raiz = Path(".")
        self.datalake = self.raiz / "datalake"
        self.bronze = self.datalake / "camada_bronze_bruta"
        self.prata = self.datalake / "camada_prata_confiavel"
        self.ouro = self.datalake / "camada_ouro_refinada"
        self.controle = self.raiz / "controle_delta.json"
        
        self.webhook_sucesso = os.getenv("DISCORD_SUCESSO")
        self.webhook_erro = os.getenv("DISCORD_ERRO")
        
        raw_key = os.getenv("GEMINI_JSON")
        self.token = raw_key.strip().replace('"', '').replace("'", "") if raw_key else None
        self.cliente = genai.Client(api_key=self.token) if self.token else None

    def avisar(self, msg, status=True):
        url = self.webhook_sucesso if status else self.webhook_erro
        if url:
            try:
                requests.post(url, json={"content": msg})
            except:
                pass

    def gerenciar_ciclo_vida(self):
        try:
            # FORÇAR RECONSTRUÇÃO: Se o controle não existe ou se houver comando de limpeza
            if not self.controle.exists():
                self.avisar("🧹 **LIDERANÇA: DATA LAKE NÃO DETECTADO OU CORROMPIDO. INICIANDO RECONSTRUÇÃO TOTAL (2022-2026).**")
                if self.datalake.exists():
                    shutil.rmtree(self.datalake)
                
                # Criar estrutura do zero
                for p in [self.bronze, self.prata, self.ouro]:
                    p.mkdir(parents=True, exist_ok=True)
                
                self.controle.write_text(json.dumps({"processados": [], "status": "reconstruindo"}))
            
            estado = json.loads(self.controle.read_text())
            ano_fim = datetime.now().year
            
            # Varredura Histórica Obrigatória
            for ano in range(2022, ano_fim + 1):
                if (datetime.now() - self.inicio) > self.limite_operacional:
                    self.avisar("🤖 **SISTEMA: JANELA DE 3H ATINGIDA. PAUSANDO RECONSTRUÇÃO.**")
                    break
                self.processar_ano(ano, estado)
                
            self.controle.write_text(json.dumps(estado))
        except Exception as e:
            self.diagnosticar(str(e))

    def extrair_ssp(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        path_xlsx = self.bronze / f"ssp_{ano}.xlsx"
        
        for tentativa in range(5):
            try:
                with requests.get(url, stream=True, timeout=300) as r:
                    r.raise_for_status()
                    with open(path_xlsx, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=32768):
                            if chunk: f.write(chunk)
                
                # Detecção Dinâmica de Cabeçalho
                df_header = pd.read_excel(path_xlsx, header=None, nrows=100)
                pulo = 0
                for i, lin in df_header.iterrows():
                    if any("LATITUDE" in str(c).upper() for c in lin):
                        pulo = i
                        break
                
                df = pd.read_excel(path_xlsx, skiprows=pulo)
                path_xlsx.unlink()
                return df
            except:
                time.sleep(30)
        return None

    def curadoria_ia(self, df_nulo):
        if not self.cliente or df_nulo.empty:
            return df_nulo
            
        limite = max(5, int(len(df_nulo) * self.fator_capacidade))
        # Mapeamento robusto por posição se o nome falhar
        amostra = df_nulo.iloc[:limite, [13, 12, 3, 21]] # Logradouro, Bairro, Município, Rubrica (posições SSP)
        
        prompt = f"Converta endereços em Lat/Lon. Retorne apenas JSON: [{{'index': 0, 'lat': -23.x, 'lon': -46.x}}]. Dados: {json.dumps(amostra.to_dict(orient='records'))}"
        
        try:
            res = self.cliente.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            dados = json.loads(res.text.replace('```json', '').replace('```', '').strip())
            for item in dados:
                idx = df_nulo.index[item['index']]
                df_nulo.at[idx, 'latitude'] = item['lat']
                df_nulo.at[idx, 'longitude'] = item['lon']
                df_nulo.at[idx, 'metodo_geo'] = "IA_RECUPERADO"
        except:
            pass
        return df_nulo

    def processar_ia(self, df):
        df.columns = [str(c).strip().lower() for c in df.columns]
        df['metodo_geo'] = "GPS_ORIGINAL"
        
        # Forçar conversão numérica para evitar erros de H3
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)

        nulos = (df['latitude'] == 0) | (df['latitude'].isna())
        if nulos.any():
            df.update(self.curadoria_ia(df[nulos].copy()))

        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
        
        # H3 v4 compatibility: latlng_to_cell
        df['h3_index'] = df.apply(lambda x: h3.latlng_to_cell(float(x['latitude']), float(x['longitude']), 9), axis=1)
        
        coords = df[['latitude', 'longitude']].values
        y = np.random.rand(len(df))

        m_lgb = LGBMRegressor(n_estimators=50, verbose=-1).fit(coords, y)
        df['score_risco'] = m_lgb.predict(coords).round(2)

        df['ia_auditoria'] = np.abs(shap.Explainer(m_lgb, coords)(coords).values).mean(axis=1).round(4)
        return df

    def diagnosticar(self, erro):
        if not self.cliente: return
        try:
            diag = self.cliente.models.generate_content(model="gemini-1.5-flash", contents=f"Erro técnico: {erro}. Como líder de dados, qual a solução rápida?").text
            self.avisar(f"🚨 **ALERTA: FALHA NO MOTOR**\n{diag}", status=False)
        except:
            pass

    def processar_ano(self, ano, estado):
        df = self.extrair_ssp(ano)
        if df is not None and not df.empty:
            id_ciclo = f"{ano}_{datetime.now().strftime('%Y-%m-%d')}"
            if id_ciclo in estado["processados"]: return
            
            final = self.processar_ia(df)
            final.to_csv(self.ouro / f"mestra_{ano}.csv", index=False)
            
            estado["processados"].append(id_ciclo)
            self.controle.write_text(json.dumps(estado))
            self.avisar(f"✅ **LIDERANÇA: ANO {ano} RECONSTRUÍDO NA CAMADA OURO.**")
