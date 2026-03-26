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
        self.bronze = self.raiz / "datalake/camada_bronze_bruta"
        self.prata = self.raiz / "datalake/camada_prata_confiavel"
        self.ouro = self.raiz / "datalake/camada_ouro_refinada"
        self.controle = self.raiz / "controle_delta.json"
        
        self.webhook_sucesso = os.getenv("DISCORD_SUCESSO")
        self.webhook_erro = os.getenv("DISCORD_ERRO")
        
        raw_key = os.getenv("GEMINI_JSON")
        self.token = raw_key.strip().replace('"', '').replace("'", "") if raw_key else None
        
        if self.token:
            self.cliente = genai.Client(api_key=self.token)
        else:
            self.cliente = None

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
                self.avisar("🧹 **LIDERANÇA: INICIANDO LIMPEZA ESTRUTURAL E DOWNLOAD DA BASE 2022-2026.**")
                if (self.raiz / "datalake").exists():
                    shutil.rmtree(self.raiz / "datalake")
                self.controle.write_text(json.dumps({"processados": []}))
            
            for p in [self.bronze, self.prata, self.ouro]:
                p.mkdir(parents=True, exist_ok=True)
            
            estado = json.loads(self.controle.read_text())
            ano_fim = datetime.now().year
            
            for ano in range(2022, ano_fim + 1):
                if (datetime.now() - self.inicio) > self.limite_operacional:
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
                
                # VARREDURA AGRESSIVA: Procura dados em qualquer parte das primeiras 100 linhas
                df_completo = pd.read_excel(path_xlsx, header=None, nrows=100)
                pulo = 0
                for i, lin in df_completo.iterrows():
                    if any("LATITUDE" in str(c).upper() for c in lin):
                        pulo = i
                        break
                
                df = pd.read_excel(path_xlsx, skiprows=pulo)
                path_xlsx.unlink()
                return df
            except:
                time.sleep(30)
        return None

    def curadoria_ia(self, df_vazio):
        if not self.cliente or df_vazio.empty:
            return df_vazio
            
        # Garante que processamos pelo menos alguns dados para não ficar parado
        limite = max(5, int(len(df_vazio) * self.fator_capacidade))
        cols = {str(c).upper().strip(): c for c in df_vazio.columns}
        
        # Mapeamento flexível para evitar KeyError: [nan]
        c_log = cols.get('LOGRADOURO', df_vazio.columns[13] if len(df_vazio.columns) > 13 else None)
        c_bai = cols.get('BAIRRO', df_vazio.columns[12] if len(df_vazio.columns) > 12 else None)
        c_mun = cols.get('NOME_MUNICIPIO', df_vazio.columns[3] if len(df_vazio.columns) > 3 else None)

        amostra = df_vazio[[c_log, c_bai, c_mun]].head(limite)
        prompt = f"Converta endereços em Lat/Lon. Retorne apenas JSON: [{{'index': 0, 'lat': -23.x, 'lon': -46.x}}]. Dados: {json.dumps(amostra.to_dict(orient='records'))}"
        
        try:
            res = self.cliente.models.generate_content(model="gemini-1.5-flash", contents=prompt)
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
        
        # Converte coordenadas para numérico forçado para evitar erros de cálculo
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)

        nulos = (df['latitude'] == 0) | (df['latitude'].isna())
        if nulos.any():
            df.update(self.curadoria_ia(df[nulos].copy()))

        # Filtragem rigorosa para garantir fidelidade H3
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
        
        # H3 V4 latlng_to_cell
        df['h3_index'] = df.apply(lambda x: h3.latlng_to_cell(float(x['latitude']), float(x['longitude']), 9), axis=1)
        
        coords = df[['latitude', 'longitude']].values
        y = np.random.rand(len(df)) # Alvo simulado para score de risco

        m_lgb = LGBMRegressor(n_estimators=50, verbose=-1).fit(coords, y)
        df['score_risco'] = m_lgb.predict(coords).round(2)

        df['ia_auditoria'] = np.abs(shap.Explainer(m_lgb, coords)(coords).values).mean(axis=1).round(4)
        return df

    def diagnosticar(self, erro):
        if not self.cliente: return
        try:
            diag = self.cliente.models.generate_content(model="gemini-1.5-flash", contents=f"Erro técnico: {erro}. Como líder de dados, qual a correção rápida?").text
            self.avisar(f"🚨 **ALERTA: FALHA NO MOTOR** 🚨\n{diag}", status=False)
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
            self.avisar(f"✅ **LIDERANÇA: ANO {ano} PROCESSADO COM SUCESSO.**")
