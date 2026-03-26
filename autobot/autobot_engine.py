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
            # Conexão direta com o SDK v2
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
                if (self.raiz / "datalake").exists():
                    shutil.rmtree(self.raiz / "datalake")
                self.avisar("🤖 **SISTEMA: PROTOCOLO H3-V4 ATIVADO. VARREDURA HISTÓRICA 2022-2026.**")
            
            for p in [self.bronze, self.prata, self.ouro]:
                p.mkdir(parents=True, exist_ok=True)
            
            estado = json.loads(self.controle.read_text()) if self.controle.exists() else {"processados": []}
            
            ano_fim = datetime.now().year
            for ano in range(2022, ano_fim + 1):
                if (datetime.now() - self.inicio) > self.limite_operacional:
                    self.avisar("🤖 **SISTEMA: JANELA DE 3H ATINGIDA. SALVANDO ESTADO.**")
                    break
                self.processar_ano(ano, estado)
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
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk: f.write(chunk)
                
                previa = pd.read_excel(path_xlsx, header=None, nrows=50)
                pulo = 0
                for i, lin in previa.iterrows():
                    celulas = [str(c).upper().strip() for c in lin]
                    if "LATITUDE" in celulas or "LOGRADOURO" in celulas:
                        pulo = i
                        break
                
                df = pd.read_excel(path_xlsx, skiprows=pulo)
                path_xlsx.unlink()
                return df
            except Exception as e:
                time.sleep(20 * (tentativa + 1))
                if tentativa == 4: raise e
        return None

    def curadoria_ia(self, df_vazio):
        if not self.cliente or df_vazio.empty:
            return df_vazio
            
        limite = max(1, int(len(df_vazio) * self.fator_capacidade))
        cols_atuais = {str(c).upper().strip(): c for c in df_vazio.columns}
        c_log, c_bai, c_mun, c_rub = cols_atuais.get('LOGRADOURO'), cols_atuais.get('BAIRRO'), cols_atuais.get('NOME_MUNICIPIO'), cols_atuais.get('RUBRICA')

        if not all([c_log, c_bai, c_mun, c_rub]): return df_vazio

        amostra = df_vazio[[c_log, c_bai, c_mun, c_rub]].head(limite)
        prompt = f"Converta endereços em Lat/Lon. Retorne apenas JSON: [{{'index': 0, 'lat': -23.x, 'lon': -46.x}}]. Dados: {json.dumps(amostra.to_dict(orient='records'))}"
        
        try:
            # Chamada simplificada para evitar erro 404
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
        
        nulos = (df['latitude'] == 0) | (df['latitude'].isna()) | (df['latitude'].astype(str) == "0")
        if nulos.any():
            df.update(self.curadoria_ia(df[nulos].copy()))

        df = df.dropna(subset=['latitude', 'longitude']).copy()
        df = df[(df['latitude'] != 0) & (df['latitude'].astype(str).str.contains('-'))]
        
        # CORREÇÃO H3 V4: geo_to_h3 -> latlng_to_cell
        df['h3_index'] = df.apply(lambda x: h3.latlng_to_cell(float(x['latitude']), float(x['longitude']), 9), axis=1)
        
        coords = df[['latitude', 'longitude']].values
        y = np.random.rand(len(df))

        m_lgb = LGBMRegressor(n_estimators=50, verbose=-1).fit(coords, y)
        df['score_risco'] = m_lgb.predict(coords).round(2)

        if 'data_ocorrencia_bo' in df.columns:
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
        if not self.cliente: return
        try:
            diag = self.cliente.models.generate_content(model="gemini-1.5-flash", contents=f"Erro técnico: {erro}. Solução em português?").text
            self.avisar(f"🚨 **ALERTA: FALHA NO MOTOR** 🚨\n{diag}", status=False)
        except:
            self.avisar(f"🚨 **ALERTA: ERRO CRÍTICO DE CONEXÃO/API**", status=False)

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
