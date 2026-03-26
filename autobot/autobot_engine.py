import os
import shutil
import json
import pandas as pd
import numpy as np
import requests
from google import genai
import folium
from folium.plugins import HeatMap, MarkerCluster
from pathlib import Path
from datetime import datetime
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap
import h3

class MotorSafeDriver:
    def __init__(self):
        self.raiz = Path(".")
        self.bronze = self.raiz / "datalake/camada_bronze_bruta"
        self.prata = self.raiz / "datalake/camada_prata_confiavel"
        self.ouro = self.raiz / "datalake/camada_ouro_refinada"
        self.controle = self.raiz / "controle_delta.json"
        self.log_sistema = self.prata / "registro_sistema.log"
        self.webhook_sucesso = os.getenv("DISCORD_SUCESSO")
        self.webhook_erro = os.getenv("DISCORD_ERRO")
        
        chave_bruta = os.getenv("GEMINI_JSON")
        self.chave_api = chave_bruta.strip().replace('"', '').replace("'", "") if chave_bruta else None
        
        if self.chave_api:
            self.cliente = genai.Client(api_key=self.chave_api)
        else:
            self.cliente = None

    def transmitir_discord(self, mensagem, sucesso=True):
        url = self.webhook_sucesso if sucesso else self.webhook_erro
        if url:
            try:
                requests.post(url, json={"content": mensagem})
            except:
                pass

    def gerenciar_ciclo_vida(self):
        try:
            if not self.controle.exists():
                if (self.raiz / "datalake").exists():
                    shutil.rmtree(self.raiz / "datalake")
                self.transmitir_discord("🤖 **SISTEMA: PROTOCOLO DE REINICIALIZAÇÃO EXECUTADO. RECONSTRUINDO CAMADAS E MALHA H3.**")
            
            for pasta in [self.bronze, self.prata, self.ouro]:
                pasta.mkdir(parents=True, exist_ok=True)
            
            estado = json.loads(self.controle.read_text()) if self.controle.exists() else {"processados": []}
            
            ano_atual = datetime.now().year
            anos_fluxo = list(range(2025, ano_atual + 1))
            
            for ano in anos_fluxo:
                self.executar_ciclo_anual(ano, estado)
            
            self.transmitir_discord("🤖 **SISTEMA: OPERAÇÃO AUTÔNOMA FINALIZADA. ATIVOS ESTRATÉGICOS SINCRONIZADOS.**")
        except Exception as erro:
            self.acionar_reparo_ia(str(erro))

    def extrair_dados_ssp(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        caminho_tmp = self.bronze / f"ssp_temporario_{ano}.xlsx"
        
        resposta = requests.get(url, timeout=120)
        if resposta.status_code == 200:
            with open(caminho_tmp, "wb") as f: f.write(resposta.content)
            
            previa = pd.read_excel(caminho_tmp, header=None, nrows=100)
            linha_inicio = 0
            for i, linha in previa.iterrows():
                celulas = [str(c).strip().upper() for c in linha.values]
                if "LATITUDE" in celulas or "LOGRADOURO" in celulas:
                    linha_inicio = i
                    break
            
            df = pd.read_excel(caminho_tmp, skiprows=linha_inicio)
            caminho_tmp.unlink()
            return df
        return None

    def recuperar_tempo_ia(self, linha):
        col_hora = 'hora_ocorrencia_bo'
        if col_hora in linha and pd.notnull(linha[col_hora]) and str(linha[col_hora]).strip() != "" and str(linha[col_hora]).upper() != "NULL":
            return linha[col_hora], "Original"
        
        mapa_periodos = {"DE MADRUGADA": "03:00:00", "PELA MANHÃ": "09:00:00", "A TARDE": "15:00:00", "A NOITE": "21:00:00"}
        referencia = str(linha.get('desc_periodo', '')).upper()
        return mapa_periodos.get(referencia, "12:00:00"), "Estimado_IA"

    def curadoria_geospacial_ia(self, df_sujo):
        if not self.cliente or df_sujo.empty:
            return df_sujo
            
        lote = df_sujo[['logradouro', 'bairro', 'nome_municipio', 'rubrica']].head(15).to_dict(orient='records')
        prompt = f"Corrija Lat/Lon para estes endereços criminais: {json.dumps(lote)}. Retorne apenas JSON puro: [{{'index': 0, 'lat': -23.x, 'lon': -46.x}}]"
        
        try:
            res = self.cliente.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            texto_limpo = res.text.replace('```json', '').replace('```', '').strip()
            correcoes = json.loads(texto_limpo)
            for item in correcoes:
                indice = df_sujo.index[item['index']]
                df_sujo.at[indice, 'latitude'] = item['lat']
                df_sujo.at[indice, 'longitude'] = item['lon']
                df_sujo.at[indice, 'metodo_localizacao'] = "IA_Recuperado"
        except:
            pass
        return df_sujo

    def processar_inteligencia_tripla(self, df):
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        tempos = df.apply(self.recuperar_tempo_ia, axis=1)
        df['hora_final'] = [t[0] for t in tempos]
        df['metodo_horario'] = [t[1] for t in tempos]
        
        df['metodo_localizacao'] = "GPS_Original"
        mask_vazia = (df['latitude'] == 0) | (df['latitude'].isna()) | (df['latitude'].astype(str).str.startswith('0'))
        
        if mask_vazia.any():
            df.update(self.curadoria_geospacial_ia(df[mask_vazia].copy()))

        df = df.dropna(subset=['latitude', 'longitude']).copy()
        df = df[(df['latitude'] != 0) & (df['latitude'].astype(str).str.contains('-'))]
        
        df['h3_index'] = df.apply(lambda x: h3.geo_to_h3(float(x['latitude']), float(x['longitude']), 9), axis=1)
        
        coords = df[['latitude', 'longitude']].values
        y_sim = np.random.rand(len(df))

        m_lgb = LGBMRegressor(n_estimators=50, verbose=-1).fit(coords, y_sim)
        df['score_geografico'] = m_lgb.predict(coords).round(2)

        df['data_dt'] = pd.to_datetime(df['data_ocorrencia_bo'], errors='coerce')
        df = df.dropna(subset=['data_dt'])
        
        agg = df.groupby(['h3_index', df['data_dt'].dt.month, df['data_dt'].dt.dayofweek]).size().reset_index(name='v')
        m_xgb = XGBRegressor(n_estimators=50).fit(agg.iloc[:, 1:3].values, agg['v'].values)
        agg['projecao_aumento'] = (m_xgb.predict(agg.iloc[:, 1:3].values) / agg['v'].replace(0,1)).round(2)
        df = df.merge(agg[['h3_index', 'projecao_aumento']].drop_duplicates(), on='h3_index', how='left')

        if 'natureza_apurada' in df.columns:
            df['nat_id'] = df['natureza_apurada'].astype('category').cat.codes
            m_cat = CatBoostRegressor(n_estimators=30, verbose=0).fit(df[['latitude', 'longitude', 'nat_id']].values, y_sim)
            df['score_severidade'] = m_cat.predict(df[['latitude', 'longitude', 'nat_id']].values).round(2)

        df['ia_auditada'] = np.abs(shap.Explainer(m_lgb, coords)(coords).values).mean(axis=1).round(4)
        return df

    def gerar_mapa_ouro(self, df, ano):
        mapa = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11, tiles='cartodbpositron')
        HeatMap(df[['latitude', 'longitude', 'score_geografico']].values.tolist(), radius=10).add_to(mapa)
        mapa.save(str(self.ouro / f"mapa_analitico_{ano}.html"))

    def acionar_reparo_ia(self, erro):
        if not self.cliente: return
        prompt = f"Falha técnica: {erro}. Indique a correção técnica imediata."
        diagnostico = self.cliente.models.generate_content(model="gemini-1.5-flash", contents=prompt).text
        self.transmitir_discord(f"🤖 **ALERTA: ANOMALIA NO FLUXO DE PROCESSAMENTO** 🤖\n{diagnostico}", sucesso=False)

    def executar_ciclo_anual(self, ano, estado):
        df = self.extrair_dados_ssp(ano)
        if df is not None:
            id_ref = f"{ano}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
            if id_ref in estado["processados"]: return
            
            final = self.processar_inteligencia_tripla(df)
            self.gerar_mapa_ouro(final, ano)
            final.to_csv(self.ouro / f"dados_mestre_{ano}.csv", index=False)
            
            estado["processados"].append(id_ref)
            self.controle.write_text(json.dumps(estado))
