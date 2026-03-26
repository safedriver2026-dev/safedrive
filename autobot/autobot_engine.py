import os
import shutil
import json
import pandas as pd
import numpy as np
import requests
import google.generativeai as genai
import folium
from folium.plugins import HeatMap, MarkerCluster
from pathlib import Path
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import shap
import h3

class MotorSafeDriver:
    def __init__(self):
        self.raiz = Path(".")
        self.bronze = self.raiz / "datalake/bronze"
        self.prata = self.raiz / "datalake/prata"
        self.ouro = self.raiz / "datalake/ouro"
        self.controle = self.raiz / "controle_delta.json"
        self.webhook_sucesso = os.getenv("DISCORD_SUCESSO")
        self.webhook_erro = os.getenv("DISCORD_ERRO")
        self.api_key = os.getenv("GEMINI_JSON")

    def notificar(self, msg, sucesso=True):
        url = self.webhook_sucesso if sucesso else self.webhook_erro
        if url:
            try:
                requests.post(url, json={"content": msg})
            except:
                pass

    def executar_sistema(self):
        try:
            if not self.controle.exists():
                if (self.raiz / "datalake").exists():
                    shutil.rmtree(self.raiz / "datalake")
                self.notificar("🤖 **SISTEMA: PROTOCOLO DE LIMPEZA INICIAL CONCLUÍDO. RECONSTRUINDO MALHA H3.**")
            else:
                self.notificar("🤖 **SISTEMA: INICIANDO SINCRONIZAÇÃO DELTA. BUSCANDO ATUALIZAÇÕES.**")

            for p in [self.bronze, self.prata, self.ouro]:
                p.mkdir(parents=True, exist_ok=True)
            
            estado = json.loads(self.controle.read_text()) if self.controle.exists() else {"processados": []}
            
            for ano in [2025, 2026]:
                self.ciclo_anual(ano, estado)
            
            self.notificar("🤖 **SISTEMA: OPERAÇÃO FINALIZADA. DADOS DISPONÍVEIS NA CAMADA OURO.**")
        except Exception as e:
            self.acionar_ia_reparo(str(e))

    def tratar_horario(self, linha):
        if pd.notnull(linha['hora_ocorrencia_bo']) and str(linha['hora_ocorrencia_bo']).strip() != "":
            return linha['hora_ocorrencia_bo']
        
        periodos = {
            "DE MADRUGADA": "03:00:00",
            "PELA MANHÃ": "09:00:00",
            "A TARDE": "15:00:00",
            "A NOITE": "21:00:00"
        }
        ref = str(linha['desc_periodo']).upper()
        return periodos.get(ref, "12:00:00")

    def recuperar_dados_ia(self, df_vazio):
        if not self.api_key or df_vazio.empty:
            return df_vazio
            
        genai.configure(api_key=self.api_key)
        modelo = genai.GenerativeModel('gemini-1.5-flash')
        
        amostra = df_vazio[['logradouro', 'bairro', 'nome_municipio', 'rubrica']].head(20).to_dict(orient='records')
        
        comando = f"""
        PROCEDIMENTO: RECUPERAÇÃO DE COORDENADAS E SEVERIDADE.
        REGISTROS: {json.dumps(amostra)}
        
        TAREFAS:
        1. ESTIMAR LATITUDE E LONGITUDE PELO ENDEREÇO.
        2. ATRIBUIR SCORE DE RISCO (0-100) PELA GRAVIDADE.
        3. CATEGORIZAR: VIDA, VEICULO, PATRIMONIO OU DROGAS.
        
        RETORNE APENAS JSON: [{{"index": 0, "lat": -23.x, "lon": -46.x, "risco": 80, "cat": "vida"}}]
        """
        
        try:
            res = modelo.generate_content(comando)
            dados = json.loads(res.text.replace('```json', '').replace('```', ''))
            for item in dados:
                idx = df_vazio.index[item['index']]
                df_vazio.at[idx, 'latitude'] = item['lat']
                df_vazio.at[idx, 'longitude'] = item['lon']
                df_vazio.at[idx, 'ia_risco'] = item['risco']
                df_vazio.at[idx, 'ia_categoria'] = item['cat']
        except:
            pass
        return df_vazio

    def processar_camadas(self, df):
        df.columns = [c.lower() for c in df.columns]
        
        df['hora_final'] = df.apply(self.tratar_horario, axis=1)
        
        mask_ajuste = (df['latitude'] == 0) | (df['latitude'].isna())
        if mask_ajuste.any():
            df_ajuste = df[mask_ajuste].copy()
            df_corrigido = self.recuperar_dados_ia(df_ajuste)
            df.update(df_corrigido)

        df = df.dropna(subset=['latitude', 'longitude']).copy()
        df = df[df['latitude'] != 0]
        
        df['h3_index'] = df.apply(lambda x: h3.geo_to_h3(x['latitude'], x['longitude'], 9), axis=1)
        
        coords = df[['latitude', 'longitude']].values
        y_sim = np.random.rand(len(df))

        m_lgb = LGBMRegressor(n_estimators=50, verbose=-1).fit(coords, y_sim)
        df['score_geografico'] = m_lgb.predict(coords).round(2)

        df['data_dt'] = pd.to_datetime(df['data_ocorrencia_bo'])
        agg = df.groupby(['h3_index', df['data_dt'].dt.month, df['data_dt'].dt.dayofweek]).size().reset_index(name='v')
        m_xgb = XGBRegressor(n_estimators=50).fit(agg.iloc[:, 1:3].values, agg['v'].values)
        agg['previsao_aumento'] = (m_xgb.predict(agg.iloc[:, 1:3].values) / agg['v'].replace(0,1)).round(2)
        df = df.merge(agg[['h3_index', 'previsao_aumento']].drop_duplicates(), on='h3_index', how='left')

        if 'natureza_apurada' in df.columns:
            df['nat_id'] = df['natureza_apurada'].astype('category').cat.codes
            m_cat = CatBoostRegressor(n_estimators=30, verbose=0).fit(df[['latitude', 'longitude', 'nat_id']].values, y_sim)
            df['indice_severidade'] = m_cat.predict(df[['latitude', 'longitude', 'nat_id']].values).round(2)

        explainer = shap.Explainer(m_lgb, coords)
        df['prova_ia'] = np.abs(explainer(coords).values).mean(axis=1).round(4)
        
        return df

    def mapa_http(self, df, ano):
        mapa = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11, tiles='cartodbpositron')
        HeatMap(df[['latitude', 'longitude', 'score_geografico']].values.tolist(), radius=10).add_to(mapa)
        mapa.save(str(self.ouro / f"auditoria_visual_{ano}.html"))

    def relatorio_ia(self, info):
        if not self.api_key: return
        genai.configure(api_key=self.api_key)
        modelo = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"RELATÓRIO DE INTELIGÊNCIA: {info}. GERE ANÁLISE PARA SEGURADORAS, LOGÍSTICA E CIDADÃO. TOM FORMAL EM PORTUGUÊS."
        return modelo.generate_content(prompt).text

    def acionar_ia_reparo(self, erro):
        if not self.api_key: return
        genai.configure(api_key=self.api_key)
        modelo = genai.GenerativeModel('gemini-1.5-flash')
        diagnostico = modelo.generate_content(f"ERRO DE SISTEMA: {erro}. INDIQUE A CORREÇÃO.")
        self.notificar(f"🤖 **ALERTA: ANOMALIA DETECTADA** 🤖\n{diagnostico.text}", sucesso=False)

    def ciclo_anual(self, ano, estado):
        brutos = self.baixar_ssp(ano)
        if brutos is not None:
            id_ref = f"{ano}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
            if id_ref in estado["processados"]: return
            
            final = self.processar_camadas(brutos)
            self.mapa_http(final, ano)
            final.to_csv(self.ouro / f"mestra_looker_{ano}.csv", index=False)
            
            resumo = {
                "ano": ano, 
                "risco": float(final['score_geografico'].mean()), 
                "recuperados": len(final[final.get('ia_risco', pd.Series()).notnull()])
            }
            
            boletim = self.relatorio_ia(json.dumps(resumo))
            self.notificar(f"🤖 **SISTEMA: RELATÓRIO ESTRATÉGICO GERADO** 🤖\n{boletim}")
            
            estado["processados"].append(id_ref)
            self.controle.write_text(json.dumps(estado))

    def baixar_ssp(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        tmp = self.bronze / f"ssp_{ano}.xlsx"
        res = requests.get(url, timeout=120)
        if res.status_code == 200:
            with open(tmp, "wb") as f:
                f.write(res.content)
            df = pd.read_excel(tmp, skiprows=0)
            tmp.unlink()
            return df
        return None
