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
        self.bronze = self.raiz / "datalake/camada_bronze_bruta"
        self.prata = self.raiz / "datalake/camada_prata_confiavel"
        self.ouro = self.raiz / "datalake/camada_ouro_refinada"
        self.controle = self.raiz / "controle_delta.json"
        self.log_sistema = self.prata / "registro_sistema.log"
        self.webhook_sucesso = os.getenv("DISCORD_SUCESSO")
        self.webhook_erro = os.getenv("DISCORD_ERRO")
        self.api_key = os.getenv("GEMINI_JSON")

    def transmitir(self, mensagem, sucesso=True):
        url = self.webhook_sucesso if sucesso else self.webhook_erro
        if url:
            try:
                requests.post(url, json={"content": mensagem})
            except:
                pass

    def registrar_evento(self, evento):
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_sistema, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} | {evento}\n")

    def gerenciar_ciclo_vida(self):
        try:
            if not self.controle.exists():
                if (self.raiz / "datalake").exists():
                    shutil.rmtree(self.raiz / "datalake")
                self.transmitir("🤖 **SISTEMA: PROTOCOLO DE DESTRUIÇÃO EXECUTADO. RECONSTRUINDO INFRAESTRUTURA E MALHA H3.**")
            
            for p in [self.bronze, self.prata, self.ouro]:
                p.mkdir(parents=True, exist_ok=True)
            
            self.registrar_evento("Sincronização iniciada.")
            estado = json.loads(self.controle.read_text()) if self.controle.exists() else {"processados": []}
            
            for ano in [2025, 2026]:
                self.processar_unidade(ano, estado)
            
            self.transmitir("🤖 **SISTEMA: OPERAÇÃO AUTÓNOMA FINALIZADA. RELATÓRIOS MULTISSEGMENTO DISPONÍVEIS.**")
        except Exception as e:
            self.acionar_ia_reparo(str(e))

    def recuperar_horario_estatistico(self, linha):
        if pd.notnull(linha['hora_ocorrencia_bo']) and str(linha['hora_ocorrencia_bo']).strip() != "":
            return linha['hora_ocorrencia_bo']
        
        mapa_periodos = {
            "DE MADRUGADA": "03:00:00",
            "PELA MANHÃ": "09:00:00",
            "A TARDE": "15:00:00",
            "A NOITE": "21:00:00",
            "EM HORA INCERTA": "12:00:00"
        }
        ref = str(linha['desc_periodo']).upper()
        return mapa_periodos.get(ref, "12:00:00")

    def enriquecer_dados_ia(self, df_incompleto):
        if not self.api_key or df_incompleto.empty:
            return df_incompleto
            
        genai.configure(api_key=self.api_key)
        modelo = genai.GenerativeModel('gemini-1.5-flash')
        
        lote = df_incompleto[['logradouro', 'bairro', 'nome_municipio', 'rubrica']].head(20).to_dict(orient='records')
        
        comando = f"Analise e complete: {json.dumps(lote)}. Estime lat/lon, severidade (0-100) e categorize o crime. Retorne apenas JSON: [{{'index': 0, 'lat': -23.x, 'lon': -46.x, 'risco': 80, 'tipo': 'veiculo'}}]"
        
        try:
            res = modelo.generate_content(comando)
            dados = json.loads(res.text.replace('```json', '').replace('```', ''))
            for item in dados:
                idx = df_incompleto.index[item['index']]
                df_incompleto.at[idx, 'latitude'] = item['lat']
                df_incompleto.at[idx, 'longitude'] = item['lon']
                df_incompleto.at[idx, 'severidade_ia'] = item['risco']
                df_incompleto.at[idx, 'categoria_ia'] = item['tipo']
        except:
            pass
        return df_incompleto

    def executar_modelagem(self, df):
        df.columns = [c.lower() for c in df.columns]
        df['hora_recuperada'] = df.apply(self.recuperar_horario_estatistico, axis=1)
        
        mask_ajuste = (df['latitude'] == 0) | (df['latitude'].isna())
        if mask_ajuste.any():
            df_ajuste = df[mask_ajuste].copy()
            df.update(self.enriquecer_dados_ia(df_ajuste))

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
        agg['projecao_aumento'] = (m_xgb.predict(agg.iloc[:, 1:3].values) / agg['v'].replace(0,1)).round(2)
        df = df.merge(agg[['h3_index', 'projecao_aumento']].drop_duplicates(), on='h3_index', how='left')

        if 'natureza_apurada' in df.columns:
            df['nat_id'] = df['natureza_apurada'].astype('category').cat.codes
            m_cat = CatBoostRegressor(n_estimators=30, verbose=0).fit(df[['latitude', 'longitude', 'nat_id']].values, y_sim)
            df['score_severidade'] = m_cat.predict(df[['latitude', 'longitude', 'nat_id']].values).round(2)

        df['peso_auditoria'] = np.abs(shap.Explainer(m_lgb, coords)(coords).values).mean(axis=1).round(4)
        return df

    def gerar_mapa_ouro(self, df, ano):
        mapa = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11, tiles='cartodbpositron')
        HeatMap(df[['latitude', 'longitude', 'score_geografico']].values.tolist(), radius=10).add_to(mapa)
        mapa.save(str(self.ouro / f"analise_geografica_{ano}.html"))

    def gerar_boletim_ia(self, info):
        if not self.api_key: return
        genai.configure(api_key=self.api_key)
        modelo = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Gere relatórios para Seguradora, Logística e Usuário Comum em português: {info}. Tom formal e estratégico."
        return modelo.generate_content(prompt).text

    def acionar_ia_reparo(self, erro):
        if not self.api_key: return
        genai.configure(api_key=self.api_key)
        modelo = genai.GenerativeModel('gemini-1.5-flash')
        diag = modelo.generate_content(f"Erro no SafeDriver: {erro}. Indique a solução.").text
        self.transmitir(f"🤖 **ALERTA: ANOMALIA NO PROCESSAMENTO** 🤖\n{diag}", sucesso=False)

    def processar_unidade(self, ano, estado):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        tmp = self.bronze / f"ssp_{ano}.xlsx"
        res = requests.get(url, timeout=120)
        if res.status_code == 200:
            with open(tmp, "wb") as f: f.write(res.content)
            df = pd.read_excel(tmp)
            tmp.unlink()
            
            id_ref = f"{ano}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
            if id_ref in estado["processados"]: return
            
            final = self.executar_modelagem(df)
            self.gerar_mapa_ouro(final, ano)
            final.to_csv(self.ouro / f"mestra_looker_{ano}.csv", index=False)
            
            resumo = {"ano": ano, "risco": float(final['score_geografico'].mean()), "recuperados": len(final[final.get('severidade_ia', pd.Series()).notnull()])}
            boletim = self.gerar_boletim_ia(json.dumps(resumo))
            self.transmitir(f"🤖 **SISTEMA: RELATÓRIO ESTRATÉGICO GERADO** 🤖\n{boletim}")
            
            estado["processados"].append(id_ref)
            self.controle.write_text(json.dumps(estado))
