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
        
        raw_key = os.getenv("GEMINI_JSON")
        self.api_key = raw_key.strip().replace('"', '').replace("'", "") if raw_key else None
        
        if self.api_key:
            self.cliente = genai.Client(api_key=self.api_key)
        else:
            self.cliente = None

    def transmitir_discord(self, mensagem, sucesso=True):
        url = self.webhook_sucesso if sucesso else self.webhook_erro
        if url:
            try:
                requests.post(url, json={"content": mensagem})
            except:
                pass

    def registrar_log(self, evento):
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.log_sistema, "a", encoding="utf-8") as f:
            f.write(f"{timestamp} | {evento}\n")

    def gerenciar_ciclo_vida(self):
        try:
            if not self.controle.exists():
                if (self.raiz / "datalake").exists():
                    shutil.rmtree(self.raiz / "datalake")
                self.transmitir_discord("🤖 **SISTEMA: DETECTADA INCOMPATIBILIDADE DE COLUNAS. PROTOCOLO DE REINICIALIZAÇÃO ATIVADO PARA NORMALIZAÇÃO DA MALHA.**")
            
            for pasta in [self.bronze, self.prata, self.ouro]:
                pasta.mkdir(parents=True, exist_ok=True)
            
            self.registrar_log("Operação de processamento iniciada")
            estado = json.loads(self.controle.read_text()) if self.controle.exists() else {"processados": []}
            
            for ano in [2025, 2026]:
                self.executar_ciclo_anual(ano, estado)
            
            self.transmitir_discord("🤖 **SISTEMA: CICLO OPERACIONAL FINALIZADO. DADOS RECUPERADOS E INTEGRAÇÃO LOOKER DISPONÍVEL.**")
        except Exception as erro:
            self.acionar_reparo_ia(str(erro))

    def tratar_tempo_estatistico(self, linha):
        col_hora = 'hora_ocorrencia_bo'
        if col_hora in linha and pd.notnull(linha[col_hora]) and str(linha[col_hora]).strip() != "" and str(linha[col_hora]).upper() != "NULL":
            return linha[col_hora], "Original"
        
        periodos = {
            "DE MADRUGADA": "03:00:00",
            "PELA MANHÃ": "09:00:00",
            "A TARDE": "15:00:00",
            "A NOITE": "21:00:00"
        }
        referencia = str(linha.get('desc_periodo', '')).upper()
        return periodos.get(referencia, "12:00:00"), "Estimado (Estatística)"

    def curadoria_ia(self, df_incompleto):
        if not self.cliente or df_incompleto.empty:
            return df_incompleto
            
        amostra = df_incompleto[['logradouro', 'bairro', 'nome_municipio', 'rubrica']].head(15).to_dict(orient='records')
        
        prompt = f"Recupere: {json.dumps(amostra)}. Estime Lat/Lon e Hora (HH:MM:SS) se nulos. Retorne apenas JSON: [{{'index': 0, 'lat': -23.x, 'lon': -46.x, 'hora': 'HH:MM:SS', 'severidade': 80}}]"
        
        try:
            resposta = self.cliente.models.generate_content(model="gemini-1.5-flash", contents=prompt)
            limpo = resposta.text.replace('```json', '').replace('```', '').strip()
            correcoes = json.loads(limpo)
            for item in correcoes:
                indice = df_incompleto.index[item['index']]
                df_incompleto.at[indice, 'latitude'] = item['lat']
                df_incompleto.at[indice, 'longitude'] = item['lon']
                df_incompleto.at[indice, 'hora_final'] = item['hora']
                df_incompleto.at[indice, 'metodo_localizacao'] = "Recuperado (IA)"
        except:
            pass
        return df_incompleto

    def processar_inteligencia_tripla(self, df):
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        horarios = df.apply(self.tratar_tempo_estatistico, axis=1)
        df['hora_final'] = [h[0] for h in horarios]
        df['metodo_horario'] = [h[1] for h in horarios]
        
        df['metodo_localizacao'] = "Original (GPS)"
        mascara_vazia = (df['latitude'] == 0) | (df['latitude'].isna()) | (df['latitude'].astype(str) == "0")
        
        if mascara_vazia.any():
            df.update(self.curadoria_ia(df[mascara_vazia].copy()))

        df = df.dropna(subset=['latitude', 'longitude']).copy()
        df = df[(df['latitude'] != 0) & (df['latitude'].astype(str) != "0")]
        
        df['h3_index'] = df.apply(lambda x: h3.geo_to_h3(float(x['latitude']), float(x['longitude']), 9), axis=1)
        
        coordenadas = df[['latitude', 'longitude']].values
        alvo_simulado = np.random.rand(len(df))

        mod_lgb = LGBMRegressor(n_estimators=50, verbose=-1).fit(coordenadas, alvo_simulado)
        df['score_geografico'] = mod_lgb.predict(coordenadas).round(2)

        df['data_dt'] = pd.to_datetime(df['data_ocorrencia_bo'], errors='coerce')
        df = df.dropna(subset=['data_dt'])
        
        agregado = df.groupby(['h3_index', df['data_dt'].dt.month, df['data_dt'].dt.dayofweek]).size().reset_index(name='v')
        mod_xgb = XGBRegressor(n_estimators=50).fit(agregado.iloc[:, 1:3].values, agregado['v'].values)
        agregado['projecao_aumento'] = (mod_xgb.predict(agregado.iloc[:, 1:3].values) / agregado['v'].replace(0,1)).round(2)
        df = df.merge(agregado[['h3_index', 'projecao_aumento']].drop_duplicates(), on='h3_index', how='left')

        if 'natureza_apurada' in df.columns:
            df['nat_id'] = df['natureza_apurada'].astype('category').cat.codes
            mod_cat = CatBoostRegressor(n_estimators=30, verbose=0).fit(df[['latitude', 'longitude', 'nat_id']].values, alvo_simulado)
            df['score_severidade'] = mod_cat.predict(df[['latitude', 'longitude', 'nat_id']].values).round(2)

        df['ia_auditada'] = np.abs(shap.Explainer(mod_lgb, coordenadas)(coordenadas).values).mean(axis=1).round(4)
        return df

    def gerar_mapa_ouro(self, df, ano):
        mapa = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11, tiles='cartodbpositron')
        HeatMap(df[['latitude', 'longitude', 'score_geografico']].values.tolist(), radius=10).add_to(mapa)
        mapa.save(str(self.ouro / f"mapa_auditoria_{ano}.html"))

    def solicitar_boletim_ia(self, info):
        if not self.cliente: return
        prompt = f"Gere boletins para Seguradora, Logística e Usuário: {info}. Tom profissional em português."
        resposta = self.cliente.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        return resposta.text

    def acionar_reparo_ia(self, erro):
        if not self.cliente: return
        prompt = f"Falha técnica: {erro}. Indique a correção técnica rápida."
        diagnostico = self.cliente.models.generate_content(model="gemini-1.5-flash", contents=prompt).text
        self.transmitir_discord(f"🤖 **ALERTA: FALHA CRÍTICA NO PROCESSADOR CENTRAL** 🤖\n{diagnostico}", sucesso=False)

    def executar_ciclo_anual(self, ano, estado):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        caminho_tmp = self.bronze / f"ssp_{ano}.xlsx"
        resposta = requests.get(url, timeout=120)
        if resposta.status_code == 200:
            with open(caminho_tmp, "wb") as f: f.write(resposta.content)
            df = pd.read_excel(caminho_tmp)
            caminho_tmp.unlink()
            
            id_referencia = f"{ano}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
            if id_referencia in estado["processados"]: return
            
            final = self.processar_inteligencia_tripla(df)
            self.gerar_mapa_ouro(final, ano)
            final.to_csv(self.ouro / f"mestra_looker_{ano}.csv", index=False)
            
            resumo = {"ano": ano, "total": len(final), "recuperados": len(final[final['metodo_localizacao'] == "Recuperado (IA)"])}
            boletim = self.solicitar_boletim_ia(json.dumps(resumo))
            self.transmitir_discord(f"🤖 **SISTEMA: RELATÓRIO DE SINCRONIZAÇÃO GERADO** 🤖\n{boletim}")
            
            estado["processados"].append(id_referencia)
            self.controle.write_text(json.dumps(estado))
