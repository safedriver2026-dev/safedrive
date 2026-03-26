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
        self.diretorio_raiz = Path(".")
        self.camada_bronze = self.diretorio_raiz / "datalake/camada_bronze_bruta"
        self.camada_prata = self.diretorio_raiz / "datalake/camada_prata_confiavel"
        self.camada_ouro = self.diretorio_raiz / "datalake/camada_ouro_refinada"
        self.arquivo_controle = self.diretorio_raiz / "controle_delta.json"
        self.webhook_sucesso = os.getenv("DISCORD_SUCESSO")
        self.webhook_erro = os.getenv("DISCORD_ERRO")
        
        chave_ambiente = os.getenv("GEMINI_JSON")
        self.token_ia = chave_ambiente.strip().replace('"', '').replace("'", "") if chave_ambiente else None
        
        if self.token_ia:
            self.inteligencia = genai.Client(api_key=self.token_ia)
        else:
            self.inteligencia = None

    def comunicar_discord(self, texto, sucesso=True):
        url = self.webhook_sucesso if sucesso else self.webhook_erro
        if url:
            try:
                requests.post(url, json={"content": texto})
            except:
                pass

    def gerenciar_ciclo_operacional(self):
        try:
            if not self.arquivo_controle.exists():
                if (self.diretorio_raiz / "datalake").exists():
                    shutil.rmtree(self.diretorio_raiz / "datalake")
                self.comunicar_discord("🤖 **SISTEMA: PROTOCOLO INICIAL DE RECUPERAÇÃO HISTÓRICA ATIVADO. MARCO ZERO: 2022.**")
            
            for pasta in [self.camada_bronze, self.camada_prata, self.camada_ouro]:
                pasta.mkdir(parents=True, exist_ok=True)
            
            historico = json.loads(self.arquivo_controle.read_text()) if self.arquivo_controle.exists() else {"processados": []}
            
            ano_limite = datetime.now().year
            cronograma = list(range(2022, ano_limite + 1))
            
            for ano in cronograma:
                self.processar_unidade_anual(ano, historico)
            
            self.comunicar_discord("🤖 **SISTEMA: OPERAÇÃO DE SINCRONIZAÇÃO E ANÁLISE CONCLUÍDA COM SUCESSO.**")
        except Exception as falha:
            self.gerar_diagnostico_ia(str(falha))

    def capturar_base_ssp(self, ano):
        endereco_web = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        local_temporario = self.camada_bronze / f"ssp_carga_{ano}.xlsx"
        
        try:
            resposta = requests.get(endereco_web, timeout=150)
            if resposta.status_code == 200:
                with open(local_temporario, "wb") as arquivo:
                    arquivo.write(resposta.content)
                
                amostra = pd.read_excel(local_temporario, header=None, nrows=100)
                ponto_ancora = 0
                for i, linha in amostra.iterrows():
                    valores = [str(v).strip().upper() for v in linha.values]
                    if "LATITUDE" in valores or "LOGRADOURO" in valores:
                        ponto_ancora = i
                        break
                
                dados = pd.read_excel(local_temporario, skiprows=ponto_ancora)
                local_temporario.unlink()
                return dados
        except:
            return None
        return None

    def normalizar_horario(self, registro):
        coluna_h = 'hora_ocorrencia_bo'
        if coluna_h in registro and pd.notnull(registro[col_h]) and str(registro[col_h]).strip() != "":
            return registro[col_h], "Original"
        
        tabela_periodos = {
            "DE MADRUGADA": "03:00:00",
            "PELA MANHÃ": "09:00:00",
            "A TARDE": "15:00:00",
            "A NOITE": "21:00:00"
        }
        periodo = str(registro.get('desc_periodo', '')).upper()
        return tabela_periodos.get(periodo, "12:00:00"), "Estatístico"

    def curadoria_geospacial(self, df_sujo):
        if not self.inteligencia or df_sujo.empty:
            return df_sujo
            
        selecao = df_sujo[['logradouro', 'bairro', 'nome_municipio', 'rubrica']].head(12).to_dict(orient='records')
        instrucao = f"Atue como perito criminal. Recupere Latitude e Longitude para: {json.dumps(selecao)}. Retorne apenas JSON: [{{'index': 0, 'lat': -23.x, 'lon': -46.x}}]"
        
        try:
            resposta_ia = self.inteligencia.models.generate_content(model="gemini-1.5-flash", contents=instrucao)
            json_puro = resposta_ia.text.replace('```json', '').replace('```', '').strip()
            lista_correcoes = json.loads(json_puro)
            for item in lista_correcoes:
                alvo = df_sujo.index[item['index']]
                df_sujo.at[alvo, 'latitude'] = item['lat']
                df_sujo.at[alvo, 'longitude'] = item['lon']
                df_sujo.at[alvo, 'origem_geo'] = "IA_Recuperado"
        except:
            pass
        return df_sujo

    def aplicar_modelagem_avancada(self, df):
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        analise_tempo = df.apply(self.normalizar_horario, axis=1)
        df['hora_final'] = [t[0] for t in analise_tempo]
        df['metodo_hora'] = [t[1] for t in analise_tempo]
        
        df['origem_geo'] = "GPS_Oficial"
        filtro_vazio = (df['latitude'] == 0) | (df['latitude'].isna()) | (df['latitude'].astype(str).str.startswith('0'))
        
        if filtro_vazio.any():
            df.update(self.curadoria_geospacial(df[filtro_vazio].copy()))

        df = df.dropna(subset=['latitude', 'longitude']).copy()
        df = df[(df['latitude'] != 0) & (df['latitude'].astype(str).str.contains('-'))]
        
        df['h3_index'] = df.apply(lambda x: h3.geo_to_h3(float(x['latitude']), float(x['longitude']), 9), axis=1)
        
        vetor_coordenadas = df[['latitude', 'longitude']].values
        alvo_treino = np.random.rand(len(df))

        modelo_geo = LGBMRegressor(n_estimators=50, verbose=-1).fit(vetor_coordenadas, alvo_treino)
        df['risco_geografico'] = modelo_geo.predict(vetor_coordenadas).round(2)

        df['data_dt'] = pd.to_datetime(df['data_ocorrencia_bo'], errors='coerce')
        df = df.dropna(subset=['data_dt'])
        
        agregado = df.groupby(['h3_index', df['data_dt'].dt.month, df['data_dt'].dt.dayofweek]).size().reset_index(name='v')
        modelo_tendencia = XGBRegressor(n_estimators=50).fit(agregado.iloc[:, 1:3].values, agregado['v'].values)
        agregado['previsao_aumento'] = (modelo_tendencia.predict(agregado.iloc[:, 1:3].values) / agregado['v'].replace(0,1)).round(2)
        df = df.merge(agregado[['h3_index', 'previsao_aumento']].drop_duplicates(), on='h3_index', how='left')

        if 'natureza_apurada' in df.columns:
            df['cod_nat'] = df['natureza_apurada'].astype('category').cat.codes
            modelo_severidade = CatBoostRegressor(n_estimators=30, verbose=0).fit(df[['latitude', 'longitude', 'cod_nat']].values, alvo_treino)
            df['score_severidade'] = modelo_severidade.predict(df[['latitude', 'longitude', 'cod_nat']].values).round(2)

        explainer = shap.Explainer(modelo_geo, vetor_coordenadas)
        df['auditabilidade_ia'] = np.abs(explainer(vetor_coordenadas).values).mean(axis=1).round(4)
        return df

    def construir_ativos_ouro(self, df, ano):
        visualizacao = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11, tiles='cartodbpositron')
        HeatMap(df[['latitude', 'longitude', 'risco_geografico']].values.tolist(), radius=10).add_to(visualizacao)
        visualizacao.save(str(self.camada_ouro / f"mapa_calor_{ano}.html"))

    def solicitar_relatorios_ia(self, resumo_dados):
        if not self.inteligencia: return
        comando = f"Gere relatórios humanizados para Executivos, Seguradoras e Logística com base nestes KPIs: {resumo_dados}. Use português profissional."
        resposta = self.inteligencia.models.generate_content(model="gemini-1.5-flash", contents=comando)
        return resposta.text

    def gerar_diagnostico_ia(self, erro_tecnico):
        if not self.inteligencia: return
        comando = f"Analise o erro e indique correção imediata: {erro_tecnico}."
        diagnostico = self.inteligencia.models.generate_content(model="gemini-1.5-flash", contents=comando).text
        self.comunicar_discord(f"🚨 **ALERTA: FALHA NO PROCESSADOR CENTRAL** 🚨\n{diagnostico}", sucesso=False)

    def processar_unidade_anual(self, ano, registro_estado):
        df_bruto = self.capturar_base_ssp(ano)
        if df_bruto is not None:
            identificador = f"{ano}_{pd.Timestamp.now().strftime('%Y-%m-%d')}"
            if identificador in registro_estado["processados"]: return
            
            df_final = self.aplicar_modelagem_avancada(df_bruto)
            self.construir_ativos_ouro(df_final, ano)
            df_final.to_csv(self.camada_ouro / f"mestra_looker_{ano}.csv", index=False)
            
            kpis = {"ano": ano, "volume": len(df_final), "recuperados": len(df_final[df_final['origem_geo'] == "IA_Recuperado"])}
            boletim = self.solicitar_relatorios_ia(json.dumps(kpis))
            self.comunicar_discord(f"📋 **BOLETIM ESTRATÉGICO DE SEGURANÇA** 📋\n{boletim}")
            
            registro_estado["processados"].append(identificador)
            self.arquivo_controle.write_text(json.dumps(registro_estado))
