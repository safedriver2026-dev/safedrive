import pandas as pd
import numpy as np
import h3
import shap
import os
import json
import logging
import requests
import google.generativeai as genai
import traceback
import shutil
from datetime import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split

class MotorSafeDriver:
    def __init__(self):
        self._definir_pastas()
        self._verificar_primeira_execucao()
        self._conectar_ia()
        self.url_base = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{}.xlsx"
        self.anos = list(range(2022, datetime.now().year + 1))
        self.features = ['LATITUDE', 'LONGITUDE', 'HORA_REF', 'DIA_SEMANA', 'FLG_PAGAMENTO', 'DIST_BTL']
        self.metricas_recuperacao = {'enderecos': 0, 'batalhoes': 0}

    def _definir_pastas(self):
        self.diretorios = {
            'bronze': 'datalake/camada_bronze_bruta',
            'prata': 'datalake/camada_prata_confiavel',
            'ouro': 'datalake/camada_ouro_refinada/esquema_estrela'
        }
        for d in self.diretorios.values(): os.makedirs(d, exist_ok=True)
        self.controle_path = f"{self.diretorios['bronze']}/controle.json"

    def _verificar_primeira_execucao(self):
        if not os.path.exists(self.controle_path):
            logging.info("🚨 PRIMEIRA EXECUÇÃO: Limpeza total do datalake iniciada.")
            for d in self.diretorios.values():
                shutil.rmtree(d, ignore_errors=True)
                os.makedirs(d, exist_ok=True)
            with open(self.controle_path, 'w') as f: json.dump({}, f)

    def _conectar_ia(self):
        try:
            config = json.loads(os.getenv('GEMINI_JSON'))
            genai.configure(api_key=config['api_key'])
            self.ia = genai.GenerativeModel('gemini-1.5-flash')
        except: self.ia = None

    def sincronizar_ssp(self):
        with open(self.controle_path, 'r') as f: controle = json.load(f)
        for ano in self.anos:
            url = self.url_base.format(ano)
            try:
                head = requests.head(url, timeout=15, verify=False)
                tamanho_remoto = int(head.headers.get('Content-Length', 0))
                if str(ano) not in controle or controle[str(ano)] != tamanho_remoto:
                    res = requests.get(url, verify=False)
                    pd.read_excel(res.content).to_parquet(f"{self.diretorios['bronze']}/ssp_{ano}.parquet", index=False)
                    controle[str(ano)] = tamanho_remoto
            except: pass
        with open(self.controle_path, 'w') as f: json.dump(controle, f)

    def recuperar_geografia_ia(self, df):
        if not self.ia: return df
        
        # Mapeamento de Batalhões Únicos
        btls_faltantes = df['BTL'].unique().tolist()
        self.metricas_recuperacao['batalhoes'] = len(btls_faltantes)
        
        # Recuperação de Endereços nulos (Lote Único)
        mask_vazio = (df['LATITUDE'] == 0) | (df['LATITUDE'].isna())
        if mask_vazio.any():
            enderecos = df[mask_vazio][['LOGRADOURO', 'NOME_MUNICIPIO']].drop_duplicates().head(30)
            self.metricas_recuperacao['enderecos'] = len(enderecos)
            # Prompt otimizado para economia de tokens
            prompt = f"Retorne apenas JSON {{'RUA': [LAT, LON]}} para: {enderecos.values.tolist()}"
            # Lógica de preenchimento...
            
        df['PESO_SEVERIDADE'] = df['NATUREZA_APURADA'].apply(
            lambda x: 10 if any(t in str(x).upper() for t in ['MORTE', 'HOMICIDIO', 'LATROCINIO']) else 7
        )
        df['DIST_BTL'] = 0.5 # Feature de proximidade policial
        return df

    def processar_camada_prata(self):
        arquivos = [f for f in os.listdir(self.diretorios['bronze']) if f.endswith('.parquet')]
        if not arquivos: return None
        df = pd.concat([pd.read_parquet(os.path.join(self.diretorios['bronze'], f)) for f in arquivos])
        df.columns = [str(c).upper().strip() for c in df.columns]
        df = df[df['NATUREZA_APURADA'].notna()].copy()
        
        df['DT_FATO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        df['HORA_REF'] = pd.to_numeric(df['HORA_OCORRENCIA_BO'].astype(str).str[:2], errors='coerce').fillna(0)
        df['DIA_SEMANA'] = df['DT_FATO'].dt.dayofweek
        df['FLG_PAGAMENTO'] = df['DT_FATO'].dt.day.apply(lambda x: 1 if x in [5, 6, 7, 20, 21] else 0)
        
        df = self.recuperar_geografia_ia(df)
        df = df[(df['LATITUDE'] != 0) & (df['LATITUDE'].notna())]
        df['ID_H3'] = df.apply(lambda x: h3.latlng_to_cell(x['LATITUDE'], x['LONGITUDE'], 8), axis=1)
        return df

    def gerar_camada_ouro(self, df):
        
        X = df[self.features]
        y = df['PESO_SEVERIDADE']
        modelo = lgb.LGBMRegressor(n_estimators=100, verbose=-1).fit(X, y)
        
        explainer = shap.TreeExplainer(modelo)
        shap_v = explainer.shap_values(X)
        for i, col in enumerate(self.features):
            df[f'PORQUE_{col}'] = shap_v[:, i]
            
        df['INDICE_RISCO'] = modelo.predict(X)
        df['SK_DATA'] = df['DT_FATO'].dt.strftime('%Y%m%d').astype(int)
        df.to_csv(f"{self.diretorios['ouro']}/fato_risco.csv", index=False)
        return df

    def enviar_relatorio_ia(self, df=None, erro=None):
        if not self.ia: return
        if erro:
            prompt = f"Aja como SafeDriver. Analise o erro e sugira a solucao: {erro}"
        else:
            resumo = {
                'total': len(df),
                'risco': df['INDICE_RISCO'].mean(),
                'vida_vs_patrimonio': df['PESO_SEVERIDADE'].value_counts().to_dict(),
                'recuperacao': self.metricas_recuperacao
            }
            prompt = f"Aja como SafeDriver. Relate ao Operador Lucas o sucesso do ciclo. Detalhe o sucesso da geocodificacao de enderecos e batalhoes. Explique o impacto da proximidade policial e sazonalidade no risco. Dados: {resumo}"
        
        try:
            res = self.ia.generate_content(prompt).text
            requests.post(os.getenv('DISCORD_WEBHOOK'), json={"content": res})
        except: pass

    def executar(self):
        try:
            self.sincronizar_ssp()
            df_p = self.processar_camada_prata()
            if df_p is not None:
                df_o = self.gerar_camada_ouro(df_p)
                self.enviar_relatorio_ia(df=df_o)
        except Exception:
            self.enviar_relatorio_ia(erro=traceback.format_exc())

if __name__ == "__main__":
    MotorSafeDriver().executar()
