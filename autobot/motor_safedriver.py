import os
import io
import requests
import pandas as pd
import numpy as np
import h3
import shap
import matplotlib.pyplot as plt
import hashlib
from pathlib import Path
from datetime import datetime
from geopy.geocoders import Nominatim
from lightgbm import LGBMRegressor

class MotorSafeDriver:
    def __init__(self):
        self.raiz = Path(".")
        self.pastas = {
            "bronze": self.raiz / "datalake" / "bronze",
            "prata": self.raiz / "datalake" / "prata",
            "ouro": self.raiz / "datalake" / "ouro",
            "docs": self.raiz / "documentacao"
        }
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        
        # ESCALABILIDADE: Range dinâmico de anos
        ano_inicial = 2022
        self.anos = list(range(ano_inicial, datetime.now().year + 1))
        self.agente = {'User-Agent': 'SafeDriver-Industrial-V12'}
        self.geolocalizador = Nominatim(user_agent="safedriver_fatec_v12")
        self.webhook = os.environ.get("DISCORD_WEBHOOK")

    def enviar_relatorio_discord(self, titulo, mensagem, cor):
        if not self.webhook: return
        payload = {
            "embeds": [{
                "title": titulo,
                "description": mensagem,
                "color": cor,
                "timestamp": datetime.now().isoformat(),
                "footer": {"text": "SafeDriver Operational Monitor"}
            }]
        }
        requests.post(self.webhook, json=payload, timeout=10)

    def descobrir_aba_de_dados(self, conteudo_excel):
        excel = pd.ExcelFile(io.BytesIO(conteudo_excel))
        for aba in excel.sheet_names:
            df_temp = excel.parse(aba, nrows=10)
            df_temp.columns = [str(c).upper().strip() for c in df_temp.columns]
            if any(k in df_temp.columns for k in ['LATITUDE', 'NATUREZA_APURADA', 'RUBRICA']):
                return excel.parse(aba)
        return None

    def processar_ia_e_explicabilidade(self, df):
        df.columns = [c.lower().strip() for c in df.columns]
        
        # 1. Classificação de Perfis (Vetor de Exposição)
        df['perfil'] = 'Geral'
        col_crime = next((c for c in ['natureza_apurada', 'rubrica'] if c in df.columns), 'rubrica')
        df['crime_texto'] = df[col_crime].fillna('').astype(str).upper()
        
        df.loc[df['crime_texto'].str.contains('VEÍCULO|MOTO|CARGA|AUTO'), 'perfil'] = 'Motorista'
        df.loc[df['crime_texto'].str.contains('BICICLETA|BIKE'), 'perfil'] = 'Ciclista'
        
        col_local = next((c for c in ['descr_tipolocal', 'descr_local'] if c in df.columns), 'descr_tipolocal')
        if col_local in df.columns:
            local_texto = df[col_local].fillna('').astype(str).upper()
            df.loc[(local_texto.str.contains('VIA PÚBLICA')) & (df['crime_texto'].str.contains('CELULAR|PESSOA')), 'perfil'] = 'Pedestre'
        
        df['peso_severidade'] = df['crime_texto'].apply(lambda x: 15 if 'ROUBO' in x else 2)

        # 2. Tratamento Geospacial
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
        
        if df.empty: raise ValueError("Nenhum dado geolocalizado válido após filtragem.")

        # 3. Agregação por Hexágonos H3 (Resolução 9)
        df['h3_index'] = df.apply(lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1)
        fato_risco = df.groupby(['h3_index', 'desc_periodo', 'perfil'])['peso_severidade'].sum().reset_index()
        
        fato_risco['lat'] = fato_risco['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        fato_risco['lon'] = fato_risco['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        fato_risco['perfil_idx'] = fato_risco['perfil'].astype('category').cat.codes
        fato_risco['periodo_idx'] = fato_risco['desc_periodo'].astype('category').cat.codes

        # 4. Machine Learning: Treino e Extração SHAP
        X = fato_risco[['lat', 'lon', 'perfil_idx', 'periodo_idx']]
        y = fato_risco['peso_severidade']
        modelo = LGBMRegressor(n_estimators=100, verbose=-1).fit(X, y)

        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X)
        
        # Adiciona influências das variáveis como colunas para o Looker
        for i, col in enumerate(X.columns):
            fato_risco[f'influencia_{col}'] = shap_values[:, i]
            
        fato_risco['score_predito'] = modelo.predict(X)

        # 5. Exportação e Assinatura Digital
        caminho_final = self.pastas["ouro"] / "base_final_looker.csv"
        fato_risco.to_csv(caminho_final, index=False)
        hash_val = hashlib.sha256(open(caminho_final, "rb").read()).hexdigest()
        with open(self.pastas["ouro"] / "assinatura.sha256", "w") as f: f.write(hash_val)
        
        return len(fato_risco)

    def iniciar(self):
        try:
            pool_dados = []
            logs_operacionais = []
            
            for ano in self.anos:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                try:
                    res = requests.get(url, headers=self.agente, timeout=300)
                    if res.status_code == 200:
                        df = self.descobrir_aba_de_dados(res.content)
                        if df is not None: 
                            pool_dados.append(df)
                            logs_operacionais.append(f"✅ {ano}: {len(df)} linhas importadas.")
                except Exception as e:
                    logs_operacionais.append(f"⚠️ {ano}: Falha na conexão.")

            if not pool_dados: raise Exception("Falha crítica: Nenhuma base de dados pôde ser carregada.")

            total_celulas = self.processar_ia_e_explicabilidade(pd.concat(pool_dados))
            
            # Relatórios Discord
            self.enviar_relatorio_discord("📊 Relatório Operacional", "\n".join(logs_operacionais), 3447003)
            self.enviar_relatorio_discord("🚀 Relatório Executivo", f"Inteligência SafeDriver atualizada.\n📍 Células de Risco: {total_celulas}\n🛡️ Integridade: SHA256 Gerado.", 3066993)

        except Exception as e:
            self.enviar_relatorio_discord("🚨 Erro Crítico no Pipeline", f"Ocorreu uma falha na etapa de processamento:\n`{str(e)}`", 15158332)
            raise e

if __name__ == "__main__":
    MotorSafeDriver().iniciar()
