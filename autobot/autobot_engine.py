import os
import json
import requests
import pandas as pd
import numpy as np
import h3
import folium
import shap
import matplotlib.pyplot as plt
import io
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime

class SafeDriverEngine:
    def __init__(self, perfil_negocio="LOGISTICA"):
        self.perfil = perfil_negocio
        self.ano_atual = datetime.now().year
        self.anos = list(range(2022, self.ano_atual + 1))
        self.raiz = Path(".")
        self.camadas = {
            "bronze": self.raiz / "datalake" / "bronze",
            "prata": self.raiz / "datalake" / "prata",
            "ouro": self.raiz / "datalake" / "ouro"
        }
        for p in self.camadas.values(): p.mkdir(parents=True, exist_ok=True)

    def notificar_discord(self, sucesso: bool, mensagem: str):
        webhook = os.environ.get("DISCORD_SUCESSO") if sucesso else os.environ.get("DISCORD_ERRO")
        if webhook:
            try: requests.post(webhook, json={"content": mensagem}, timeout=5)
            except: pass

    def aplicar_pesos_negocio(self, df):
        """Camada de Inteligência de Negócio: Traduz o Código Penal em Risco Financeiro"""
        if self.perfil == "LOGISTICA":
            pesos = {
                'ROUBO DE CARGA': 15.0, 'ROUBO DE VEICULO': 10.0, 
                'FURTO DE VEICULO': 8.0, 'LATROCINIO': 12.0, 'OUTROS': 1.0
            }
        else: # Segurança Pública (TCC)
            pesos = {
                'HOMICIDIO': 10.0, 'ESTUPRO': 10.0, 'ROUBO': 5.0, 
                'FURTO': 1.0, 'LESAO CORPORAL': 3.0
            }
        
        def calcular(rubrica):
            rubrica = str(rubrica).upper()
            for crime, peso in pesos.items():
                if crime in rubrica: return peso
            return 1.0
        
        df['peso_gravidade'] = df['rubrica'].apply(calcular)
        return df

    def ingestao_delta_bronze(self, ano):
        """Delta Sync: Só baixa se o arquivo Parquet não existir localmente"""
        arq_pq = self.camadas["bronze"] / f"bruto_{ano}.parquet"
        if arq_pq.exists(): return pd.read_parquet(arq_pq)
        
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            xls = pd.ExcelFile(io.BytesIO(r.content))
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet)
                df.columns = [str(c).upper().strip() for c in df.columns]
                if 'LATITUDE' in df.columns:
                    df.to_parquet(arq_pq, index=False)
                    return df
            return None
        except: return None

    def refinamento_prata(self, df, ano):
        arq_prata = self.camadas["prata"] / f"prata_{ano}.parquet"
        if arq_prata.exists(): return pd.read_parquet(arq_prata)
        
        df.columns = [str(c).lower() for c in df.columns]
        df = self.aplicar_pesos_negocio(df)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
        df.to_parquet(arq_prata, index=False)
        return df

    def inteligencia_ouro(self, df_total):
        # Indexação H3 (Hexágonos)
        df_total['h3_index'] = df_total.apply(lambda x: h3.latlng_to_cell(float(x['latitude']), float(x['longitude']), 9), axis=1)
        
        # Agrupamento Ponderado (Gravidade Penal)
        analise = df_total.groupby(['h3_index', 'desc_periodo'])['peso_gravidade'].sum().reset_index(name='impacto_total')
        analise['lat'] = analise['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        analise['lon'] = analise['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        analise['periodo_cat'] = analise['desc_periodo'].astype('category').cat.codes

        X = analise[['lat', 'lon', 'periodo_cat']]
        y = analise['impacto_total']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Ensemble: LGBM(40%) + CAT(40%) + KNN(20%)
        m_lgb = LGBMRegressor(n_estimators=100, random_state=42, verbose=-1)
        m_cat = CatBoostRegressor(n_estimators=100, verbose=0, random_state=42)
        m_knn = KNeighborsRegressor(n_neighbors=5, weights='distance')

        m_lgb.fit(X_train, y_train); m_cat.fit(X_train, y_train); m_knn.fit(X_train, y_train)
        
        preds = (m_lgb.predict(X_test)*0.4) + (m_cat.predict(X_test)*0.4) + (m_knn.predict(X_test)*0.2)
        
        # Auditoria SHAP
        explainer = shap.TreeExplainer(m_lgb)
        shap_v = explainer.shap_values(X)
        plt.figure(); shap.summary_plot(shap_v, X, show=False)
        plt.savefig(self.camadas["ouro"] / "ia_audit.png", bbox_inches='tight'); plt.close()

        # Score Final 0-100
        analise['score_risco'] = (((m_lgb.predict(X)*0.4 + m_cat.predict(X)*0.4 + m_knn.predict(X)*0.2) - y.min()) / (y.max() - y.min() + 1e-9) * 100).round(2)
        analise.to_csv(self.camadas["ouro"] / "base_looker.csv", index=False)
        
        return analise

    def executar(self):
        try:
            lista_prata = []
            for a in self.anos:
                b = self.ingestao_delta_bronze(a)
                if b is not None: lista_prata.append(self.refinamento_prata(b, a))
            
            if not lista_prata:
                self.notificar_discord(True, "⚠️ SafeDriver: Cold Start - Criando infraestrutura e aguardando dados SSP.")
                pd.DataFrame(columns=['h3_index','score_risco']).to_csv(self.camadas["ouro"] / "base_looker.csv", index=False)
                return

            self.inteligencia_ouro(pd.concat(lista_prata, ignore_index=True))
            self.notificar_discord(True, "✅ SafeDriver: Delta Sync e IA atualizados!")
        except Exception as e:
            self.notificar_discord(False, f"🚨 Erro: {str(e)}")
            raise e

if __name__ == "__main__":
    SafeDriverEngine(perfil_negocio="LOGISTICA").executar()
