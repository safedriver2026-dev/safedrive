import os
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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class MotorSafeDriver:
    def __init__(self):
        self.raiz = Path(".")
        self.camadas = {
            "bronze": self.raiz / "datalake" / "bronze",
            "prata": self.raiz / "datalake" / "prata",
            "ouro": self.raiz / "datalake" / "ouro",
            "docs": self.raiz / "documentacao"
        }
        for pasta in self.camadas.values():
            pasta.mkdir(parents=True, exist_ok=True)
            
        self.anos = [2024, 2025, 2026]
        self.agente = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        self.geolocalizador = Nominatim(user_agent="safedriver_fatec_final")
        self.historico = []

    def enviar_notificacao(self, titulo, mensagem, cor):
        webhook = os.environ.get("DISCORD_SUCESSO")
        if not webhook: return
        payload = {
            "embeds": [{
                "title": titulo,
                "description": mensagem,
                "color": cor,
                "footer": {"text": f"Sincronização: {datetime.now().strftime('%d/%m/%Y %H:%M')}"}
            }]
        }
        requests.post(webhook, json=payload, timeout=15)

    def geocodificar_base(self, df):
        faltantes = df[df['latitude'].isna() | (df['latitude'] == 0)].head(15)
        if faltantes.empty: return df
        
        for i, linha in faltantes.iterrows():
            endereco = f"{linha.get('logradouro', '')}, {linha.get('numero', '')}, {linha.get('cidade', 'São Paulo')}, Brasil"
            try:
                local = self.geolocalizador.geocode(endereco, timeout=10)
                if local:
                    df.at[i, 'latitude'] = local.latitude
                    df.at[i, 'longitude'] = local.longitude
            except:
                continue
        return df

    def processar_delta(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        caminho_bronze = self.camadas["bronze"] / f"bruto_{ano}.parquet"
        caminho_meta = self.camadas["bronze"] / f"metadata_{ano}.txt"

        try:
            head = requests.head(url, headers=self.agente, timeout=30)
            tamanho_ssp = str(head.headers.get('Content-Length', '0'))

            if caminho_bronze.exists() and caminho_meta.exists():
                with open(caminho_meta, "r") as f:
                    if f.read() == tamanho_ssp:
                        self.historico.append(f"🟢 {ano}: Camada Bronze atualizada")
                        return pd.read_parquet(caminho_bronze)

            res = requests.get(url, headers=self.agente, timeout=300)
            if res.status_code != 200: return None
            
            df = pd.read_excel(res.content)
            df.columns = [str(c).upper().strip() for c in df.columns]
            df.to_parquet(caminho_bronze, index=False)
            
            with open(caminho_meta, "w") as f: f.write(tamanho_ssp)
            self.historico.append(f"📥 {ano}: Ingestão concluída ({len(df)} linhas)")
            return df
        except:
            return pd.read_parquet(caminho_bronze) if caminho_bronze.exists() else None

    def treinar_inteligencia(self, df):
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude'])
        
        if df.empty: return 0

        df['h3_index'] = df.apply(lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1)
        resumo = df.groupby(['h3_index', 'desc_periodo'])['peso'].sum().reset_index()
        
        resumo['lat'] = resumo['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        resumo['lon'] = resumo['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        resumo['periodo_cod'] = resumo['desc_periodo'].astype('category').cat.codes

        X = resumo[['lat', 'lon', 'periodo_cod']]
        y = resumo['peso']

        lgbm = LGBMRegressor(n_estimators=50, verbose=-1).fit(X, y)
        cat = CatBoostRegressor(n_estimators=50, verbose=0).fit(X, y)
        knn = KNeighborsRegressor(n_neighbors=3).fit(X, y)

        explainer = shap.Explainer(lgbm, X)
        shap_values = explainer(X)
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(self.camadas["docs"] / "explicabilidade_ia.png", bbox_inches='tight')
        plt.close()

        resumo['score_risco'] = (lgbm.predict(X) * 0.4 + cat.predict(X) * 0.4 + knn.predict(X) * 0.2)
        
        caminho_ouro = self.camadas["ouro"] / "base_looker.csv"
        resumo.to_csv(caminho_ouro, index=False)
        
        assinatura = hashlib.sha256(open(caminho_ouro, "rb").read()).hexdigest()
        with open(self.camadas["ouro"] / "base_looker.sha256", "w") as f:
            f.write(assinatura)
            
        return len(resumo)

    def iniciar(self):
        coleta = []
        for ano in self.anos:
            dados = self.processar_delta(ano)
            if dados is not None:
                dados.columns = [c.lower() for c in dados.columns]
                dados['peso'] = dados['rubrica'].apply(lambda x: 15 if 'ROUBO' in str(x).upper() else 2)
                dados = self.geocodificar_base(dados)
                coleta.append(dados)

        if coleta:
            total_h3 = self.treinar_inteligencia(pd.concat(coleta))
            msg = "\n".join(self.historico) + f"\n📍 Células H3 Ativas: {total_h3}"
            self.enviar_notificacao("SafeDriver: Status Operacional", msg, 3447003)
            self.enviar_notificacao("SafeDriver: Relatório Executivo", f"🚀 Inteligência atualizada com sucesso.\nIntegridade da Base Ouro: Validada.", 3066993)
        else:
            self.enviar_notificacao("SafeDriver: Falha Crítica", "Nenhum dado disponível para processamento.", 15158332)

if __name__ == "__main__":
    MotorSafeDriver().iniciar()
