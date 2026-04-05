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
from geopy.extra.rate_limiter import RateLimiter
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
            
        self.anos = list(range(2022, datetime.now().year + 1))
        self.cabecalhos = {'User-Agent': 'SafeDriver-Industrial-Bot'}
        self.geolocalizador = Nominatim(user_agent="safedriver_fatec")
        self.logs_operacionais = []
        self.resumo_executivo = {"total": 0, "novos": 0, "erros": 0}

    def notificar(self, titulo, mensagem, cor):
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
        requests.post(webhook, json=payload, timeout=10)

    def recuperar_geo(self, df):
        df_falha = df[df['latitude'].isna() | (df['latitude'] == 0)].copy()
        if df_falha.empty: return df
        
        for idx, linha in df_falha.head(20).iterrows():
            endereco = f"{linha.get('logradouro', '')}, {linha.get('numero', '')}, {linha.get('cidade', 'São Paulo')}, Brasil"
            try:
                local = self.geolocalizador.geocode(endereco, timeout=10)
                if local:
                    df.at[idx, 'latitude'] = local.latitude
                    df.at[idx, 'longitude'] = local.longitude
            except:
                continue
        return df

    def gerenciar_delta(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        caminho_bronze = self.camadas["bronze"] / f"bruto_{ano}.parquet"
        caminho_hash = self.camadas["bronze"] / f"tamanho_{ano}.txt"

        try:
            head = requests.head(url, headers=self.cabecalhos, timeout=30)
            tamanho_servidor = str(head.headers.get('Content-Length', '0'))

            if caminho_bronze.exists() and caminho_hash.exists():
                with open(caminho_hash, "r") as f:
                    if f.read() == tamanho_servidor:
                        self.logs_operacionais.append(f"🟢 {ano}: DeltaSync mantido")
                        return pd.read_parquet(caminho_bronze)

            res = requests.get(url, headers=self.cabecalhos, timeout=300)
            df = pd.read_excel(res.content)
            df.columns = [c.upper().strip() for c in df.columns]
            df.to_parquet(caminho_bronze, index=False)
            with open(caminho_hash, "w") as f: f.write(tamanho_servidor)
            self.logs_operacionais.append(f"📥 {ano}: Novos dados integrados")
            return df
        except:
            return pd.read_parquet(caminho_bronze) if caminho_bronze.exists() else None

    def gerar_ia_e_assinatura(self, df):
        df['h3_index'] = df.apply(lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1)
        base_h3 = df.groupby(['h3_index', 'desc_periodo'])['peso'].sum().reset_index()
        
        base_h3['lat'] = base_h3['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        base_h3['lon'] = base_h3['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        base_h3['periodo_idx'] = base_h3['desc_periodo'].astype('category').cat.codes

        X = base_h3[['lat', 'lon', 'periodo_idx']]
        y = base_h3['peso']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        lgb = LGBMRegressor(n_estimators=100, verbose=-1).fit(X_train, y_train)
        cat = CatBoostRegressor(n_estimators=100, verbose=0).fit(X_train, y_train)
        knn = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)

        explainer = shap.Explainer(lgb, X)
        shap_values = explainer(X)
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(self.camadas["docs"] / "explicabilidade_ia.png", bbox_inches='tight')
        plt.close()

        base_h3['score_risco'] = (lgb.predict(X) * 0.4 + cat.predict(X) * 0.4 + knn.predict(X) * 0.2)
        
        caminho_csv = self.camadas["ouro"] / "base_looker.csv"
        base_h3.to_csv(caminho_csv, index=False)
        
        with open(caminho_csv, "rb") as f:
            assinatura = hashlib.sha256(f.read()).hexdigest()
        with open(self.camadas["ouro"] / "base_looker.sha256", "w") as f:
            f.write(assinatura)
            
        return len(base_h3)

    def executar(self):
        acumulado = []
        for ano in self.anos:
            dados = self.gerenciar_delta(ano)
            if dados is not None:
                dados.columns = [c.lower() for c in dados.columns]
                dados['peso'] = dados['rubrica'].apply(lambda x: 15 if 'ROUBO' in str(x).upper() else 2)
                dados = self.recuperar_geo(dados)
                dados = dados.dropna(subset=['latitude', 'longitude'])
                dados.to_parquet(self.camadas["prata"] / f"prata_{ano}.parquet", index=False)
                acumulado.append(dados)

        if acumulado:
            total_h3 = self.gerar_ia_e_assinatura(pd.concat(acumulado))
            msg_op = "\n".join(self.logs_operacionais)
            self.notificar("Relatório Operacional", f"**Engenharia de Dados:**\n{msg_op}", 3447003)
            self.notificar("Relatório Executivo", f"🚀 **IA SafeDriver:**\n📍 Hexágonos Ativos: {total_h3}\n🛡️ Base Assinada Digitalmente", 3066993)

if __name__ == "__main__":
    MotorSafeDriver().executar()
