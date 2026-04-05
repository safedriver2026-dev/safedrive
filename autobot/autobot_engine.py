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
        self.agente = {'User-Agent': 'SafeDriver-Industrial-V6'}
        self.geolocalizador = Nominatim(user_agent="safedriver_fatec_final")
        self.logs = []

    def disparar_notificacao(self, titulo, mensagem, cor):
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

    def classificar_vetor_exposicao(self, df):
        df['natureza_alvo'] = df['natureza_apurada'].fillna(df.get('rubrica', '')).astype(str).upper()
        df['local_alvo'] = df['descr_tipolocal'].fillna('').astype(str).upper()
        
        df['perfil'] = 'Geral'
        
        motorista = df['natureza_alvo'].str.contains('VEÍCULO|CARGA|AUTO|MOTO|CONDUZIR')
        df.loc[motorista, 'perfil'] = 'Motorista'
        
        ciclista = df['natureza_alvo'].str.contains('BICICLETA|BIKE')
        df.loc[ciclista, 'perfil'] = 'Ciclista'
        
        pedestre = (df['local_alvo'].str.contains('VIA PÚBLICA|RUA|AVENIDA')) & \
                   (df['natureza_alvo'].str.contains('CELULAR|TRANSEUNTE|PESSOA'))
        df.loc[pedestre, 'perfil'] = 'Pedestre'
        
        return df

    def geocodificar_lacunas(self, df):
        faltantes = df[df['latitude'].isna() | (df['latitude'] == 0) | (df['latitude'] == "0")].head(20)
        if faltantes.empty: return df
        for i, linha in faltantes.iterrows():
            endereco = f"{linha.get('logradouro', '')}, {linha.get('numero_logradouro', '')}, {linha.get('nome_municipio', 'São Bernardo do Campo')}, Brasil"
            try:
                local = self.geolocalizador.geocode(endereco, timeout=10)
                if local:
                    df.at[i, 'latitude'] = local.latitude
                    df.at[i, 'longitude'] = local.longitude
            except:
                continue
        return df

    def sincronizar_bronze(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        caminho_bruto = self.camadas["bronze"] / f"bruto_{ano}.parquet"
        caminho_meta = self.camadas["bronze"] / f"size_{ano}.txt"

        try:
            head = requests.head(url, headers=self.agente, timeout=30)
            tamanho_ssp = str(head.headers.get('Content-Length', '0'))

            if caminho_bruto.exists() and caminho_meta.exists():
                with open(caminho_meta, "r") as f:
                    if f.read() == tamanho_ssp:
                        self.logs.append(f"🟢 {ano}: Sincronizado")
                        return pd.read_parquet(caminho_bruto)

            res = requests.get(url, headers=self.agente, timeout=300)
            df = pd.read_excel(io.BytesIO(res.content))
            df.columns = [str(c).upper().strip() for c in df.columns]
            df.to_parquet(caminho_bruto, index=False)
            with open(caminho_meta, "w") as f: f.write(tamanho_ssp)
            self.logs.append(f"📥 {ano}: Ingestão Delta concluída")
            return df
        except:
            return pd.read_parquet(caminho_bruto) if caminho_bruto.exists() else None

    def processar_ia_ouro(self, df):
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
        
        if df.empty: return 0

        df['h3_index'] = df.apply(lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1)
        
        agrupado = df.groupby(['h3_index', 'desc_periodo', 'perfil'])['peso'].sum().reset_index()
        agrupado['lat'] = agrupado['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        agrupado['lon'] = agrupado['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        agrupado['perfil_idx'] = agrupado['perfil'].astype('category').cat.codes
        agrupado['periodo_idx'] = agrupado['desc_periodo'].astype('category').cat.codes

        X, y = agrupado[['lat', 'lon', 'perfil_idx', 'periodo_idx']], agrupado['peso']
        
        lgb = LGBMRegressor(n_estimators=100, verbose=-1).fit(X, y)
        cat = CatBoostRegressor(n_estimators=100, verbose=0).fit(X, y)
        
        shap.summary_plot(shap.Explainer(lgb, X)(X), X, show=False)
        plt.savefig(self.camadas["docs"] / "explicabilidade_ia.png", bbox_inches='tight')
        plt.close()

        agrupado['score_risco'] = (lgb.predict(X) * 0.5 + cat.predict(X) * 0.5)
        
        caminho_csv = self.camadas["ouro"] / "base_looker.csv"
        agrupado.to_csv(caminho_csv, index=False)
        
        assinatura = hashlib.sha256(open(caminho_csv, "rb").read()).hexdigest()
        with open(self.camadas["ouro"] / "base_looker.sha256", "w") as f:
            f.write(assinatura)
            
        return len(agrupado)

    def iniciar(self):
        coleta = []
        for ano in self.anos:
            dados = self.sincronizar_bronze(ano)
            if dados is not None:
                dados.columns = [c.lower() for c in dados.columns]
                dados = self.classificar_vetor_exposicao(dados)
                dados['peso'] = dados['natureza_alvo'].apply(lambda x: 15 if 'ROUBO' in str(x) else 2)
                dados = self.geocodificar_lacunas(dados)
                coleta.append(dados)

        if coleta:
            total = self.processar_ia_ouro(pd.concat(coleta))
            msg = "\n".join(self.logs) + f"\n📍 Células de Risco: {total}"
            self.disparar_notificacao("SafeDriver: Pipeline Executado", msg, 3066993)
        else:
            self.disparar_notificacao("SafeDriver: Falha", "Nenhum dado capturado.", 15158332)

if __name__ == "__main__":
    MotorSafeDriver().iniciar()
