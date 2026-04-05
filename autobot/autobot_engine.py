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
        self.agente = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        self.geolocalizador = Nominatim(user_agent="safedriver_industrial_v6")
        self.historico = []

    def notificar_discord(self, titulo, mensagem, cor):
        webhook = os.environ.get("DISCORD_SUCESSO")
        if not webhook: return
        payload = {
            "embeds": [{
                "title": titulo,
                "description": mensagem,
                "color": cor,
                "footer": {"text": f"Módulo: {datetime.now().strftime('%H:%M')}"}
            }]
        }
        requests.post(webhook, json=payload, timeout=15)

    def classificar_perfil(self, df):
        # Busca a melhor coluna para a natureza do crime
        col_natureza = next((c for c in ['natureza_apurada', 'rubrica', 'descr_conduta'] if c in df.columns), None)
        col_local = next((c for c in ['descr_tipolocal', 'descr_local'] if c in df.columns), None)
        
        if not col_natureza:
            df['perfil'] = 'Geral'
            df['peso'] = 1.0
            return df

        df['natureza_norm'] = df[col_natureza].fillna('').astype(str).upper()
        df['local_norm'] = df[col_local].fillna('').astype(str).upper() if col_local else ""
        
        df['perfil'] = 'Geral'
        df['peso'] = 2.0 # Peso padrão (Furto/Outros)

        # Lógica de perfis e pesos (Roubo = 15)
        moto_mask = df['natureza_norm'].str.contains('VEÍCULO|CARGA|AUTO|MOTO|CONDUZIR')
        df.loc[moto_mask, 'perfil'] = 'Motorista'
        df.loc[moto_mask, 'peso'] = 15.0
        
        bike_mask = df['natureza_norm'].str.contains('BICICLETA|BIKE')
        df.loc[bike_mask, 'perfil'] = 'Ciclista'
        df.loc[bike_mask, 'peso'] = 15.0
        
        ped_mask = df['natureza_norm'].str.contains('CELULAR|TRANSEUNTE|PESSOA')
        df.loc[ped_mask, 'perfil'] = 'Pedestre'
        df.loc[ped_mask, 'peso'] = 15.0
        
        return df

    def geocodificar_lacunas(self, df):
        alvos = df[df['latitude'].isna() | (df['latitude'] == 0) | (df['latitude'] == "0")].head(15)
        if alvos.empty: return df
        for i, linha in alvos.iterrows():
            endereco = f"{linha.get('logradouro', '')}, {linha.get('numero_logradouro', '')}, SP, Brasil"
            try:
                local = self.geolocalizador.geocode(endereco, timeout=10)
                if local:
                    df.at[i, 'latitude'] = local.latitude
                    df.at[i, 'longitude'] = local.longitude
            except: continue
        return df

    def sincronizar_bronze(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        caminho_bruto = self.camadas["bronze"] / f"bruto_{ano}.parquet"
        caminho_meta = self.camadas["bronze"] / f"size_{ano}.txt"

        try:
            head = requests.head(url, headers=self.agente, timeout=20)
            tamanho_ssp = str(head.headers.get('Content-Length', '0'))

            if caminho_bruto.exists() and caminho_meta.exists():
                with open(caminho_meta, "r") as f:
                    if f.read() == tamanho_ssp:
                        self.historico.append(f"🟢 {ano}: Camada Bronze atualizada")
                        return pd.read_parquet(caminho_bruto)

            res = requests.get(url, headers=self.agente, timeout=300)
            df = pd.read_excel(io.BytesIO(res.content))
            df.columns = [str(c).upper().strip() for c in df.columns]
            df.to_parquet(caminho_bruto, index=False)
            with open(caminho_meta, "w") as f: f.write(tamanho_ssp)
            self.historico.append(f"📥 {ano}: Novos dados processados")
            return df
        except:
            return pd.read_parquet(caminho_bruto) if caminho_bruto.exists() else None

    def gerar_ouro(self, df):
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)]
        
        if df.empty: return 0

        df['h3_index'] = df.apply(lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1)
        resumo = df.groupby(['h3_index', 'desc_periodo', 'perfil'])['peso'].sum().reset_index()
        
        resumo['lat'] = resumo['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        resumo['lon'] = resumo['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        resumo['perfil_idx'] = resumo['perfil'].astype('category').cat.codes
        resumo['periodo_idx'] = resumo['desc_periodo'].astype('category').cat.codes

        X, y = resumo[['lat', 'lon', 'perfil_idx', 'periodo_idx']], resumo['peso']
        modelo = LGBMRegressor(n_estimators=100, verbose=-1).fit(X, y)
        
        plt.figure()
        shap.summary_plot(shap.Explainer(modelo, X)(X), X, show=False)
        plt.savefig(self.camadas["docs"] / "explicabilidade_ia.png", bbox_inches='tight')
        plt.close()

        resumo['score_risco'] = modelo.predict(X)
        caminho_csv = self.camadas["ouro"] / "base_looker.csv"
        resumo.to_csv(caminho_csv, index=False)
        
        assinatura = hashlib.sha256(open(caminho_csv, "rb").read()).hexdigest()
        with open(self.camadas["ouro"] / "base_looker.sha256", "w") as f:
            f.write(assinatura)
        return len(resumo)

    def iniciar(self):
        colecao = []
        for ano in self.anos:
            dados = self.sincronizar_bronze(ano)
            if dados is not None:
                dados.columns = [c.lower() for c in dados.columns]
                dados = self.classificar_perfil(dados)
                dados = self.geocodificar_lacunas(dados)
                colecao.append(dados)

        if colecao:
            total = self.gerar_ouro(pd.concat(colecao))
            msg = "\n".join(self.historico) + f"\n📍 Células H3: {total}"
            self.notificar_discord("SafeDriver: Pipeline Executado", msg, 3066993)
        else:
            self.notificar_discord("SafeDriver: Falha", "Sem dados capturados.", 15158332)

if __name__ == "__main__":
    MotorSafeDriver().iniciar()
