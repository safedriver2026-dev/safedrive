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
from lightgbm import LGBMRegressor

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
            
        ano_inicial = 2022
        self.anos = list(range(ano_inicial, datetime.now().year + 1))
        self.agente = {'User-Agent': 'SafeDriver-Industrial-V10-SHAP'}
        self.geolocalizador = Nominatim(user_agent="safedriver_final_engine")
        self.logs = []

    def disparar_discord(self, titulo, mensagem, cor):
        webhook = os.environ.get("DISCORD_SUCESSO")
        if not webhook: return
        payload = {"embeds": [{"title": titulo, "description": mensagem, "color": cor}]}
        requests.post(webhook, json=payload, timeout=15)

    def localizar_folha_dados(self, conteudo):
        excel = pd.ExcelFile(io.BytesIO(conteudo))
        for nome_aba in excel.sheet_names:
            df_teste = excel.parse(nome_aba, nrows=10)
            df_teste.columns = [str(c).upper().strip() for c in df_teste.columns]
            if any(col in df_teste.columns for col in ['LATITUDE', 'RUBRICA', 'NATUREZA_APURADA']):
                return excel.parse(nome_aba)
        return None

    def classificar_perfis(self, df):
        df.columns = [str(c).lower().strip() for c in df.columns]
        col_nat = next((c for c in ['natureza_apurada', 'rubrica', 'natureza'] if c in df.columns), None)
        df['perfil'] = 'Geral'
        if col_nat:
            df['nat_clean'] = df[col_nat].fillna('').astype(str).upper()
            df.loc[df['nat_clean'].str.contains('VEÍCULO|CARGA|AUTO|MOTO'), 'perfil'] = 'Motorista'
            df.loc[df['nat_clean'].str.contains('BICICLETA|BIKE'), 'perfil'] = 'Ciclista'
            col_loc = next((c for c in ['descr_tipolocal', 'descr_local'] if c in df.columns), None)
            if col_loc:
                loc_clean = df[col_loc].fillna('').astype(str).upper()
                ped_mask = (loc_clean.str.contains('VIA PÚBLICA|RUA')) & (df['nat_clean'].str.contains('CELULAR|PESSOA'))
                df.loc[ped_mask, 'perfil'] = 'Pedestre'
            df['peso'] = df['nat_clean'].apply(lambda x: 15 if 'ROUBO' in x else 2)
        return df

    def sincronizar_delta(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        bruto, meta = self.camadas["bronze"] / f"bruto_{ano}.parquet", self.camadas["bronze"] / f"size_{ano}.txt"
        try:
            head = requests.head(url, headers=self.agente, timeout=30)
            tam_remoto = str(head.headers.get('Content-Length', '0'))
            if bruto.exists() and meta.exists():
                with open(meta, "r") as f:
                    if f.read() == tam_remoto: return pd.read_parquet(bruto)
            res = requests.get(url, headers=self.agente, timeout=300)
            df = self.localizar_folha_dados(res.content)
            if df is not None:
                df.columns = [str(c).upper().strip() for c in df.columns]
                df.to_parquet(bruto, index=False)
                with open(meta, "w") as f: f.write(tam_remoto)
                return df
        except: return pd.read_parquet(bruto) if bruto.exists() else None

    def gerar_ouro_com_explicabilidade(self, df):
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
        
        X = resumo[['lat', 'lon', 'perfil_idx', 'periodo_idx']]
        y = resumo['peso']
        lgb = LGBMRegressor(n_estimators=100, verbose=-1).fit(X, y)
        
      
        explainer = shap.TreeExplainer(lgb)
        shap_values = explainer.shap_values(X)
        shap_df = pd.DataFrame(shap_values, columns=[f'shap_{c}' for c in X.columns])
        
       
        resumo = pd.concat([resumo.reset_index(drop=True), shap_df], axis=1)
        resumo['score_risco'] = lgb.predict(X)
        
        
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False, plot_type="bar")
        plt.savefig(self.camadas["docs"] / "explicabilidade_ia.png", bbox_inches='tight')
        plt.close()

        ouro_path = self.camadas["ouro"] / "base_looker.csv"
        resumo.to_csv(ouro_path, index=False)
        hash_val = hashlib.sha256(open(ouro_path, "rb").read()).hexdigest()
        with open(self.camadas["ouro"] / "base_looker.sha256", "w") as f: f.write(hash_val)
        return len(resumo)

    def iniciar(self):
        pool = []
        for ano in self.anos:
            dados = self.sincronizar_delta(ano)
            if dados is not None:
                pool.append(self.classificar_perfis(dados))
        if pool:
            total = self.gerar_ouro_com_explicabilidade(pd.concat(pool))
            self.disparar_discord("SafeDriver: IA Explicável Ativa", f"📍 {total} Células prontas para o Looker.", 3066993)

if __name__ == "__main__":
    MotorSafeDriver().iniciar()
