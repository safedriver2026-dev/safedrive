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
        self.agente = {'User-Agent': 'SafeDriver-Industrial-V8'}
        self.geolocalizador = Nominatim(user_agent="safedriver_fatec_final")
        self.logs = []

    def disparar_discord(self, titulo, mensagem, cor):
        webhook = os.environ.get("DISCORD_SUCESSO")
        if not webhook: return
        payload = {
            "embeds": [{
                "title": titulo,
                "description": mensagem,
                "color": cor,
                "footer": {"text": f"Sincronização: {datetime.now().strftime('%H:%M')}"}
            }]
        }
        requests.post(webhook, json=payload, timeout=15)

    def localizar_folha_dados(self, conteudo):
        excel = pd.ExcelFile(io.BytesIO(conteudo))
        for nome_aba in excel.sheet_names:
            df_teste = excel.parse(nome_aba, nrows=5)
            df_teste.columns = [str(c).upper().strip() for c in df_teste.columns]
            if 'LATITUDE' in df_teste.columns or 'RUBRICA' in df_teste.columns:
                return excel.parse(nome_aba)
        return None

    def classificar_perfis(self, df):
        df.columns = [str(c).lower().strip() for c in df.columns]
        col_nat = next((c for c in ['natureza_apurada', 'rubrica'] if c in df.columns), None)
        df['perfil'] = 'Geral'
        if col_nat:
            df['natureza_upper'] = df[col_nat].fillna('').astype(str).upper()
            df.loc[df['natureza_upper'].str.contains('VEÍCULO|CARGA|AUTO|MOTO'), 'perfil'] = 'Motorista'
            df.loc[df['natureza_upper'].str.contains('BICICLETA|BIKE'), 'perfil'] = 'Ciclista'
            if 'descr_tipolocal' in df.columns:
                loc_upper = df['descr_tipolocal'].fillna('').astype(str).upper()
                ped_mask = (loc_upper.str.contains('VIA PÚBLICA|RUA')) & (df['natureza_upper'].str.contains('CELULAR|PESSOA'))
                df.loc[ped_mask, 'perfil'] = 'Pedestre'
            df['peso'] = df['natureza_upper'].apply(lambda x: 15 if 'ROUBO' in x else 2)
        return df

    def geocodificar_falhas(self, df):
        if 'latitude' not in df.columns: return df
        faltas = df[df['latitude'].isna() | (df['latitude'] == 0) | (df['latitude'] == "0")].head(15)
        for i, linha in faltas.iterrows():
            endereco = f"{linha.get('logradouro', '')}, {linha.get('numero_logradouro', '')}, SP, Brasil"
            try:
                local = self.geolocalizador.geocode(endereco, timeout=10)
                if local:
                    df.at[i, 'latitude'], df.at[i, 'longitude'] = local.latitude, local.longitude
            except: continue
        return df

    def sincronizar_bronze(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        caminho_bruto = self.camadas["bronze"] / f"bruto_{ano}.parquet"
        caminho_meta = self.camadas["bronze"] / f"size_{ano}.txt"
        try:
            head = requests.head(url, headers=self.agente, timeout=30)
            tam_nuvem = str(head.headers.get('Content-Length', '0'))
            if caminho_bruto.exists() and caminho_meta.exists():
                with open(caminho_meta, "r") as f:
                    if f.read() == tam_nuvem:
                        self.logs.append(f"🟢 {ano}: Camada Bronze atualizada")
                        return pd.read_parquet(caminho_bruto)
            res = requests.get(url, headers=self.agente, timeout=300)
            df = self.localizar_folha_dados(res.content)
            if df is not None:
                df.columns = [str(c).upper().strip() for c in df.columns]
                df.to_parquet(caminho_bruto, index=False)
                with open(caminho_meta, "w") as f: f.write(tam_nuvem)
                self.logs.append(f"📥 {ano}: Ingestão concluída")
                return df
        except Exception as e:
            self.logs.append(f"🔥 {ano}: Erro - {str(e)}")
        return pd.read_parquet(caminho_bruto) if caminho_bruto.exists() else None

    def gerar_ia_explicavel(self, df):
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
        lgb = LGBMRegressor(n_estimators=100, verbose=-1).fit(X, y)
        
        # Correção Crítica do SHAP
        plt.figure(figsize=(10, 6))
        explainer = shap.TreeExplainer(lgb)
        shap_values = explainer.shap_values(X)
        
        shap.summary_plot(shap_values, X, show=False, plot_type="bar")
        plt.title("Fatores Determinantes do Risco (SHAP)")
        plt.savefig(self.camadas["docs"] / "explicabilidade_ia.png", bbox_inches='tight')
        plt.close()

        resumo['score_risco'] = lgb.predict(X)
        caminho_csv = self.camadas["ouro"] / "base_looker.csv"
        resumo.to_csv(caminho_csv, index=False)
        
        hash_val = hashlib.sha256(open(caminho_csv, "rb").read()).hexdigest()
        with open(self.camadas["ouro"] / "base_looker.sha256", "w") as f: f.write(hash_val)
        return len(resumo)

    def iniciar(self):
        colecao = []
        for ano in self.anos:
            dados = self.sincronizar_bronze(ano)
            if dados is not None:
                dados = self.classificar_perfis(dados)
                dados = self.geocodificar_falhas(dados)
                colecao.append(dados)
        if colecao:
            total = self.gerar_ia_explicavel(pd.concat(colecao))
            msg = "\n".join(self.logs) + f"\n📍 Células de Risco: {total}"
            self.disparar_discord("SafeDriver: Pipeline e SHAP OK", msg, 3066993)
        else:
            self.disparar_discord("SafeDriver: Falha", "Sem dados capturados.", 15158332)

if __name__ == "__main__":
    MotorSafeDriver().iniciar()
