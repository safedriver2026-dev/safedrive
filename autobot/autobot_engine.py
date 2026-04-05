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
        self.cabecalhos = {'User-Agent': 'SafeDriver-Industrial-V5'}
        self.geolocalizador = Nominatim(user_agent="safedriver_fatec_v5")
        self.logs = []

    def disparar_notificacao(self, titulo, mensagem, cor):
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

    def classificar_perfil(self, df):
        df['natureza_str'] = df['natureza_apurada'].fillna(df.get('rubrica', '')).astype(str).upper()
        df['local_str'] = df['descr_tipolocal'].fillna('').astype(str).upper()
        
        df['perfil'] = 'Geral'
        
        moto_mask = df['natureza_str'].str.contains('VEÍCULO|CARGA|AUTO|MOTO|CONDUZIR')
        df.loc[moto_mask, 'perfil'] = 'Motorista'
        
        bike_mask = df['natureza_str'].str.contains('BICICLETA|BIKE')
        df.loc[bike_mask, 'perfil'] = 'Ciclista'
        
        ped_mask = (df['local_str'].str.contains('VIA PÚBLICA|RUA|AVENIDA')) & \
                   (df['natureza_str'].str.contains('CELULAR|TRANSEUNTE|PESSOA'))
        df.loc[ped_mask, 'perfil'] = 'Pedestre'
        
        return df

    def gerenciar_delta(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        bronze_file = self.camadas["bronze"] / f"bruto_{ano}.parquet"
        hash_file = self.camadas["bronze"] / f"size_{ano}.txt"

        try:
            head = requests.head(url, headers=self.cabecalhos, timeout=30)
            tamanho_nuvem = str(head.headers.get('Content-Length', '0'))

            if bronze_file.exists() and hash_file.exists():
                with open(hash_file, "r") as f:
                    if f.read() == tamanho_nuvem:
                        self.logs.append(f"🟢 {ano}: Camada Bronze sincronizada")
                        return pd.read_parquet(bronze_file)

            res = requests.get(url, headers=self.cabecalhos, timeout=300)
            df = pd.read_excel(io.BytesIO(res.content))
            df.columns = [str(c).upper().strip() for c in df.columns]
            df.to_parquet(bronze_file, index=False)
            with open(hash_file, "w") as f: f.write(tamanho_nuvem)
            self.logs.append(f"📥 {ano}: Captura de novo volume concluída")
            return df
        except:
            return pd.read_parquet(bronze_file) if bronze_file.exists() else None

    def executar_ia(self, df):
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
        cat = CatBoostRegressor(n_estimators=100, verbose=0).fit(X, y)
        
        shap.summary_plot(shap.Explainer(lgb, X)(X), X, show=False)
        plt.savefig(self.camadas["docs"] / "explicabilidade_ia.png", bbox_inches='tight')
        plt.close()

        resumo['score_risco'] = (lgb.predict(X) * 0.5 + cat.predict(X) * 0.5)
        
        ouro_path = self.camadas["ouro"] / "base_looker.csv"
        resumo.to_csv(ouro_path, index=False)
        
        with open(ouro_path, "rb") as f:
            check = hashlib.sha256(f.read()).hexdigest()
        with open(self.camadas["ouro"] / "base_looker.sha256", "w") as f:
            f.write(check)
            
        return len(resumo)

    def iniciar(self):
        acumulado = []
        for ano in self.anos:
            dados = self.gerenciar_delta(ano)
            if dados is not None:
                dados.columns = [c.lower() for c in dados.columns]
                dados = self.classificar_perfil(dados)
                dados['peso'] = dados['natureza_str'].apply(lambda x: 15 if 'ROUBO' in str(x) else 2)
                acumulado.append(dados)

        if acumulado:
            total_h3 = self.executar_ia(pd.concat(acumulado))
            msg = "\n".join(self.logs) + f"\n📍 Células Ativas: {total_h3}"
            self.disparar_notificacao("SafeDriver Core: Pipeline OK", msg, 3066993)
        else:
            self.disparar_notificacao("SafeDriver: Falha de Ingestão", "Nenhum dado capturado.", 15158332)

if __name__ == "__main__":
    MotorSafeDriver().iniciar()
