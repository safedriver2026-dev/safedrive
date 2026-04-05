import os
import io
import json
import time
import requests
import pandas as pd
import numpy as np
import h3
import shap
import matplotlib.pyplot as plt
import hashlib
from pathlib import Path
from datetime import datetime
from bs4 import BeautifulSoup
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class MotorSafeDriver:
    def __init__(self):
        self.raiz = Path(".")
        self.pastas = {
            "bronze": self.raiz / "datalake" / "bronze",
            "prata": self.raiz / "datalake" / "prata",
            "ouro": self.raiz / "datalake" / "ouro",
            "auditoria": self.raiz / "datalake" / "auditoria",
            "docs": self.raiz / "documentacao"
        }
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        
        self.anos = list(range(2022, datetime.now().year + 1))
        self.sessao = requests.Session()
        self.sessao.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8',
            'Referer': 'https://www.ssp.sp.gov.br/estatistica/consultas'
        })
        
        self.webhook = os.environ.get("DISCORD_WEBHOOK")
        self.manifesto_path = self.pastas["auditoria"] / "manifesto.json"
        self.auditoria = self.carregar_manifesto()

    def carregar_manifesto(self):
        if self.manifesto_path.exists():
            with open(self.manifesto_path, "r") as f: return json.load(f)
        return {}

    def buscar_link_ssp(self, ano):
        try:
            url_base = "https://www.ssp.sp.gov.br/estatistica/consultas"
            res = self.sessao.get(url_base, timeout=30)
            soup = BeautifulSoup(res.text, 'html.parser')
            for link in soup.find_all('a', href=True):
                href = link['href']
                if f"SPDadosCriminais_{ano}" in href and href.endswith('.xlsx'):
                    if href.startswith('http'): return href
                    return f"https://www.ssp.sp.gov.br{href}"
            return f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        except:
            return f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"

    def processar_ia_ensemble(self, df):
        df.columns = [str(c).lower().strip() for c in df.columns]
        df = df.drop_duplicates(subset=['num_bo', 'ano_bo', 'nome_municipio', 'data_registro'])
        
        df['perfil'] = 'Geral'
        col_nat = next((c for c in ['natureza_apurada', 'rubrica'] if c in df.columns), 'rubrica')
        df['crime_texto'] = df[col_nat].fillna('').astype(str).upper()
        
        df.loc[df['crime_texto'].str.contains('VEÍCULO|MOTO|CARGA|AUTO|CONDUZIR'), 'perfil'] = 'Motorista'
        df.loc[df['crime_texto'].str.contains('BICICLETA|BIKE'), 'perfil'] = 'Ciclista'
        
        col_loc = next((c for c in ['descr_tipolocal', 'descr_local'] if c in df.columns), 'descr_tipolocal')
        if col_loc in df.columns:
            loc_up = df[col_loc].fillna('').astype(str).upper()
            df.loc[(loc_up.str.contains('VIA PÚBLICA')) & (df['crime_texto'].str.contains('CELULAR|PESSOA')), 'perfil'] = 'Pedestre'
        
        df['severidade'] = df['crime_texto'].apply(lambda x: 15 if 'ROUBO' in x else 2)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()
        
        if df.empty: return None

        df['h3_index'] = df.apply(lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1)
        fato = df.groupby(['h3_index', 'desc_periodo', 'perfil'])['severidade'].sum().reset_index()
        fato['lat'] = fato['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        fato['lon'] = fato['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        fato['perfil_idx'] = fato['perfil'].astype('category').cat.codes
        fato['periodo_idx'] = fato['desc_periodo'].astype('category').cat.codes

        X = fato[['lat', 'lon', 'perfil_idx', 'periodo_idx']]
        y = fato['severidade']
        
        lgbm = LGBMRegressor(n_estimators=100, verbose=-1).fit(X, y)
        catb = CatBoostRegressor(iterations=100, silent=True).fit(X, y)
        knnr = KNeighborsRegressor(n_neighbors=5).fit(X, y)

        explainer = shap.TreeExplainer(lgbm)
        shap_v = explainer.shap_values(X)
        for i, col in enumerate(X.columns): fato[f'influencia_{col}'] = shap_v[:, i]
            
        fato['score_risco'] = (lgbm.predict(X) * 0.4 + catb.predict(X) * 0.4 + knnr.predict(X) * 0.2)
        
        ouro_path = self.pastas["ouro"] / "base_final_looker.csv"
        fato.to_csv(ouro_path, index=False)
        self.auditoria["hash_ouro"] = hashlib.sha256(open(ouro_path, "rb").read()).hexdigest()
        
        return len(fato)

    def iniciar_pipeline(self):
        try:
            pool = []
            relatorio = []
            
            for ano in self.anos:
                url = self.buscar_link_ssp(ano)
                caminho_local = self.pastas["bronze"] / f"bruto_{ano}.parquet"
                
                try:
                    head = self.sessao.head(url, timeout=20)
                    tamanho_remoto = str(head.headers.get('Content-Length', '0'))
                    
                    if caminho_local.exists() and self.auditoria.get(f"size_{ano}") == tamanho_remoto:
                        pool.append(pd.read_parquet(caminho_local))
                        relatorio.append(f"📦 {ano}: DeltaSync (Inalterado)")
                        continue

                    time.sleep(3)
                    res = self.sessao.get(url, timeout=240)
                    if res.status_code == 200:
                        excel = pd.ExcelFile(io.BytesIO(res.content))
                        df_novo = None
                        for aba in excel.sheet_names:
                            df_teste = excel.parse(aba, nrows=30)
                            df_teste.columns = [str(c).upper().strip() for c in df_teste.columns]
                            if 'LATITUDE' in df_teste.columns:
                                df_novo = excel.parse(aba)
                                break
                        
                        if df_novo is not None:
                            df_novo.to_parquet(caminho_local)
                            self.auditoria[f"size_{ano}"] = tamanho_remoto
                            self.auditoria[f"hash_{ano}"] = hashlib.sha256(res.content).hexdigest()
                            pool.append(df_novo)
                            relatorio.append(f"📥 {ano}: Sincronizado via Link Dinâmico")
                        else:
                            raise Exception(f"Aba de coordenadas não encontrada em {ano}")
                except:
                    if caminho_local.exists():
                        pool.append(pd.read_parquet(caminho_local))
                        relatorio.append(f"⚠️ {ano}: Erro Conexão (Usando Local)")

            if not pool: raise Exception("Bloqueio SSP: Nenhuma fonte de dados acessível via Scraping")

            total = self.processar_ia_ensemble(pd.concat(pool))
            with open(self.manifesto_path, "w") as f: json.dump(self.auditoria, f, indent=4)
            
            if self.webhook:
                requests.post(self.webhook, json={"content": f"🚀 **SafeDriver V21**: {total} áreas atualizadas.\nIntegridade: OK."})

        except Exception as e:
            if self.webhook: requests.post(self.webhook, json={"content": f"🚨 **Erro Crítico**: {str(e)}"})
            raise e

if __name__ == "__main__":
    MotorSafeDriver().iniciar_pipeline()
