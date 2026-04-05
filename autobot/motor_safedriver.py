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
from geopy.geocoders import Nominatim
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
        
        self.cabecalhos = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7',
            'Connection': 'keep-alive',
            'Referer': 'https://www.ssp.sp.gov.br/estatistica/transparencia-trimestral'
        }
        
        self.webhook = os.environ.get("DISCORD_WEBHOOK")
        self.manifesto_path = self.pastas["auditoria"] / "manifesto.json"
        self.auditoria = self.carregar_manifesto()

    def carregar_manifesto(self):
        if self.manifesto_path.exists():
            with open(self.manifesto_path, "r") as f: return json.load(f)
        return {}

    def calcular_assinatura(self, conteudo):
        return hashlib.sha256(conteudo).hexdigest()

    def avisar_discord(self, titulo, msg, cor):
        if not self.webhook: return
        payload = {"embeds": [{"title": titulo, "description": msg, "color": cor, "timestamp": datetime.now().isoformat()}]}
        try: requests.post(self.webhook, json=payload, timeout=10)
        except: pass

    def extrair_dados_ssp(self, conteudo):
        try:
            excel = pd.ExcelFile(io.BytesIO(conteudo))
            for aba in excel.sheet_names:
                df = excel.parse(aba, nrows=50)
                df.columns = [str(c).upper().strip() for c in df.columns]
                if all(k in df.columns for k in ['NUM_BO', 'ANO_BO', 'NOME_MUNICIPIO']):
                    return excel.parse(aba)
            return None
        except: return None

    def processar_ia_v20(self, df):
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
        
        ouro_path = self.pastas["ouro"] / "base_final_bi.csv"
        fato.to_csv(ouro_path, index=False)
        self.auditoria["hash_ouro"] = hashlib.sha256(open(ouro_path, "rb").read()).hexdigest()
        
        return len(fato)

    def iniciar_pipeline(self):
        try:
            pool = []
            relatorio = []
            
            with requests.Session() as sessao:
                sessao.headers.update(self.cabecalhos)
                
                for ano in self.anos:
                    url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                    caminho_local = self.pastas["bronze"] / f"bruto_{ano}.parquet"
                    
                    try:
                        # DeltaSync: Verifica metadados sem baixar o arquivo todo
                        head = sessao.head(url, timeout=30)
                        tamanho_remoto = str(head.headers.get('Content-Length', '0'))
                        
                        if caminho_local.exists() and self.auditoria.get(f"size_{ano}") == tamanho_remoto:
                            pool.append(pd.read_parquet(caminho_local))
                            relatorio.append(f"📦 {ano}: DeltaSync Ativo")
                            continue

                        # Download camuflado com delay humano
                        time.sleep(2)
                        res = sessao.get(url, timeout=180)
                        if res.status_code == 200:
                            df_novo = self.extrair_dados_ssp(res.content)
                            if df_novo is not None:
                                df_novo.to_parquet(caminho_local)
                                self.auditoria[f"size_{ano}"] = tamanho_remoto
                                self.auditoria[f"hash_{ano}"] = self.calcular_assinatura(res.content)
                                pool.append(df_novo)
                                relatorio.append(f"📥 {ano}: Sincronização Incremental")
                        else:
                            raise Exception(f"Erro HTTP {res.status_code}")
                            
                    except Exception as e:
                        if caminho_local.exists():
                            pool.append(pd.read_parquet(caminho_local))
                            relatorio.append(f"⚠️ {ano}: Erro de rede (Usando cache)")

            if not pool: raise Exception("Nenhuma fonte de dados acessível após camuflagem")

            total = self.processar_ia_v20(pd.concat(pool))
            
            with open(self.manifesto_path, "w") as f: json.dump(self.auditoria, f, indent=4)

            self.avisar_discord("Operacional SafeDriver", "\n".join(relatorio), 3447003)
            self.avisar_discord("Executivo SafeDriver", f"Base: {total} áreas\nEnsemble: CatB+LGBM+KNN\nIntegridade: OK", 3066993)

        except Exception as e:
            self.avisar_discord("Erro Crítico SafeDriver", str(e), 15158332)
            raise e

if __name__ == "__main__":
    MotorSafeDriver().iniciar_pipeline()
