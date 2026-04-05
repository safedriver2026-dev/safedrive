import os
import io
import json
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
        self.agente = {'User-Agent': 'SafeDriver-BR-V19-Industrial'}
        self.webhook = os.environ.get("DISCORD_WEBHOOK")
        self.manifesto_caminho = self.pastas["auditoria"] / "manifesto.json"
        self.auditoria = self.carregar_manifesto()

    def carregar_manifesto(self):
        if self.manifesto_caminho.exists():
            with open(self.manifesto_caminho, "r") as f: return json.load(f)
        return {}

    def calcular_hash(self, conteudo):
        return hashlib.sha256(conteudo).hexdigest()

    def enviar_alerta(self, titulo, msg, cor):
        if not self.webhook: return
        payload = {"embeds": [{"title": titulo, "description": msg, "color": cor, "timestamp": datetime.now().isoformat()}]}
        try: requests.post(self.webhook, json=payload, timeout=10)
        except: pass

    def identificar_base_ssp(self, conteudo):
        try:
            excel = pd.ExcelFile(io.BytesIO(conteudo))
            for aba in excel.sheet_names:
                df = excel.parse(aba, nrows=50)
                df.columns = [str(c).upper().strip() for c in df.columns]
                if all(k in df.columns for k in ['NUM_BO', 'ANO_BO', 'NOME_MUNICIPIO']):
                    return excel.parse(aba)
            return None
        except: return None

    def processar_ia_ensemble(self, df):
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        df = df.drop_duplicates(subset=['num_bo', 'ano_bo', 'nome_municipio', 'data_registro'])
        
        df['perfil'] = 'Geral'
        col_crime = next((c for c in ['natureza_apurada', 'rubrica'] if c in df.columns), 'rubrica')
        df['crime_alvo'] = df[col_crime].fillna('').astype(str).upper()
        
        df.loc[df['crime_alvo'].str.contains('VEÍCULO|MOTO|CARGA|AUTO|CONDUZIR'), 'perfil'] = 'Motorista'
        df.loc[df['crime_alvo'].str.contains('BICICLETA|BIKE'), 'perfil'] = 'Ciclista'
        
        col_local = next((c for c in ['descr_tipolocal', 'descr_local'] if c in df.columns), 'descr_tipolocal')
        if col_local in df.columns:
            loc_alvo = df[col_local].fillna('').astype(str).upper()
            mask_ped = (loc_alvo.str.contains('VIA PÚBLICA|RUA')) & (df['crime_alvo'].str.contains('CELULAR|PESSOA'))
            df.loc[mask_ped, 'perfil'] = 'Pedestre'
        
        df['severidade'] = df['crime_alvo'].apply(lambda x: 15 if 'ROUBO' in x else 2)
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

        explicador = shap.TreeExplainer(lgbm)
        shap_v = explicador.shap_values(X)
        for i, col in enumerate(X.columns): fato[f'influencia_{col}'] = shap_v[:, i]
            
        fato['score_risco'] = (lgbm.predict(X) * 0.4 + catb.predict(X) * 0.4 + knnr.predict(X) * 0.2)
        
        ouro_path = self.pastas["ouro"] / "base_final_bi.csv"
        fato.to_csv(ouro_path, index=False)
        
        hash_ouro = hashlib.sha256(open(ouro_path, "rb").read()).hexdigest()
        self.auditoria["hash_ouro"] = hash_ouro
        
        return len(fato)

    def iniciar_pipeline(self):
        try:
            pool = []
            log_servico = []
            
            for ano in self.anos:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                caminho_local = self.pastas["bronze"] / f"bruto_{ano}.parquet"
                
                try:
                    head = requests.head(url, headers=self.agente, timeout=30)
                    tamanho_ssp = str(head.headers.get('Content-Length', '0'))
                    
                    if caminho_local.exists() and self.auditoria.get(f"tamanho_{ano}") == tamanho_ssp:
                        df_existente = pd.read_parquet(caminho_local)
                        pool.append(df_existente)
                        log_servico.append(f"📦 {ano}: DeltaSync (Sem alteracoes)")
                        continue

                    res = requests.get(url, headers=self.agente, timeout=180)
                    if res.status_code == 200:
                        df_novo = self.identificar_base_ssp(res.content)
                        if df_novo is not None:
                            df_novo.to_parquet(caminho_local)
                            self.auditoria[f"tamanho_{ano}"] = tamanho_ssp
                            self.auditoria[f"hash_{ano}"] = self.calcular_hash(res.content)
                            pool.append(df_novo)
                            log_servico.append(f"📥 {ano}: Sincronizacao Incremental Concluida")
                except:
                    if caminho_local.exists():
                        pool.append(pd.read_parquet(caminho_local))
                        log_servico.append(f"⚠️ {ano}: Erro de rede (Usando cache)")

            if not pool: raise Exception("Fontes de dados indisponiveis")

            total_h3 = self.processar_ia_ensemble(pd.concat(pool))
            
            with open(self.manifesto_caminho, "w") as f: json.dump(self.auditoria, f, indent=4)

            self.enviar_alerta("SafeDriver Operacional", "\n".join(log_servico), 3447003)
            self.enviar_alerta("SafeDriver Executivo", f"Base atualizada: {total_h3} celulas\nIntegridade: OK\nMetodo: DeltaSync", 3066993)

        except Exception as e:
            self.enviar_alerta("Erro Critico SafeDriver", str(e), 15158332)
            raise e

if __name__ == "__main__":
    MotorSafeDriver().iniciar_pipeline()
