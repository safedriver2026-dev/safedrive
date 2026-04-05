import os
import io
import json
import time
import random
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
        
        self.webhook_sucesso = os.environ.get("DISCORD_SUCESSO")
        self.webhook_erro = os.environ.get("DISCORD_ERRO")
        
        self.agentes = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
        ]
        
        self.manifesto_path = self.pastas["auditoria"] / "manifesto.json"
        self.auditoria = self.carregar_manifesto()

    def carregar_manifesto(self):
        if self.manifesto_path.exists():
            with open(self.manifesto_path, "r") as f: return json.load(f)
        return {}

    def configurar_sessao(self):
        self.sessao.headers.update({
            'User-Agent': random.choice(self.agentes),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Referer': 'https://www.ssp.sp.gov.br/estatistica/consultas'
        })

    def enviar_notificacao(self, titulo, msg, cor, sucesso=True):
        webhook = self.webhook_sucesso if sucesso else self.webhook_erro
        if not webhook: return
        payload = {"embeds": [{"title": titulo, "description": msg, "color": cor, "timestamp": datetime.now().isoformat()}]}
        try: requests.post(webhook, json=payload, timeout=10)
        except: pass

    def extrair_dados_validos(self, conteudo):
        try:
            excel = pd.ExcelFile(io.BytesIO(conteudo))
            for aba in excel.sheet_names:
                df = excel.parse(aba, nrows=40)
                df.columns = [str(c).upper().strip() for c in df.columns]
                if all(k in df.columns for k in ['NUM_BO', 'ANO_BO', 'LATITUDE', 'LONGITUDE']):
                    return excel.parse(aba)
            return None
        except: return None

    def processar_ia_v22(self, df):
        df.columns = [str(c).lower().strip() for c in df.columns]
        
        df = df.drop_duplicates(subset=['num_bo', 'ano_bo', 'nome_municipio', 'data_registro'])
        
        df['perfil'] = 'Geral'
        col_crime = next((c for c in ['natureza_apurada', 'rubrica'] if c in df.columns), 'rubrica')
        df['crime_alvo'] = df[col_crime].fillna('').astype(str).upper()
        
        df.loc[df['crime_alvo'].str.contains('VEÍCULO|MOTO|CARGA|AUTO'), 'perfil'] = 'Motorista'
        df.loc[df['crime_alvo'].str.contains('BICICLETA|BIKE'), 'perfil'] = 'Ciclista'
        
        col_local = next((c for c in ['descr_tipolocal', 'descr_local'] if c in df.columns), 'descr_tipolocal')
        if col_local in df.columns:
            loc_alvo = df[col_local].fillna('').astype(str).upper()
            df.loc[(loc_alvo.str.contains('VIA PÚBLICA')) & (df['crime_alvo'].str.contains('CELULAR|PESSOA')), 'perfil'] = 'Pedestre'
        
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
        
        ouro_path = self.pastas["ouro"] / "base_final_looker.csv"
        fato.to_csv(ouro_path, index=False)
        self.auditoria["hash_ouro"] = hashlib.sha256(open(ouro_path, "rb").read()).hexdigest()
        
        return len(fato)

    def iniciar_pipeline(self):
        try:
            pool = []
            log_op = []
            mudanca = False
            
            for ano in self.anos:
                self.configurar_sessao()
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                caminho_local = self.pastas["bronze"] / f"bruto_{ano}.parquet"
                
                try:
                    head = self.sessao.head(url, timeout=30)
                    tamanho_remoto = str(head.headers.get('Content-Length', '0'))
                    
                    if caminho_local.exists() and self.auditoria.get(f"size_{ano}") == tamanho_remoto:
                        pool.append(pd.read_parquet(caminho_local))
                        log_op.append(f"📦 {ano}: DeltaSync Inalterado")
                        continue

                    time.sleep(random.uniform(3, 7))
                    res = self.sessao.get(url, timeout=240)
                    if res.status_code == 200:
                        df_novo = self.extrair_dados_validos(res.content)
                        if df_novo is not None:
                            df_novo.to_parquet(caminho_local)
                            self.auditoria[f"size_{ano}"] = tamanho_remoto
                            self.auditoria[f"hash_{ano}"] = hashlib.sha256(res.content).hexdigest()
                            pool.append(df_novo)
                            log_op.append(f"📥 {ano}: Sincronizado via DeltaSync")
                            mudanca = True
                except:
                    if caminho_local.exists():
                        pool.append(pd.read_parquet(caminho_local))
                        log_op.append(f"⚠️ {ano}: Erro Conexao (Usando Cache)")

            if not pool: raise Exception("Bloqueio SSP: Sem acesso aos dados criminais")

            total = self.processar_ia_v22(pd.concat(pool))
            with open(self.manifesto_path, "w") as f: json.dump(self.auditoria, f, indent=4)

            self.enviar_notificacao("SafeDriver: Monitor Operacional", "\n".join(log_op), 3447003, sucesso=True)
            if mudanca:
                self.enviar_notificacao("SafeDriver: Relatorio Executivo", f"Areas Mapeadas: {total}\nEnsemble: Ativo\nIntegridade: SHA256", 3066993, sucesso=True)

        except Exception as e:
            self.enviar_notificacao("SafeDriver: Alerta de Erro Critico", str(e), 15158332, sucesso=False)
            raise e

if __name__ == "__main__":
    MotorSafeDriver().iniciar_pipeline()
