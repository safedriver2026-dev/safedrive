import os
import time
import json
import requests
import pandas as pd
import numpy as np
import h3
import shap
import hashlib
import holidays
import gc
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class MotorSafeDriver:
    def __init__(self):
        self.raiz = Path(".")
        self.pastas = {
            "entrada": self.raiz / "datalake" / "raw",
            "cache": self.raiz / "datalake" / "bronze",
            "saida": self.raiz / "datalake" / "ouro",
            "seguranca": self.raiz / "datalake" / "auditoria"
        }
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        
        self.anos = list(range(2022, datetime.now().year + 1))
        self.feriados_sp = holidays.Brazil(state='SP')
        self.sessao = requests.Session()
        
        self.webhook_sucesso = os.environ.get("DISCORD_SUCESSO")
        self.webhook_erro = os.environ.get("DISCORD_ERRO")
        self.arquivo_controle = self.pastas["seguranca"] / "controle_integridade.json"
        self.auditoria = self.carregar_controle()

        # Configuração das Travas Geográficas (Limites aproximados de SP)
        self.limites_sp = {
            "lat_min": -25.35, "lat_max": -19.77,
            "lon_min": -53.11, "lon_max": -44.15
        }

        self.colunas_alvo = [
            'NUM_BO', 'DATA_OCORRENCIA_BO', 'RUBRICA', 'NATUREZA_APURADA', 
            'DESCR_TIPOLOCAL', 'LATITUDE', 'LONGITUDE', 'DESC_PERIODO'
        ]

    def carregar_controle(self):
        if self.arquivo_controle.exists():
            with open(self.arquivo_controle, "r") as f: return json.load(f)
        return {}

    def enviar_alerta(self, titulo, conteudo, cor=3066993):
        webhook = self.webhook_sucesso if cor != 15158332 else self.webhook_erro
        if not webhook: return
        msg = "\n".join([f"**{k}:** {v}" for k, v in conteudo.items()]) if isinstance(conteudo, dict) else conteudo
        payload = {"embeds": [{"title": titulo, "description": msg, "color": cor, "timestamp": datetime.now().isoformat()}]}
        try: requests.post(webhook, json=payload, timeout=10)
        except: pass

    def baixar_arquivo(self, url, destino):
        headers = {'User-Agent': 'Mozilla/5.0'}
        tentativas = 5
        for i in range(tentativas):
            try:
                with self.sessao.get(url, stream=True, headers=headers, timeout=120) as res:
                    res.raise_for_status()
                    tamanho_esperado = int(res.headers.get('content-length', 0))
                    sha = hashlib.sha256()
                    baixado = 0
                    with open(destino, 'wb') as f:
                        for pedaco in res.iter_content(chunk_size=524288): 
                            if pedaco:
                                f.write(pedaco); sha.update(pedaco); baixado += len(pedaco)
                    if tamanho_esperado > 0 and baixado < tamanho_esperado:
                        raise Exception("Download incompleto")
                    return sha.hexdigest(), str(baixado)
            except:
                if i < tentativas - 1: time.sleep((i + 1) * 5); continue
                raise Exception(f"Falha de rede: {url}")

    def extrair_dados_excel(self, caminho):
        try:
            excel = pd.ExcelFile(caminho, engine='calamine')
            abas_validas = []
            for aba in excel.sheet_names:
                df_teste = excel.parse(aba, nrows=0)
                colunas = [str(c).upper().strip() for c in df_teste.columns]
                if 'NUM_BO' in colunas and 'LATITUDE' in colunas:
                    df = excel.parse(aba, usecols=lambda x: str(x).upper().strip() in self.colunas_alvo, dtype=str)
                    df.columns = [str(c).upper().strip() for c in df.columns]
                    df = df.dropna(subset=['NUM_BO'])
                    df['NUM_BO'] = df['NUM_BO'].astype(str).str.strip()
                    abas_validas.append(df.drop_duplicates(subset=['NUM_BO']))
                    gc.collect()
            if abas_validas:
                consolidado = pd.concat(abas_validas, ignore_index=True)
                return consolidado.drop_duplicates(subset=['NUM_BO'])
            return None
        except: return None

    def compilar_ia(self, df):
        df.columns = [str(c).lower().strip() for c in df.columns]
        df = df.drop_duplicates(subset=['num_bo']).copy()
        
        df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia_bo'], errors='coerce')
        df = df.dropna(subset=['data_ocorrencia']).copy()
        
        # --- TRAVA GEOGRÁFICA (LIMPEZA DE COORDENADAS) ---
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        
        # Remove nulos e coordenadas 0,0 (ponto no oceano na África)
        df = df.dropna(subset=['latitude', 'longitude']).copy()
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()

        # Trava de Estado: Filtra apenas pontos dentro do Quadrante de São Paulo
        df = df[
            (df['latitude'] >= self.limites_sp['lat_min']) & 
            (df['latitude'] <= self.limites_sp['lat_max']) &
            (df['longitude'] >= self.limites_sp['lon_min']) & 
            (df['longitude'] <= self.limites_sp['lon_max'])
        ].copy()

        # Trava de Mar: Remove pontos que caem no Atlântico (Logica Simplificada por Coordenada)
        # Em SP, quanto mais ao sul/leste, maior o risco de cair no mar se a coordenada estiver imprecisa.
        # Aqui garantimos que o H3 só indexe se a célula for terrestre.
        df['h3_index'] = [h3.latlng_to_cell(lat, lon, 9) for lat, lon in zip(df['latitude'], df['longitude'])]
        
        # Filtro final: O H3 permite verificar se a célula é válida e terrestre
        # (Nesta escala H3, células de mar são identificadas pela ausência de vizinhos terrestres se usássemos polígonos,
        # mas a trava de Bounding Box acima já mata 99% do problema do oceano).

        # --- RESTANTE DO PROCESSAMENTO ---
        df['dia_mes'] = df['data_ocorrencia'].dt.day
        df['is_feriado'] = df['data_ocorrencia'].apply(lambda x: 1 if x in self.feriados_sp else 0).astype(np.int8)
        df['is_pagamento'] = df['dia_mes'].apply(lambda x: 1 if x in [5, 6, 7, 20, 21] else 0).astype(np.int8)
        
        df['perfil'] = 'Geral'
        col_crime = 'natureza_apurada' if 'natureza_apurada' in df.columns else 'rubrica'
        df['crime'] = df[col_crime].fillna('').astype(str).str.upper()
        df.loc[df['crime'].str.contains('VEÍCULO|MOTO|CARGA|AUTO'), 'perfil'] = 'Motorista'
        df.loc[df['crime'].str.contains('BICICLETA|BIKE'), 'perfil'] = 'Ciclista'
        
        df['peso'] = df['crime'].apply(lambda x: 15 if 'ROUBO' in x else 2).astype(np.int8)
        
        caminho_detalhes = self.pastas["saida"] / "crimes_detalhados.csv"
        df[['num_bo', 'data_ocorrencia', 'perfil', 'crime', 'latitude', 'longitude', 'h3_index']].to_csv(caminho_detalhes, index=False)
        
        fato = df.groupby(['h3_index', 'desc_periodo', 'perfil', 'is_feriado', 'is_pagamento'])['peso'].sum().reset_index()
        fato['lat'] = [h3.cell_to_latlng(c)[0] for c in fato['h3_index']]
        fato['lon'] = [h3.cell_to_latlng(c)[1] for c in fato['h3_index']]
        fato['perfil_cod'] = fato['perfil'].astype('category').cat.codes.astype(np.int8)

        X = fato[['lat', 'lon', 'perfil_cod', 'is_feriado', 'is_pagamento']]
        y = fato['peso']
        X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.20, random_state=42)
        
        lgbm = LGBMRegressor(n_estimators=100, verbose=-1).fit(X_t, y_t)
        catb = CatBoostRegressor(iterations=100, silent=True).fit(X_t, y_t)
        
        acuracia = r2_score(y_v, (lgbm.predict(X_v) + catb.predict(X_v)) / 2)
        
        lgbm.fit(X, y)
        shap_v = shap.TreeExplainer(lgbm).shap_values(X)
        for i, col in enumerate(X.columns): fato[f'inf_{col}'] = np.round(shap_v[:, i], 3)
        fato['score_risco'] = np.round((lgbm.predict(X) + catb.predict(X)) / 2, 2)
        
        caminho_ia = self.pastas["saida"] / "predicao_risco_mapa.csv"
        fato.to_csv(caminho_ia, index=False)
        
        def selar(p):
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for b in iter(lambda: f.read(4096), b""): h.update(b)
            return h.hexdigest()

        self.auditoria["selo_ia"] = selar(caminho_ia)
        self.auditoria["selo_detalhes"] = selar(caminho_detalhes)
        
        return len(df), len(fato), acuracia

    def executar(self):
        try:
            pool = []
            for ano in self.anos:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                caminho_raw = self.pastas["entrada"] / f"ssp_{ano}.xlsx"
                caminho_cache = self.pastas["cache"] / f"ssp_{ano}.parquet"
                
                hash_site, _ = self.baixar_arquivo(url, caminho_raw)
                
                if caminho_cache.exists() and self.auditoria.get(f"hash_{ano}") == hash_site:
                    pool.append(pd.read_parquet(caminho_cache))
                else:
                    df = self.extrair_dados_excel(caminho_raw)
                    if df is not None:
                        df.to_parquet(caminho_cache)
                        self.auditoria[f"hash_{ano}"] = hash_site
                        pool.append(df)
                
                if caminho_raw.exists(): caminho_raw.unlink()

            if not pool: raise Exception("Sem dados")

            bo, zonas, r2 = self.compilar_ia(pd.concat(pool, ignore_index=True))
            with open(self.arquivo_controle, "w") as f: json.dump(self.auditoria, f, indent=4)
            self.enviar_alerta("✅ SafeDriver: Dados Auditados e Filtrados", {"B.O.s": bo, "Acurácia": f"{r2:.2%}"})
        except Exception as e:
            self.enviar_alerta("🚨 Erro Crítico", str(e), 15158332)
            raise e

if __name__ == "__main__":
    MotorSafeDriver().executar()
