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

        # Colunas essenciais para o cálculo (economiza RAM)
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
        
        # Formata o dicionário de métricas para o Discord
        if isinstance(conteudo, dict):
            msg = "\n".join([f"**{k}:** {v}" for k, v in conteudo.items()])
        else:
            msg = conteudo

        payload = {"embeds": [{"title": titulo, "description": msg, "color": cor, "timestamp": datetime.now().isoformat()}]}
        try: requests.post(webhook, json=payload, timeout=10)
        except: pass

    def baixar_arquivo(self, url, destino):
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            res = self.sessao.get(url, stream=True, headers=headers, timeout=(60, 1800))
            res.raise_for_status()
            
            sha = hashlib.sha256()
            with open(destino, 'wb') as f:
                for pedaco in res.iter_content(chunk_size=1048576):
                    if pedaco:
                        f.write(pedaco)
                        sha.update(pedaco)
            return sha.hexdigest(), str(res.headers.get('Content-Length', '0'))
        except:
            raise Exception(f"Erro ao conectar com a base da SSP: {url}")

    def extrair_dados_excel(self, caminho):
        try:
            # Engine 'calamine' é muito mais rápida para ler Excel pesado
            excel = pd.ExcelFile(caminho, engine='calamine')
            abas_validas = []
            
            for aba in excel.sheet_names:
                df_teste = excel.parse(aba, nrows=0)
                colunas = [str(c).upper().strip() for c in df_teste.columns]
                
                if 'NUM_BO' in colunas and 'LATITUDE' in colunas:
                    # Lê apenas as colunas que precisamos
                    df = excel.parse(aba, usecols=lambda x: str(x).upper().strip() in self.colunas_alvo)
                    df.columns = [str(c).upper().strip() for c in df.columns]
                    
                    # Limpeza imediata de duplicados para liberar memória
                    df = df.dropna(subset=['NUM_BO'])
                    df['NUM_BO'] = df['NUM_BO'].astype(str).str.strip()
                    abas_validas.append(df.drop_duplicates(subset=['NUM_BO']))
                    gc.collect()
            
            if abas_validas:
                consolidado = pd.concat(abas_validas, ignore_index=True)
                return consolidado.drop_duplicates(subset=['NUM_BO'])
            return None
        except:
            return None

    def processar_ia(self, df):
        df.columns = [str(c).lower().strip() for c in df.columns]
        df = df.drop_duplicates(subset=['num_bo']).copy()
        total_bo_unicos = len(df)
        
        # Tratamento de datas e sazonalidade
        df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia_bo'], errors='coerce')
        df = df.dropna(subset=['data_ocorrencia']).copy()
        
        df['dia_mes'] = df['data_ocorrencia'].dt.day
        df['is_feriado'] = df['data_ocorrencia'].apply(lambda x: 1 if x in self.feriados_sp else 0).astype(np.int8)
        df['is_pagamento'] = df['dia_mes'].apply(lambda x: 1 if x in [5, 6, 7, 20, 21] else 0).astype(np.int8)
        
        # Definição de perfis (Pedestre, Motorista, Ciclista)
        df['perfil'] = 'Geral'
        col_crime = 'natureza_apurada' if 'natureza_apurada' in df.columns else 'rubrica'
        df['crime'] = df[col_crime].fillna('').astype(str).str.upper()
        
        df.loc[df['crime'].str.contains('VEÍCULO|MOTO|CARGA|AUTO'), 'perfil'] = 'Motorista'
        df.loc[df['crime'].str.contains('BICICLETA|BIKE'), 'perfil'] = 'Ciclista'
        
        # Severidade do crime
        df['peso'] = df['crime'].apply(lambda x: 15 if 'ROUBO' in x else 2).astype(np.int8)
        
        # Limpeza geográfica
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()

        # Criação dos hexágonos H3 (Vetorizado para ser rápido)
        df['h3_index'] = [h3.latlng_to_cell(lat, lon, 9) for lat, lon in zip(df['latitude'], df['longitude'])]
        
        # --- SALVANDO A BASE DE DETALHES (PRATA) ---
        caminho_detalhes = self.pastas["saida"] / "crimes_detalhados.csv"
        df[['num_bo', 'data_ocorrencia', 'perfil', 'crime', 'latitude', 'longitude', 'h3_index']].to_csv(caminho_detalhes, index=False)
        
        # Agrupamento para a IA
        fato = df.groupby(['h3_index', 'desc_periodo', 'perfil', 'is_feriado', 'is_pagamento'])['peso'].sum().reset_index()
        fato['lat'] = [h3.cell_to_latlng(c)[0] for c in fato['h3_index']]
        fato['lon'] = [h3.cell_to_latlng(c)[1] for c in fato['h3_index']]
        fato['perfil_cod'] = fato['perfil'].astype('category').cat.codes.astype(np.int8)

        X = fato[['lat', 'lon', 'perfil_cod', 'is_feriado', 'is_pagamento']]
        y = fato['peso']
        
        # Validação da confiança (80% treino, 20% teste)
        X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=0.20, random_state=42)
        
        modelos = [
            LGBMRegressor(n_estimators=100, verbose=-1).fit(X_t, y_t),
            CatBoostRegressor(iterations=100, silent=True).fit(X_t, y_t)
        ]
        
        # Média das predições
        preds = (modelos[0].predict(X_v) + modelos[1].predict(X_v)) / 2
        acuracia = r2_score(y_v, preds)

        # Treino final e explicabilidade SHAP
        modelos[0].fit(X, y)
        shap_v = shap.TreeExplainer(modelos[0]).shap_values(X)
        for i, col in enumerate(X.columns): fato[f'inf_{col}'] = np.round(shap_v[:, i], 3)
        fato['score_risco'] = np.round((modelos[0].predict(X) + modelos[1].predict(X)) / 2, 2)
        
        caminho_ia = self.pastas["saida"] / "predicao_risco_mapa.csv"
        fato.to_csv(caminho_ia, index=False)
        
        # Atualiza selos de auditoria
        def selar(p):
            h = hashlib.sha256()
            with open(p, 'rb') as f:
                for b in iter(lambda: f.read(4096), b""): h.update(b)
            return h.hexdigest()

        self.auditoria["selo_ia"] = selar(caminho_ia)
        self.auditoria["selo_detalhes"] = selar(caminho_detalhes)
        
        return total_bo_unicos, len(fato), acuracia

    def executar(self):
        try:
            pool = []
            for ano in self.anos:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                caminho_raw = self.pastas["entrada"] / f"ssp_{ano}.xlsx"
                caminho_cache = self.pastas["cache"] / f"ssp_{ano}.parquet"
                
                hash_atual, tamanho = self.baixar_arquivo(url, caminho_raw)
                
                # Só processa se o arquivo na SSP mudou ou se não temos o cache
                if caminho_cache.exists() and self.auditoria.get(f"hash_{ano}") == hash_atual:
                    pool.append(pd.read_parquet(caminho_cache))
                else:
                    df = self.extrair_dados_excel(caminho_raw)
                    if df is not None:
                        df.to_parquet(caminho_cache)
                        self.auditoria[f"hash_{ano}"] = hash_atual
                        pool.append(df)
                
                if caminho_raw.exists(): caminho_raw.unlink() # Limpa o Excel para poupar espaço

            if not pool: raise Exception("Sem dados para processar")

            bo_total, zonas, nota = self.compilar_inteligencia(pd.concat(pool, ignore_index=True))
            
            with open(self.arquivo_controle, "w") as f: json.dump(self.auditoria, f, indent=4)
            
            self.enviar_alerta("✅ SafeDriver: Atualização Concluída", {
                "B.O.s Únicos Auditados": f"{bo_total:,}",
                "Áreas de Risco Mapeadas": f"{zonas:,}",
                "Confiança do Modelo (R²)": f"{nota:.2%}"
            })
            
        except Exception as e:
            self.enviar_alerta("🚨 SafeDriver: Falha no Processamento", str(e), 15158332)
            raise e

if __name__ == "__main__":
    MotorSafeDriver().executar()
