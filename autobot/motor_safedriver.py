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
from pathlib import Path
from datetime import datetime
from sklearn.neighbors import KNeighborsRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

class MotorSafeDriver:
    def __init__(self):
        self.raiz = Path(".")
        self.pastas = {
            "raw": self.raiz / "datalake" / "raw",
            "bronze": self.raiz / "datalake" / "bronze",
            "prata": self.raiz / "datalake" / "prata",
            "ouro": self.raiz / "datalake" / "ouro",
            "auditoria": self.raiz / "datalake" / "auditoria"
        }
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        
        self.anos = list(range(2022, datetime.now().year + 1))
        self.feriados_sp = holidays.Brazil(state='SP')
        self.sessao = requests.Session()
        
        self.webhook_sucesso = os.environ.get("DISCORD_SUCESSO")
        self.webhook_erro = os.environ.get("DISCORD_ERRO")
        self.manifesto_path = self.pastas["auditoria"] / "manifesto.json"
        self.auditoria = self.carregar_manifesto()

    def carregar_manifesto(self):
        if self.manifesto_path.exists():
            with open(self.manifesto_path, "r") as f: return json.load(f)
        return {}

    def despachar_alerta(self, titulo, msg, cor, sucesso=True):
        webhook = self.webhook_sucesso if sucesso else self.webhook_erro
        if not webhook: return
        payload = {"embeds": [{"title": titulo, "description": msg, "color": cor, "timestamp": datetime.now().isoformat()}]}
        try: requests.post(webhook, json=payload, timeout=10)
        except: pass

    def baixar_dados_resiliente(self, url, caminho_destino):
        retries = 5
        backoff_factor = 0.5
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }

        for attempt in range(retries):
            try:
                res = self.sessao.get(url, stream=True, headers=headers, timeout=(60, 1800))
                res.raise_for_status()
                
                sha256 = hashlib.sha256()
                with open(caminho_destino, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=1048576):
                        if chunk:
                            f.write(chunk)
                            sha256.update(chunk)
                return sha256.hexdigest(), str(res.headers.get('Content-Length', '0'))
                
            except requests.exceptions.RequestException:
                if attempt < retries - 1:
                    time.sleep(backoff_factor * (2 ** attempt))
                else:
                    raise

    def processar_planilha_bruta(self, caminho_xlsx):
        try:
            excel = pd.ExcelFile(caminho_xlsx)
            for aba in excel.sheet_names:
                df = excel.parse(aba, nrows=40)
                df.columns = [str(c).upper().strip() for c in df.columns]
                if all(k in df.columns for k in ['NUM_BO', 'ANO_BO', 'LATITUDE', 'DATA_OCORRENCIA_BO']):
                    return excel.parse(aba)
            return None
        except: return None

    def limpar_pasta_raw(self):
        for f in self.pastas["raw"].glob("*"):
            try: f.unlink()
            except: pass

    def compilar_inteligencia(self, df):
        df.columns = [str(c).lower().strip() for c in df.columns]
        df = df.drop_duplicates(subset=['num_bo', 'ano_bo', 'nome_municipio', 'data_registro'])
        
        df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia_bo'], errors='coerce')
        df = df.dropna(subset=['data_ocorrencia']).copy()
        
        df['dia_semana'] = df['data_ocorrencia'].dt.dayofweek
        df['dia_mes'] = df['data_ocorrencia'].dt.day
        
        df['is_feriado'] = df['data_ocorrencia'].apply(lambda x: 1 if x in self.feriados_sp else 0)
        df['is_pagamento'] = df['dia_mes'].apply(lambda x: 1 if x in [5, 6, 7, 20, 21] else 0)
        df['is_fim_semana'] = df['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)
        
        df['perfil'] = 'Geral'
        col_crime = next((c for c in ['natureza_apurada', 'rubrica'] if c in df.columns), 'rubrica')
        df['crime_alvo'] = df[col_crime].fillna('').astype(str).upper()
        
        df.loc[df['crime_alvo'].str.contains('VEÍCULO|MOTO|CARGA|AUTO'), 'perfil'] = 'Motorista'
        df.loc[df['crime_alvo'].str.contains('BICICLETA|BIKE'), 'perfil'] = 'Ciclista'
        
        col_local = next((c for c in ['descr_tipolocal', 'descr_local'] if c in df.columns), 'descr_tipolocal')
        if col_local in df.columns:
            loc_alvo = col_local.fillna('').astype(str).upper()
            df.loc[(loc_alvo.str.contains('VIA PÚBLICA')) & (df['crime_alvo'].str.contains('CELULAR|PESSOA')), 'perfil'] = 'Pedestre'
        
        df['severidade'] = df['crime_alvo'].apply(lambda x: 15 if 'ROUBO' in x else 2)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()

        df['h3_index'] = df.apply(lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1)
        
        fato = df.groupby(['h3_index', 'desc_periodo', 'perfil', 'is_feriado', 'is_pagamento', 'is_fim_semana'])['severidade'].sum().reset_index()
        fato['lat'] = fato['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0])
        fato['lon'] = fato['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1])
        fato['perfil_idx'] = fato['perfil'].astype('category').cat.codes
        fato['periodo_idx'] = fato['desc_periodo'].astype('category').cat.codes

        X = fato[['lat', 'lon', 'perfil_idx', 'periodo_idx', 'is_feriado', 'is_pagamento', 'is_fim_semana']]
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
        
        sha256 = hashlib.sha256()
        with open(ouro_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""): sha256.update(chunk)
        self.auditoria["hash_ouro"] = sha256.hexdigest()
        
        return len(fato)

    def executar(self):
        try:
            pool = []
            log_op = []
            mudanca = False
            
            for ano in self.anos:
                self.limpar_pasta_raw()
                url_direta = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                caminho_xlsx = self.pastas["raw"] / f"bruto_{ano}.xlsx"
                caminho_parquet = self.pastas["bronze"] / f"limpo_{ano}.parquet"
                
                try:
                    hash_arquivo, tamanho_atual = self.baixar_dados_resiliente(url_direta, caminho_xlsx)
                    
                    if caminho_parquet.exists() and self.auditoria.get(f"hash_{ano}") == hash_arquivo:
                        pool.append(pd.read_parquet(caminho_parquet))
                        log_op.append(f"📦 {ano}: Sincronizacao Ignorada (Hash Identico)")
                        self.limpar_pasta_raw()
                        continue
                    
                    df_novo = self.processar_planilha_bruta(caminho_xlsx)
                    
                    if df_novo is not None:
                        df_novo.to_parquet(caminho_parquet)
                        self.auditoria[f"hash_{ano}"] = hash_arquivo
                        self.auditoria[f"size_{ano}"] = tamanho_atual
                        pool.append(df_novo)
                        log_op.append(f"📥 {ano}: Processamento Nativo Concluido")
                        mudanca = True
                    else:
                        raise Exception(f"Layout estrutural invalido em {ano}")
                        
                    self.limpar_pasta_raw()
                        
                except Exception as e:
                    self.limpar_pasta_raw()
                    if caminho_parquet.exists():
                        pool.append(pd.read_parquet(caminho_parquet))
                        log_op.append(f"⚠️ {ano}: Erro de Extracao ({str(e)}). Base de Cache Aplicada")

            if not pool: raise Exception("Falha Critica: Fontes de dados indisponiveis para o ciclo atual.")

            total = self.compilar_inteligencia(pd.concat(pool))
            with open(self.manifesto_path, "w") as f: json.dump(self.auditoria, f, indent=4)

            self.despachar_alerta("SafeDriver: Status Operacional", "\n".join(log_op), 3447003, sucesso=True)
            if mudanca:
                self.despachar_alerta("SafeDriver: Relatorio Analitico", f"Pipeline Atualizado.\nAreas Mapeadas: {total}\nFatores Comportamentais: Ativos", 3066993, sucesso=True)

        except Exception as e:
            self.despachar_alerta("SafeDriver: Incidente Critico", str(e), 15158332, sucesso=False)
            raise e

if __name__ == "__main__":
    MotorSafeDriver().executar()
