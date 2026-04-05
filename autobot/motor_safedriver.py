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
from sklearn.metrics import mean_absolute_error, r2_score

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

        self.colunas_taticas = [
            'NUM_BO', 'DATA_OCORRENCIA_BO', 'RUBRICA', 'NATUREZA_APURADA', 
            'DESCR_TIPOLOCAL', 'LATITUDE', 'LONGITUDE', 'DESC_PERIODO'
        ]

    def carregar_manifesto(self):
        if self.manifesto_path.exists():
            with open(self.manifesto_path, "r") as f: return json.load(f)
        return {}

    def despachar_alerta_operacional(self, msg, cor=3447003):
        if not self.webhook_sucesso: return
        payload = {"embeds": [{"title": "⚙️ Log Operacional", "description": msg, "color": cor}]}
        try: requests.post(self.webhook_sucesso, json=payload, timeout=5)
        except: pass

    def despachar_alerta_executivo(self, titulo, kpis, cor=3066993, sucesso=True):
        webhook = self.webhook_sucesso if sucesso else self.webhook_erro
        if not webhook: return
        
        descricao = "\n".join([f"**{k}:** {v}" for k, v in kpis.items()]) if isinstance(kpis, dict) else kpis
        payload = {"embeds": [{"title": titulo, "description": descricao, "color": cor, "timestamp": datetime.now().isoformat()}]}
        try: requests.post(webhook, json=payload, timeout=10)
        except: pass

    def baixar_dados_resiliente(self, url, caminho_destino):
        retries = 5
        backoff_factor = 0.5
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        }

        for attempt in range(retries):
            try:
                res = self.sessao.get(url, stream=True, headers=headers, timeout=(60, 1800))
                res.raise_for_status()
                
                tamanho_atual = str(res.headers.get('Content-Length', '0'))
                sha256 = hashlib.sha256()
                
                with open(caminho_destino, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=1048576):
                        if chunk:
                            f.write(chunk)
                            sha256.update(chunk)
                            
                return sha256.hexdigest(), tamanho_atual
                
            except requests.exceptions.RequestException:
                if attempt < retries - 1: time.sleep(backoff_factor * (2 ** attempt))
                else: raise

    def processar_planilha_bruta(self, caminho_xlsx):
        try:
            excel = pd.ExcelFile(caminho_xlsx)
            abas_compiladas = []
            
            for aba in excel.sheet_names:
                df_amostra = excel.parse(aba, nrows=0)
                colunas_reais = [str(c).upper().strip() for c in df_amostra.columns]
                
                if 'NUM_BO' in colunas_reais and 'LATITUDE' in colunas_reais:
                    colunas_leitura = [c for c in colunas_reais if c in self.colunas_taticas]
                    
                    df_reduzido = excel.parse(aba, usecols=lambda x: str(x).upper().strip() in colunas_leitura)
                    df_reduzido.columns = [str(c).upper().strip() for c in df_reduzido.columns]
                    
                    df_reduzido = df_reduzido.dropna(subset=['NUM_BO'])
                    df_reduzido['NUM_BO'] = df_reduzido['NUM_BO'].astype(str).str.strip()
                    df_reduzido = df_reduzido.drop_duplicates(subset=['NUM_BO'])
                    
                    abas_compiladas.append(df_reduzido)
                    gc.collect()
            
            if abas_compiladas:
                df_final = pd.concat(abas_compiladas, ignore_index=True)
                df_final = df_final.drop_duplicates(subset=['NUM_BO'])
                del abas_compiladas
                gc.collect()
                return df_final
            return None
        except Exception:
            return None

    def limpar_pasta_raw(self):
        for f in self.pastas["raw"].glob("*"):
            try: f.unlink()
            except: pass

    def compilar_inteligencia(self, df):
        df.columns = [str(c).lower().strip() for c in df.columns]
        df = df.drop_duplicates(subset=['num_bo']).copy()
        linhas_iniciais = len(df)
        
        df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia_bo'], errors='coerce')
        df = df.dropna(subset=['data_ocorrencia']).copy()
        
        df['dia_semana'] = df['data_ocorrencia'].dt.dayofweek
        df['dia_mes'] = df['data_ocorrencia'].dt.day
        
        df['is_feriado'] = df['data_ocorrencia'].apply(lambda x: 1 if x in self.feriados_sp else 0).astype(np.int8)
        df['is_pagamento'] = df['dia_mes'].apply(lambda x: 1 if x in [5, 6, 7, 20, 21] else 0).astype(np.int8)
        df['is_fim_semana'] = df['dia_semana'].apply(lambda x: 1 if x >= 5 else 0).astype(np.int8)
        
        df['perfil'] = 'Geral'
        col_crime = next((c for c in ['natureza_apurada', 'rubrica'] if c in df.columns), 'rubrica')
        df['crime_alvo'] = df[col_crime].fillna('').astype(str).str.upper()
        
        df.loc[df['crime_alvo'].str.contains('VEÍCULO|MOTO|CARGA|AUTO'), 'perfil'] = 'Motorista'
        df.loc[df['crime_alvo'].str.contains('BICICLETA|BIKE'), 'perfil'] = 'Ciclista'
        
        col_local = next((c for c in ['descr_tipolocal', 'descr_local'] if c in df.columns), 'descr_tipolocal')
        if col_local in df.columns:
            loc_alvo = df[col_local].fillna('').astype(str).str.upper()
            df.loc[(loc_alvo.str.contains('VIA PÚBLICA')) & (df['crime_alvo'].str.contains('CELULAR|PESSOA')), 'perfil'] = 'Pedestre'
        
        df['severidade'] = df['crime_alvo'].apply(lambda x: 15 if 'ROUBO' in x else 2).astype(np.int8)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce').fillna(0)
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce').fillna(0)
        df = df[(df['latitude'] != 0) & (df['longitude'] != 0)].copy()

        df['h3_index'] = df.apply(lambda x: h3.latlng_to_cell(x['latitude'], x['longitude'], 9), axis=1)
        
        caminho_detalhes = self.pastas["ouro"] / "base_crimes_detalhados.csv"
        cols_exportacao = ['num_bo', 'data_ocorrencia', 'perfil', 'crime_alvo', 'latitude', 'longitude', 'h3_index']
        cols_presentes = [c for c in cols_exportacao if c in df.columns]
        df[cols_presentes].to_csv(caminho_detalhes, index=False)
        
        fato = df.groupby(['h3_index', 'desc_periodo', 'perfil', 'is_feriado', 'is_pagamento', 'is_fim_semana'])['severidade'].sum().reset_index()
        fato['lat'] = fato['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[0]).astype(np.float32)
        fato['lon'] = fato['h3_index'].apply(lambda x: h3.cell_to_latlng(x)[1]).astype(np.float32)
        fato['perfil_idx'] = fato['perfil'].astype('category').cat.codes.astype(np.int8)
        fato['periodo_idx'] = fato['desc_periodo'].astype('category').cat.codes.astype(np.int8)

        X = fato[['lat', 'lon', 'perfil_idx', 'periodo_idx', 'is_feriado', 'is_pagamento', 'is_fim_semana']]
        y = fato['severidade']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        lgbm = LGBMRegressor(n_estimators=100, verbose=-1).fit(X_train, y_train)
        catb = CatBoostRegressor(iterations=100, silent=True).fit(X_train, y_train)
        knnr = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)

        pred_test = (lgbm.predict(X_test) * 0.4 + catb.predict(X_test) * 0.4 + knnr.predict(X_test) * 0.2)
        score_r2 = r2_score(y_test, pred_test)
        margem_erro = mean_absolute_error(y_test, pred_test)

        lgbm.fit(X, y)
        catb.fit(X, y)
        knnr.fit(X, y)

        explicador = shap.TreeExplainer(lgbm)
        shap_v = explicador.shap_values(X)
        for i, col in enumerate(X.columns): fato[f'influencia_{col}'] = np.round(shap_v[:, i], 3)
            
        fato['score_risco'] = np.round((lgbm.predict(X) * 0.4 + catb.predict(X) * 0.4 + knnr.predict(X) * 0.2), 2)
        
        ouro_path = self.pastas["ouro"] / "base_final_looker.csv"
        fato.to_csv(ouro_path, index=False)
        
        sha256_ia = hashlib.sha256()
        with open(ouro_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""): sha256_ia.update(chunk)
        self.auditoria["hash_ouro_ia"] = sha256_ia.hexdigest()

        return linhas_iniciais, len(fato), score_r2, margem_erro

    def executar(self):
        inicio_op = time.time()
        try:
            pool = []
            log_op = []
            mudanca = False
            
            self.despachar_alerta_operacional("Iniciando ingestão de dados e validação de chaves primárias.")
            
            for ano in self.anos:
                self.limpar_pasta_raw()
                url_direta = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                caminho_xlsx = self.pastas["raw"] / f"bruto_{ano}.xlsx"
                caminho_parquet = self.pastas["bronze"] / f"limpo_{ano}.parquet"
                
                try:
                    hash_arquivo, tamanho_atual = self.baixar_dados_resiliente(url_direta, caminho_xlsx)
                    
                    if caminho_parquet.exists() and self.auditoria.get(f"hash_{ano}") == hash_arquivo:
                        pool.append(pd.read_parquet(caminho_parquet))
                        log_op.append(f"✅ {ano}: Sincronização Ignorada (Hash Idêntico)")
                        self.limpar_pasta_raw()
                        continue
                    
                    df_novo = self.processar_planilha_bruta(caminho_xlsx)
                    
                    if df_novo is not None:
                        df_novo.to_parquet(caminho_parquet)
                        self.auditoria[f"hash_{ano}"] = hash_arquivo
                        self.auditoria[f"size_{ano}"] = tamanho_atual
                        pool.append(df_novo)
                        log_op.append(f"📥 {ano}: Processado e Otimizado em Memória")
                        mudanca = True
                    else:
                        raise Exception("Estrutura tabular inválida")
                        
                    self.limpar_pasta_raw()
                    gc.collect()
                        
                except Exception as e:
                    self.limpar_pasta_raw()
                    if caminho_parquet.exists():
                        pool.append(pd.read_parquet(caminho_parquet))
                        log_op.append(f"⚠️ {ano}: Erro de Extração. Utilizando Cache Local.")

            if not pool: raise Exception("Falha Crítica: Fontes de dados indisponíveis.")

            self.despachar_alerta_operacional("\n".join(log_op))
            self.despachar_alerta_operacional("Ingestão concluída. Iniciando divisão Treino/Teste e cálculo SHAP.")

            df_consolidado = pd.concat(pool, ignore_index=True)
            df_consolidado = df_consolidado.drop_duplicates(subset=['NUM_BO'])
            
            linhas_puras, total_hex, score_r2, margem_erro = self.compilar_inteligencia(df_consolidado)
            
            with open(self.manifesto_path, "w") as f: json.dump(self.auditoria, f, indent=4)

            tempo_total = round((time.time() - inicio_op) / 60, 2)

            if mudanca:
                kpis = {
                    "B.O.s Únicos Processados": f"{linhas_puras:,}".replace(',', '.'),
                    "Zonas de Risco (H3) Mapeadas": f"{total_hex:,}".replace(',', '.'),
                    "Acurácia do Modelo (R²)": f"{score_r2:.2%}",
                    "Margem de Erro Média": f"± {margem_erro:.2f} pontos de severidade",
                    "Tempo de Execução": f"{tempo_total} minutos"
                }
                self.despachar_alerta_executivo("📊 SafeDriver: Relatório Executivo de IA", kpis, 3066993)

        except Exception as e:
            self.despachar_alerta_executivo("🚨 SafeDriver: Incidente Crítico", str(e), 15158332, False)
            raise e

if __name__ == "__main__":
    MotorSafeDriver().executar()
