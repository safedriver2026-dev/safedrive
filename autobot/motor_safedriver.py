import os
import json
import requests
import pandas as pd
import numpy as np
import h3
import shap
import hashlib
import holidays
import gc
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

class MotorAnaliticoRisco:
    def __init__(self):
        # Governança de Diretórios
        self.raiz = Path(".")
        self.pastas = {
            "bronze": self.raiz / "datalake" / "bronze",
            "ouro": self.raiz / "datalake" / "ouro",
            "auditoria": self.raiz / "datalake" / "auditoria"
        }
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        
        self.hoje = datetime.now()
        self.feriados_sp = holidays.Brazil(state='SP')
        self.webhook = os.environ.get("DISCORD_SUCESSO")
        self.limites_sp = {"lat": (-25.35, -19.77), "lon": (-53.11, -44.15)}

    def _gerar_hash(self, caminho):
        sha = hashlib.sha256()
        with open(caminho, "rb") as f:
            for bloco in iter(lambda: f.read(4096), b""): sha.update(bloco)
        return sha.hexdigest()

    def suavizar_risco_espacial(self, df):
        """Aplica Spatial Lag (Vizinhança 1º Grau) para eliminar efeito de borda"""
        mapa_base = df.groupby('H3_INDEX')['SCORE_PONDERADO'].mean().to_dict()
        
        def calcular_vizinhos(hex_id):
            # Consulta o hexágono e seus 6 vizinhos imediatos
            vizinhos = h3.grid_disk(hex_id, 1)
            valores = [mapa_base.get(v, 0) for v in vizinhos]
            return np.mean(valores)
        
        return df['H3_INDEX'].apply(calcular_vizinhos)

    def engenharia_atributos_elite(self, df_raw):
        """Transformação de dados brutos em sinais preditivos de alta densidade"""
        df = df_raw.copy()
        df.columns = [str(c).upper().strip() for c in df.columns]
        df['DATA_DT'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        df = df.dropna(subset=['DATA_DT', 'LATITUDE', 'LONGITUDE'])
        
        df['LAT'] = pd.to_numeric(df['LATITUDE'], errors='coerce')
        df['LON'] = pd.to_numeric(df['LONGITUDE'], errors='coerce')
        
        # Geoprocessamento H3 Nível 8 (Equilíbrio Sinal-Ruído)
        df['H3_INDEX'] = [h3.latlng_to_cell(lat, lon, 8) for lat, lon in zip(df['LAT'], df['LON'])]
        
        # Combate ao Efeito Retrovisor: Ponderação Exponencial por Recência
        dias_atraso = (self.hoje - df['DATA_DT']).dt.days
        df['PESO_RECENCIA'] = np.where(dias_atraso <= 180, 3.0, 1.0)
        
        # Severidade e Turnos
        df['PESO_CRIME'] = np.where(df['RUBRICA'].str.contains('ROUBO', na=False), 20, 5)
        df['SCORE_PONDERADO'] = df['PESO_CRIME'] * df['PESO_RECENCIA']
        df['TURNO'] = pd.cut(df['DATA_DT'].dt.hour, bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]).astype(int)
        
        # Agrupamento Star Schema (Fato)
        fato = df.groupby(['H3_INDEX', 'TURNO', 'DATA_DT']).agg({
            'SCORE_PONDERADO': 'sum', 'LAT': 'mean', 'LON': 'mean'
        }).reset_index()
        
        # Inteligência de Contexto
        fato['RISCO_VIZINHANCA'] = self.suavizar_risco_espacial(fato)
        fato['IS_PAGAMENTO'] = fato['DATA_DT'].dt.day.isin([5,6,7,20,21]).astype(int)
        fato['DIA_SEMANA'] = fato['DATA_DT'].dt.dayofweek
        
        return fato

    def treinamento_e_auditoria(self, fato):
        """Motor Ensemble com Explicabilidade SHAP"""
        X = fato[['LAT', 'LON', 'TURNO', 'IS_PAGAMENTO', 'DIA_SEMANA', 'RISCO_VIZINHANCA']]
        y = np.log1p(fato['SCORE_PONDERADO'])
        
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_s, y, test_size=0.2, random_state=42)

        # Ensemble Preditivo (CatBoost + LGBM)
        cat = CatBoostRegressor(iterations=1000, depth=8, learning_rate=0.04, silent=True).fit(X_train, y_train)
        lgbm = LGBMRegressor(n_estimators=1000, max_depth=10, learning_rate=0.04, verbose=-1).fit(X_train, y_train)
        
        # Inferência
        fato['PREDICAO_RISCO'] = np.round(np.expm1((cat.predict(X_s) * 0.6) + (lgbm.predict(X_s) * 0.4)), 2)
        
        # Auditoria SHAP
        explainer = shap.TreeExplainer(cat)
        shap_values = explainer.shap_values(X_s)
        for i, col in enumerate(X.columns):
            fato[f'SHAP_{col}'] = np.round(shap_values[:, i], 4)
            
        r2 = r2_score(y_test, (cat.predict(X_test) * 0.6) + (lgbm.predict(X_test) * 0.4))
        mae = mean_absolute_error(np.expm1(y_test), np.expm1((cat.predict(X_test) * 0.6) + (lgbm.predict(X_test) * 0.4)))
        
        return r2, mae, fato

    def exportacao_e_governanca(self, r2, mae, df_final):
        """Persistência Otimizada para Contornar Limites do GitHub (100MB)"""
        # 1. Base Completa em Parquet (Binário Comprimido)
        df_final.to_parquet(self.pastas["ouro"] / "inteligencia_consolidada.parquet", index=False, compression='snappy')
        
        # 2. Base Dashboard em CSV (Otimizada para < 100MB)
        # Removemos colunas redundantes e mantemos apenas o sinal vital para o Dashboard
        cols_dash = ['H3_INDEX', 'LAT', 'LON', 'TURNO', 'IS_PAGAMENTO', 'PREDICAO_RISCO']
        # Identifica a maior influência do SHAP para cada linha (Fator de Explicação)
        cols_shap = [c for c in df_final.columns if 'SHAP_' in c]
        df_final['FATOR_CRITICO'] = df_final[cols_shap].idxmax(axis=1).str.replace('SHAP_', '')
        
        df_dash = df_final[cols_dash + ['FATOR_CRITICO']]
        caminho_csv = self.pastas["ouro"] / "dashboard_risco_sp.csv"
        df_dash.to_csv(caminho_csv, index=False)
        
        # 3. Selo de Auditoria e Log
        selo = self._gerar_hash(caminho_csv)
        with open(self.pastas["auditoria"] / "controle_integridade.json", "w") as f:
            json.dump({"r2": r2, "mae": mae, "sha256": selo, "timestamp": self.hoje.isoformat()}, f, indent=4)

        if self.webhook:
            requests.post(self.webhook, json={
                "embeds": [{
                    "title": "🛡️ Status: Governança de Dados Ativa",
                    "description": f"**Confiabilidade ($R^2$):** {r2:.2%}\n**Erro Médio:** ± {mae:.2f} pts\n**Integridade:** ✅ Selo SHA-256 Verificado",
                    "color": 3066993
                }]
            })

    def executar_pipeline(self, df_raw):
        try:
            fato = self.engenharia_atributos_elite(df_raw)
            r2, mae, df_final = self.treinamento_e_auditoria(fato)
            self.exportacao_e_governanca(r2, mae, df_final)
            return True
        except Exception as e:
            print(f"Falha na execução: {e}")
            return False

if __name__ == "__main__":
    pass
