import os
import json
import requests
import pandas as pd
import numpy as np
import h3
import hashlib
import gc
import time
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

class MotorAnaliseSafeDriver:
    def __init__(self):
        self.raiz = Path(".")
        self.pastas = {
            "raw": self.raiz / "datalake" / "raw",
            "ouro": self.raiz / "datalake" / "ouro",
            "auditoria": self.raiz / "datalake" / "auditoria"
        }
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        self.hoje = datetime.now()
        self.webhook = os.environ.get("DISCORD_SUCESSO")

    def _gerar_assinatura(self, caminho):
        sha = hashlib.sha256()
        with open(caminho, "rb") as f:
            for bloco in iter(lambda: f.read(4096), b""): sha.update(bloco)
        return sha.hexdigest()

    def extrair_dados(self):
        arquivos = list(self.pastas["raw"].glob("*.parquet")) + list(self.pastas["raw"].glob("*.csv"))
        if not arquivos:
            # Mock Data Estadual para Testes de CI/CD
            np.random.seed(42)
            n = 5000
            datas = pd.date_range(end=self.hoje, periods=n, freq='30min')
            locais = np.random.choice(['capital', 'interior'], n, p=[0.6, 0.4])
            lats = np.where(locais == 'capital', np.random.normal(-23.55, 0.02, n), np.random.normal(-22.90, 0.05, n))
            lons = np.where(locais == 'capital', np.random.normal(-46.63, 0.02, n), np.random.normal(-47.06, 0.05, n))
            is_risco = ((datas.hour > 18) | (datas.hour < 6)) & (np.isin(datas.day, [5,6,7,20,21]))
            rubricas = np.where(is_risco, "ROUBO", "FURTO")
            return pd.DataFrame({'DATA_OCORRENCIA_BO': datas, 'LATITUDE': lats, 'LONGITUDE': lons, 'RUBRICA': rubricas})
        
        return pd.read_parquet(arquivos[0]) if arquivos[0].suffix == '.parquet' else pd.read_csv(arquivos[0])

    def processar(self, df_raw):
        t_ini = time.time()
        v_bruto = len(df_raw)
        
        # Limpeza e Tipagem
        df = df_raw.copy()
        df.columns = [str(c).upper().strip() for c in df.columns]
        df['DATA_DT'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        df.dropna(subset=['DATA_DT', 'LATITUDE', 'LONGITUDE'], inplace=True)
        v_limpo = len(df)

        # Feature Engineering (Otimizada para RAM)
        df['LAT'] = pd.to_numeric(df['LATITUDE']).astype(np.float32)
        df['LON'] = pd.to_numeric(df['LONGITUDE']).astype(np.float32)
        df['H3'] = [h3.latlng_to_cell(la, lo, 8) for la, lo in zip(df['LAT'], df['LON'])]
        df['PESO'] = np.where((self.hoje - df['DATA_DT']).dt.days <= 180, 3.0, 1.0).astype(np.float32)
        df['RISCO'] = (np.where(df['RUBRICA'].str.contains('ROUBO', na=False), 20, 5) * df['PESO']).astype(np.float32)
        df['TURNO'] = pd.cut(df['DATA_DT'].dt.hour, bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]).astype(np.int8)

        # Agregação Star Schema
        fato = df.groupby(['H3', 'TURNO', 'DATA_DT']).agg({'RISCO': 'sum', 'LAT': 'mean', 'LON': 'mean'}).reset_index()
        fato['IS_PGTO'] = fato['DATA_DT'].dt.day.isin([5,6,7,20,21]).astype(np.int8)
        fato['DIA_SEM'] = fato['DATA_DT'].dt.dayofweek.astype(np.int8)
        
        # ML Engine
        X = fato[['LAT', 'LON', 'TURNO', 'IS_PGTO', 'DIA_SEM']]
        y = np.log1p(fato['RISCO'])
        X_s = StandardScaler().fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X_s, y, test_size=0.2, random_state=42)

        model = CatBoostRegressor(iterations=600, depth=8, learning_rate=0.05, silent=True).fit(X_tr, y_tr)
        preds = model.predict(X_te)
        
        r2 = r2_score(y_te, preds)
        mae = mean_absolute_error(np.expm1(y_te), np.expm1(preds))
        
        fato['PRED'] = np.round(np.expm1(model.predict(X_s)), 2).astype(np.float32)
        fato['ERRO'] = np.abs(fato['RISCO'] - fato['PRED']).astype(np.float32)
        assertividade = (fato['ERRO'] <= 5.0).sum() / len(fato)

        # Exportação e Auditoria
        caminho_csv = self.pastas["ouro"] / "dashboard_risco_real.csv"
        fato.to_parquet(self.pastas["ouro"] / "fato_risco_real.parquet", index=False)
        fato[['H3', 'DATA_DT', 'TURNO', 'RISCO', 'PRED', 'ERRO']].to_csv(caminho_csv, index=False)
        
        selo = self._gerar_assinatura(caminho_csv)
        with open(self.pastas["auditoria"] / "controle_integridade.json", "w") as f:
            json.dump({"r2": float(r2), "sha256": selo, "ts": self.hoje.isoformat()}, f)

        self._enviar_discord(taxa=assertividade, r2=r2, mae=mae, bruto=v_bruto, limpo=v_limpo, fato=len(fato), tempo=time.time()-t_ini, selo=selo)

    def _enviar_discord(self, **k):
        if not self.webhook: return
        p = { "embeds": [
            { "title": "🎯 SafeDriver: Relatório Executivo (IA)", "color": 3066993, "fields": [
                {"name": "Assertividade", "value": f"**{k['taxa']:.1%}**", "inline": True},
                {"name": "Confiança (R²)", "value": f"{k['r2']:.2%}", "inline": True},
                {"name": "Erro Médio", "value": f"± {k['mae']:.2f} pts", "inline": True}
            ]},
            { "title": "⚙️ SafeDriver: Relatório Operacional (Engenharia)", "color": 8359053, "fields": [
                {"name": "Ingestão", "value": f"{k['bruto']:,} linhas", "inline": True},
                {"name": "Processamento", "value": f"{k['fato']:,} registros", "inline": True},
                {"name": "Tempo", "value": f"{k['tempo']:.2f}s", "inline": True},
                {"name": "Selo SHA-256", "value": f"`{k['selo'][:12]}`", "inline": False}
            ]}
        ]}
        requests.post(self.webhook, json=p)

if __name__ == "__main__":
    motor = MotorAnaliseSafeDriver()
    motor.processar(motor.extrair_dados())
