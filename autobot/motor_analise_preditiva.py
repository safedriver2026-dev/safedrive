import os
import json
import requests
import pandas as pd
import numpy as np
import h3
import gc
import time
import warnings
from pathlib import Path
from datetime import datetime
from google.cloud import storage # Biblioteca para gerenciar o Bucket
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

class MotorSafeDriverCloud:
    def __init__(self):
        self.raiz = Path(".")
        self.bucket_nome = os.environ.get("GCP_BUCKET_NAME")
        self.pastas = {
            "raw": self.raiz / "datalake" / "raw",
            "prata": self.raiz / "datalake" / "prata",
            "ouro": self.raiz / "datalake" / "ouro",
            "auditoria": self.raiz / "datalake" / "auditoria"
        }
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        self.hoje = datetime.now()
        self.webhook = os.environ.get("DISCORD_SUCESSO")
        
        # Inicializa cliente do Google Cloud Storage
        self.storage_client = storage.Client()

    def garantir_infraestrutura_bucket(self):
        """Verifica se o bucket existe. Se não existir, cria na primeira rodagem."""
        try:
            bucket = self.storage_client.get_bucket(self.bucket_nome)
            print(f"✅ Bucket {self.bucket_nome} já existe. Seguindo...")
        except Exception:
            print(f"🚀 Primeira Rodagem Detectada! Criando Bucket: {self.bucket_nome}")
            self.storage_client.create_bucket(self.bucket_nome, location="US-EAST1")
            print("✅ Infraestrutura de Nuvem Provisionada.")

    def extrair_dados(self):
        arquivos = list(self.pastas["raw"].glob("*.parquet")) + list(self.pastas["raw"].glob("*.csv"))
        if not arquivos:
            # Gerador de Mock (Caso o GitHub não tenha os 2.5M de linhas no primeiro teste)
            np.random.seed(42)
            n = 5000
            df = pd.DataFrame({
                'DATA_OCORRENCIA_BO': pd.date_range(end=self.hoje, periods=n, freq='30min'),
                'LATITUDE': np.random.normal(-23.55, 0.05, n),
                'LONGITUDE': np.random.normal(-46.63, 0.05, n),
                'RUBRICA': np.random.choice(["ROUBO", "FURTO"], n)
            })
            df.to_parquet(self.pastas["raw"] / "base_bronze_inicial.parquet", index=False)
            return df
        return pd.read_parquet(arquivos[0]) if arquivos[0].suffix == '.parquet' else pd.read_csv(arquivos[0])

    def processar_e_salvar_camadas(self, df_raw):
        t_ini = time.time()
        
        # --- PRATA (Data Cleaning) ---
        df = df_raw.copy()
        df.columns = [str(c).upper().strip() for c in df.columns]
        df['DATA_DT'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'], errors='coerce')
        df.dropna(subset=['DATA_DT', 'LATITUDE', 'LONGITUDE'], inplace=True)
        df['LAT'] = pd.to_numeric(df['LATITUDE']).astype(np.float32)
        df['LON'] = pd.to_numeric(df['LONGITUDE']).astype(np.float32)
        df['H3'] = [h3.latlng_to_cell(la, lo, 8) for la, lo in zip(df['LAT'], df['LON'])]
        
        caminho_prata = self.pastas["prata"] / "camada_prata_limpa.parquet"
        df.to_parquet(caminho_prata, compression='snappy', index=False)

        # --- OURO (Business Logic & IA) ---
        df['PESO'] = np.where((self.hoje - df['DATA_DT']).dt.days <= 180, 3.0, 1.0).astype(np.float32)
        df['RISCO'] = (np.where(df['RUBRICA'].str.contains('ROUBO', na=False), 20, 5) * df['PESO']).astype(np.float32)
        df['TURNO'] = pd.cut(df['DATA_DT'].dt.hour, bins=[-1, 6, 12, 18, 24], labels=[0, 1, 2, 3]).astype(np.int8)

        fato = df.groupby(['H3', 'TURNO', 'DATA_DT']).agg({'RISCO': 'sum', 'LAT': 'mean', 'LON': 'mean'}).reset_index()
        fato['IS_PGTO'] = fato['DATA_DT'].dt.day.isin([5,6,7,20,21]).astype(np.int8)
        fato['DIA_SEM'] = fato['DATA_DT'].dt.dayofweek.astype(np.int8)
        
        X = fato[['LAT', 'LON', 'TURNO', 'IS_PGTO', 'DIA_SEM']]
        y = np.log1p(fato['RISCO'])
        X_s = StandardScaler().fit_transform(X)
        model = CatBoostRegressor(iterations=300, silent=True).fit(X_s, y)
        
        fato['PRED'] = np.round(np.expm1(model.predict(X_s)), 2).astype(np.float32)
        caminho_ouro = self.pastas["ouro"] / "fato_risco_real.parquet"
        fato.to_parquet(caminho_ouro, index=False)

        # --- UPLOAD AUTOMATIZADO DE TODAS AS CAMADAS ---
        self.garantir_infraestrutura_bucket()
        self.fazer_upload_diretorio(self.raiz / "datalake")
        
        print(f"🏁 Pipeline Finalizado em {time.time()-t_ini:.2f}s. Todas as camadas na Nuvem.")

    def fazer_upload_diretorio(self, caminho_local):
        bucket = self.storage_client.get_bucket(self.bucket_nome)
        for arquivo in Path(caminho_local).rglob("*"):
            if arquivo.is_file():
                blob_path = str(arquivo.relative_to(self.raiz))
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(arquivo))
                print(f"📤 Uploaded: {blob_path}")

if __name__ == "__main__":
    motor = MotorSafeDriverCloud()
    motor.processar_e_salvar_camadas(motor.extrair_dados())
