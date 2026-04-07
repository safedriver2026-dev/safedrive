import sys
import os
import json
import requests
import traceback
import polars as pl
import pandas as pd
import numpy as np
import h3
import gc
import time
import holidays
import warnings
import fastexcel
import hashlib
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

print("🚀 [SISTEMA] Motor SafeDriver iniciado...", flush=True)
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
        self.webhook_sucesso = os.environ.get("DISCORD_SUCESSO")
        
        self.storage_client = storage.Client(project="safe-driver-fc3a9")
        
        self.feriados_br = holidays.Brazil(years=[self.hoje.year, self.hoje.year-1, self.hoje.year-2])
        self.hashes_seguranca = dict() 

    def gerar_hash_sha256(self, caminho_ficheiro):
        sha256_hash = hashlib.sha256()
        with open(caminho_ficheiro, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def processar_camada_raw(self):
        print("--- [Camada Raw] A extrair dados da SSP ---", flush=True)
        ano_inicio, ano_atual = 2022, self.hoje.year
        mapeamento = {
            'DATAOCORRENCIA': 'DATA_OCORRENCIA_BO', 'DATA DO FATO': 'DATA_OCORRENCIA_BO',
            'NATUREZA': 'RUBRICA', 'NUMERO_BOLETIM': 'NUM_BO', 'NÚMERO DO BO': 'NUM_BO'
        }

        for ano in range(ano_inicio, ano_atual + 1):
            url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            parquet_raw = self.pastas["raw"] / f"ssp_bruto_{ano}.parquet"
            xlsx_temp = self.pastas["raw"] / f"temp_{ano}.xlsx"
            
            if parquet_raw.exists():
                print(f"✅ Ano {ano} já processado.", flush=True)
                self.hashes_seguranca[parquet_raw.name] = self.gerar_hash_sha256(parquet_raw)
                continue

            for t in range(3):
                try:
                    print(f"📥 A descarregar ano {ano}...", flush=True)
                    r = requests.get(url, stream=True, verify=False, timeout=60)
                    if r.status_code == 200:
                        with open(xlsx_temp, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=1024*1024): f.write(chunk)
                        break
                except: time.sleep(5)

            if not xlsx_temp.exists(): continue

            try:
                excel = fastexcel.read_excel(str(xlsx_temp))
                abas = [a for a in excel.sheet_names if "capa" not in a.lower()]
                dfs = []
                for aba in abas:
                    df_aba = pl.read_excel(str(xlsx_temp), sheet_name=aba, engine="calamine")
                    df_aba = df_aba.rename({c: str(c).upper().strip() for c in df_aba.columns})
                    for v, n in mapeamento.items():
                        if v in df_aba.columns: df_aba = df_aba.rename({v: n})
                    dfs.append(df_aba.with_columns(pl.all().cast(pl.String)))
                
                if dfs:
                    df_final = pl.concat(dfs, how="diagonal")
                    df_final.write_parquet(parquet_raw)
                    self.hashes_seguranca[parquet_raw.name] = self.gerar_hash_sha256(parquet_raw)
                os.remove(xlsx_temp)
            except Exception as e:
                print(f"⚠️ Erro no ano {ano}: {e}", flush=True)

    def processar_camada_prata(self):
        print("--- [Camada Prata] A limpar dados ---", flush=True)
        arquivos = [str(p) for p in self.pastas["raw"].glob("*.parquet")]
        df = pl.concat([pl.scan_parquet(f) for f in arquivos], how="diagonal").collect()
        
        df_prata = (
            df.lazy()
            .with_columns([
                pl.col("NUM_BO").replace(["nan", ""], None),
                pl.col("DATA_OCORRENCIA_BO").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).alias("DATA_DT"),
                pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LAT"),
                pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LON"),
            ])
            .filter((pl.col("LAT").is_not_null()) & (pl.col("DATA_DT").is_not_null()))
            .unique(subset=["NUM_BO"])
            .collect()
        )

        coords = df_prata.select(["LAT", "LON"]).unique().to_pandas()
        coords['H3'] = coords.apply(lambda r: h3.latlng_to_cell(r['LAT'], r['LON'], 8), axis=1)
        df_final = df_prata.join(pl.from_pandas(coords), on=["LAT", "LON"], how="left")
        
        df_final.write_parquet(self.pastas["prata"] / "camada_prata_limpa.parquet")
        return df_final

    def processar_camada_ouro_e_ml(self, df):
        print("--- [Camada Ouro] A gerar inteligência ---", flush=True)
        col_crime = 'NATUREZA_APURADA' if 'NATUREZA_APURADA' in df.columns else 'RUBRICA'
        
        df = df.with_columns([
            pl.col(col_crime).str.contains("ROUBO").alias("IS_ROUBO"),
            pl.col(col_crime).str.contains("FURTO").alias("IS_FURTO"),
            pl.col("DATA_DT").dt.hour().alias("HORA")
        ]).with_columns([
            pl.when(pl.col("IS_ROUBO")).then(20).when(pl.col("IS_FURTO")).then(10).otherwise(5).alias("RISCO_BASE")
        ])

        fato = df.group_by(["H3", "HORA"]).agg([pl.col("RISCO_BASE").sum().alias("RISCO"), pl.col("LAT").mean().alias("LAT"), pl.col("LON").mean().alias("LON")])
        
        X_cols = ['LAT', 'LON', 'HORA']
        X = StandardScaler().fit_transform(fato.select(X_cols).to_pandas())
        y = np.log1p(fato.select("RISCO").to_numpy().ravel())
        
        model = CatBoostRegressor(iterations=200, silent=True).fit(X, y)
        preds = np.round(np.expm1(model.predict(X)), 2)
        
        fato_final = fato.with_columns([pl.Series("PREVISAO", preds)])
        fato_final.write_parquet(self.pastas["ouro"] / "dashboard_risco.parquet")
        
        print("--- A enviar para o Google Cloud Storage ---", flush=True)
        bucket = self.storage_client.bucket(self.bucket_nome)
        for f in self.raiz.rglob("datalake/*/*.parquet"):
            blob = bucket.blob(str(f.relative_to(self.raiz)))
            blob.upload_from_filename(str(f))
            print(f"✅ Upload: {f.name}", flush=True)

        if self.webhook_sucesso:
            requests.post(self.webhook_sucesso, json={"content": "✅ SafeDriver: Dados atualizados no Storage com sucesso."})

if __name__ == "__main__":
    try:
        motor = MotorSafeDriverCloud()
        motor.processar_camada_raw()
        df_prata = motor.processar_camada_prata()
        motor.processar_camada_ouro_e_ml(df_prata)
    except Exception:
        err = traceback.format_exc()
        print(f"❌ ERRO:\n{err}", flush=True)
        webhook_erro = os.environ.get("DISCORD_ERRO")
        if webhook_erro:
            requests.post(webhook_erro, json={"content": f"❌ Falha no SafeDriver:\n
http://googleusercontent.com/immersive_entry_chip/0
