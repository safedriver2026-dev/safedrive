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
        self.storage_client = storage.Client()
        self.feriados_br = holidays.Brazil(years=[self.hoje.year, self.hoje.year-1, self.hoje.year-2])
        self.linhas_descartadas = 0
        self.hashes_seguranca = dict() 

    def gerar_hash_sha256(self, caminho_arquivo):
        sha256_hash = hashlib.sha256()
        with open(caminho_arquivo, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    # ==========================================
    # CAMADA BRUTA: Download e Assinatura (SHA)
    # ==========================================
    def processar_camada_raw(self):
        ano_inicio, ano_atual = 2022, self.hoje.year
        mapeamento = {
            'DATAOCORRENCIA': 'DATA_OCORRENCIA_BO', 'DATA DO FATO': 'DATA_OCORRENCIA_BO',
            'NATUREZA': 'RUBRICA', 'NUMERO_BOLETIM': 'NUM_BO', 'NÚMERO DO BO': 'NUM_BO'
        }

        print("--- [Camada Bruta] Iniciando ingestão ---")
        for ano in range(ano_inicio, ano_atual + 1):
            url = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_" + str(ano) + ".xlsx"
            xlsx_temp = self.pastas["raw"] / ("temp_" + str(ano) + ".xlsx")
            parquet_raw = self.pastas["raw"] / ("ssp_bruto_" + str(ano) + ".parquet")
            
            if parquet_raw.exists():
                print("Ano " + str(ano) + " em cache. Gerando hash...")
                self.hashes_seguranca[parquet_raw.name] = self.gerar_hash_sha256(parquet_raw)
                continue

            for t in range(3):
                try:
                    r = requests.get(url, stream=True, verify=False, timeout=(60, 1800))
                    if r.status_code == 200:
                        with open(xlsx_temp, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=2*1024*1024): f.write(chunk)
                        break
                except: time.sleep(5)

            if not xlsx_temp.exists(): continue

            try:
                excel_reader = fastexcel.read_excel(str(xlsx_temp))
                abas = [a for a in excel_reader.sheet_names if "capa" not in a.lower()]
                
                dfs_ano = []
                for aba in abas:
                    df_aba = pl.read_excel(str(xlsx_temp), sheet_name=aba, engine="calamine")
                    df_aba = df_aba.rename({c: str(c).upper().strip() for c in df_aba.columns})
                    for velho, novo in mapeamento.items():
                        if velho in df_aba.columns: df_aba = df_aba.rename({velho: novo})
                    df_aba = df_aba.with_columns(pl.all().cast(pl.String))
                    dfs_ano.append(df_aba)
                
                if dfs_ano:
                    # Concatena na diagonal para aceitar colunas novas (como CMD)
                    df_ano_completo = pl.concat(dfs_ano, how="diagonal")
                    df_ano_completo.write_parquet(parquet_raw, compression='snappy')
                    self.hashes_seguranca[parquet_raw.name] = self.gerar_hash_sha256(parquet_raw)
                    print("Salvo: " + parquet_raw.name)
                
                os.remove(xlsx_temp)
                gc.collect()
            except Exception as e:
                print("Erro no processamento de " + str(ano) + ": " + str(e))

    # ==========================================
    # CAMADA PRATA: Limpeza e H3
    # ==========================================
    def processar_camada_prata(self):
        print("--- [Camada Prata] Limpeza e Normalização ---")
        arquivos = [str(p) for p in self.pastas["raw"].glob("*.parquet")]
        df = pl.concat([pl.scan_parquet(f) for f in arquivos], how="diagonal").collect()
        
        vol_ini = df.height
        df_limpo = (
            df.lazy()
            .with_columns([
                pl.col("NUM_BO").replace(["nan", ""], None),
                pl.col("DATA_OCORRENCIA_BO").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).alias("DATA_DT"),
            ])
            .filter(pl.col("NUM_BO").is_not_null())
            .unique(subset=["NUM_BO"])
            .with_columns([
                pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LAT"),
                pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LON"),
            ])
            .filter((pl.col("LAT").is_not_null()) & (pl.col("DATA_DT").is_not_null()))
            .collect()
        )

        coords = df_limpo.select(["LAT", "LON"]).unique().to_pandas()
        coords['H3'] = coords.apply(lambda r: h3.latlng_to_cell(r['LAT'], r['LON'], 8), axis=1)
        df_prata = df_limpo.join(pl.from_pandas(coords), on=["LAT", "LON"], how="left")
        
        self.linhas_descartadas = vol_ini - df_prata.height
        df_prata.write_parquet(self.pastas["prata"] / "camada_prata_limpa.parquet")
        return df_prata

    # ==========================================
    # CAMADA OURO: Inteligência e Upload
    # ==========================================
    def processar_camada_ouro_e_ml(self, df):
        print("--- [Camada Ouro] Treino e Predição ---")
        col_crime = 'NATUREZA_APURADA' if 'NATUREZA_APURADA' in df.columns else 'RUBRICA'
        
        df = df.with_columns([
            pl.col(col_crime).str.contains("ROUBO").alias("IS_ROUBO"),
            pl.col(col_crime).str.contains("FURTO").alias("IS_FURTO"),
            pl.col("DATA_DT").dt.hour().alias("HORA")
        ]).with_columns([
            pl.when(pl.col("IS_ROUBO")).then(20).when(pl.col("IS_FURTO")).then(10).otherwise(5).alias("RISCO_BASE"),
            pl.when(pl.col("HORA") <= 12).then(1).when(pl.col("HORA") <= 18).then(2).otherwise(3).alias("TURNO")
        ])

        fato = df.group_by(["H3", "TURNO", "DATA_DT"]).agg([
            pl.col("RISCO_BASE").sum().alias("RISCO"),
            pl.col("LAT").mean().alias("LAT"),
            pl.col("LON").mean().alias("LON")
        ]).with_columns([
            pl.col("DATA_DT").dt.weekday().alias("DIA_SEM"),
            pl.col("DATA_DT").dt.month().alias("MES")
        ])

        X_cols = ['LAT', 'LON', 'TURNO', 'DIA_SEM', 'MES']
        X = StandardScaler().fit_transform(fato.select(X_cols).to_pandas())
        y = np.log1p(fato.select("RISCO").to_numpy().ravel())
        
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = CatBoostRegressor(iterations=500, silent=True).fit(X_tr, y_tr)
        
        # SHAP nativo (sem crashes)
        pool = Pool(X_te[:1000])
        importancia = model.get_feature_importance(pool, type='ShapValues')[:, :-1]
        top_driver = X_cols[np.argmax(np.abs(importancia).mean(axis=0))]

        preds = np.round(np.expm1(model.predict(X)), 2)
        fato_final = fato.select(['H3', 'DATA_DT', 'TURNO', 'RISCO']).with_columns([pl.Series("PREVISAO", preds)])

        fato_final.write_parquet(self.pastas["ouro"] / "dashboard_risco.parquet")
        
        # Auditoria
        manifesto = dict(hashes=self.hashes_seguranca, r2=float(r2_score(y_te, model.predict(X_te))), timestamp=self.hoje.isoformat())
        with open(self.pastas["auditoria"] / "auditoria.json", "w") as f: json.dump(manifesto, f)
        
        # Upload para o Bucket (Foca em alimentar, sem tentar criar)
        print("--- Alimentando o Bucket ---")
        bucket = self.storage_client.bucket(self.bucket_nome)
        for aqv in self.raiz.rglob("datalake/*/*.parquet"):
            blob = bucket.blob(str(aqv.relative_to(self.raiz)))
            blob.upload_from_filename(str(aqv))
            print("Upload: " + aqv.name)
        
        if self.webhook_sucesso:
            requests.post(self.webhook_sucesso, json={"content": "✅ SafeDriver: Dados enviados ao Bucket. Driver: " + top_driver})

if __name__ == "__main__":
    try:
        motor = MotorSafeDriverCloud()
        motor.processar_camada_raw()
        df_prata = motor.processar_camada_prata()
        motor.processar_camada_ouro_e_ml(df_prata)
    except Exception:
        err = traceback.format_exc()
        print(err)
        if os.environ.get("DISCORD_ERRO"):
            requests.post(os.environ.get("DISCORD_ERRO"), json={"content": "❌ Falha: " + err[:1800]})
        sys.exit(1)
