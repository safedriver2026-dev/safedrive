import sys
import os
import json
import requests
import traceback
import hashlib
import polars as pl
import pandas as pd
import numpy as np
import h3
import gc
import warnings
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score

print("[SYS_BOOT] Motor Autônomo SafeDriver inicializado. Carregando módulos de análise...", flush=True)
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
        self.storage_client = storage.Client(project="sandbox-suprimentos")
        self.hashes_seguranca = {}
        self.anomalias_detectadas = 0

    def gerar_hash_sha256(self, caminho):
        sha256_hash = hashlib.sha256()
        with open(caminho, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def validar_geografia(self, df):
        return df.filter(
            (pl.col("LAT").between(-25.5, -19.5)) & 
            (pl.col("LON").between(-53.5, -44.0))
        )

    def processar_camada_raw(self):
        print("[WORKER_RAW] Iniciando protocolo DeltaSync e verificação de integridade.", flush=True)
        ano_atual = self.hoje.year
        anos_foco = [ano_atual - 2, ano_atual - 1, ano_atual]
        cabecalho = {'User-Agent': 'Mozilla/5.0'}
        mapeamento = {'DATAOCORRENCIA': 'DATA_OCORRENCIA_BO', 'DATA DO FATO': 'DATA_OCORRENCIA_BO', 'NATUREZA': 'RUBRICA'}

        for ano in anos_foco:
            parquet_raw = self.pastas["raw"] / f"ssp_{ano}.parquet"
            url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            
            if parquet_raw.exists() and ano < ano_atual:
                self.hashes_seguranca[f"ssp_{ano}"] = self.gerar_hash_sha256(parquet_raw)
                continue

            try:
                r = requests.get(url, stream=True, verify=False, timeout=60, headers=cabecalho)
                if r.status_code == 200:
                    temp_xlsx = self.pastas["raw"] / "temp.xlsx"
                    with open(temp_xlsx, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=2*1024*1024): f.write(chunk)
                    
                    import fastexcel
                    excel = fastexcel.read_excel(str(temp_xlsx))
                    df_novo = pl.read_excel(str(temp_xlsx), sheet_name=excel.sheet_names[0], engine="calamine")
                    df_novo = df_novo.rename({c: str(c).upper().strip() for c in df_novo.columns})
                    for v, n in mapeamento.items():
                        if v in df_novo.columns: df_novo = df_novo.rename({v: n})
                    
                    df_novo = df_novo.with_columns(pl.all().cast(pl.String))
                    if parquet_raw.exists():
                        df_final = pl.concat([pl.read_parquet(parquet_raw), df_novo], how="diagonal")
                        df_final = df_final.unique(subset=["NUM_BO"], keep="last")
                    else:
                        df_final = df_novo.unique(subset=["NUM_BO"])

                    df_final.write_parquet(parquet_raw)
                    self.hashes_seguranca[f"ssp_{ano}"] = self.gerar_hash_sha256(parquet_raw)
                    os.remove(temp_xlsx)
                    gc.collect()
            except: pass

    def processar_camada_prata(self):
        print("[WORKER_PRATA] Executando rotinas de higienizacao e isolamento geoespacial.", flush=True)
        arquivos = list(self.pastas["raw"].glob("*.parquet"))
        lf = pl.scan_parquet([str(f) for f in arquivos])
        
        df_prata = (
            lf.with_columns([
                pl.col("DATA_OCORRENCIA_BO").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).alias("DATA_DT"),
                pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LAT"),
                pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LON"),
            ])
            .filter(pl.col("LAT").is_not_null() & pl.col("DATA_DT").is_not_null())
            .unique(subset=["NUM_BO"]) 
            .collect()
        )

        df_prata = self.validar_geografia(df_prata)

        coords = df_prata.select(["LAT", "LON"]).unique().to_pandas()
        coords['H3'] = coords.apply(lambda r: h3.latlng_to_cell(r['LAT'], r['LON'], 8), axis=1)
        
        df_final = df_prata.join(pl.from_pandas(coords), on=["LAT", "LON"], how="left")
        df_final.write_parquet(self.pastas["prata"] / "camada_prata.parquet")
        return df_final

    def aplicar_suavizacao_h3(self, df_fato):
        mapa_inc = dict(zip(df_fato['H3'], df_fato['INCIDENTES']))
        suavizado = []
        for h, inc in zip(df_fato['H3'], df_fato['INCIDENTES']):
            vizinhos = h3.grid_disk(h, 1)
            peso_viz = sum(mapa_inc.get(v, 0) for v in vizinhos if v != h) * 0.4
            suavizado.append(inc + peso_viz)
        return df_fato.with_columns([pl.Series("RISCO_GEO", suavizado)])

    def detectar_anomalias_zscore(self, df_fato):
        arr_inc = df_fato.select("INCIDENTES").to_numpy().ravel()
        media = np.mean(arr_inc)
        desvio = np.std(arr_inc)
        if desvio > 0:
            z_scores = np.abs((arr_inc - media) / desvio)
            self.anomalias_detectadas = int(np.sum(z_scores > 3))
        else:
            self.anomalias_detectadas = 0

    def processar_camada_ouro_e_ml(self, df):
        print("[ML_CORE] Calibrando rede preditiva e gerando metricas de auditoria.", flush=True)
        col_crime = 'NATUREZA_APURADA' if 'NATUREZA_APURADA' in df.columns else 'RUBRICA'
        
        fato = (
            df.with_columns([
                pl.col(col_crime).str.contains("ROUBO").alias("IS_ROUBO"),
                pl.col("DATA_DT").dt.hour().alias("HORA"),
                pl.col("DATA_DT").dt.weekday().alias("DIA_SEMANA")
            ])
            .group_by(["H3", "HORA", "DIA_SEMANA"])
            .agg([pl.col("LAT").mean().alias("LAT"), pl.col("LON").mean().alias("LON"), pl.len().alias("INCIDENTES")])
        )

        self.detectar_anomalias_zscore(fato)
        fato = self.aplicar_suavizacao_h3(fato)

        X_df = fato.select(['LAT', 'LON', 'HORA', 'DIA_SEMANA', 'RISCO_GEO']).to_pandas()
        X = StandardScaler().fit_transform(X_df)
        y = np.log1p(fato.select("INCIDENTES").to_numpy().ravel())
        
        cb = CatBoostRegressor(iterations=150, thread_count=-1, silent=True).fit(X, y)
        lgb = LGBMRegressor(n_estimators=100, n_jobs=-1, importance_type='gain').fit(X, y)
        
        y_pred = (cb.predict(X) * 0.7) + (lgb.predict(X) * 0.3)
        r2 = r2_score(y, y_pred)
        
        manifesto = {
            "timestamp": self.hoje.isoformat(),
            "r2_final": float(r2),
            "total_bo_unicos": df.height,
            "anomalias_estatisticas_z3": self.anomalias_detectadas,
            "hashes_seguranca": self.hashes_seguranca,
            "versao_motor": "5.0.0-autonomo"
        }
        with open(self.pastas["auditoria"] / "auditoria.json", "w") as f:
            json.dump(manifesto, f, indent=4)

        preds = np.round(np.expm1(y_pred), 2)
        fato = fato.with_columns([pl.Series("PREVISAO_FINAL", preds)])
        fato.write_parquet(self.pastas["ouro"] / "dashboard_final.parquet")
        
        print("[CLOUD_SYNC] Transmitindo pacotes processados para o repositorio remoto.", flush=True)
        bucket = self.storage_client.bucket(self.bucket_nome)
        for f in self.raiz.rglob("datalake/*/*.parquet"):
            bucket.blob(str(f.relative_to(self.raiz))).upload_from_filename(str(f))
        
        bucket.blob("datalake/auditoria/auditoria.json").upload_from_filename(
            str(self.pastas["auditoria"] / "auditoria.json")
        )

        if self.webhook_sucesso:
            msg = dict(content="[STATUS] Operacao autonoma concluida. Metricas: R2=" + str(round(r2, 2)) + " | Ocorrencias validadas=" + str(df.height) + " | Anomalias isoladas=" + str(self.anomalias_detectadas))
            requests.post(self.webhook_sucesso, json=msg)

if __name__ == "__main__":
    try:
        motor = MotorSafeDriverCloud()
        motor.processar_camada_raw()
        df_prata = motor.processar_camada_prata()
        motor.processar_camada_ouro_e_ml(df_prata)
    except Exception:
        err = traceback.format_exc()
        print("[ALERTA] Excecao nao tratada capturada durante a execucao:", flush=True)
        print(err, flush=True)
        webhook_erro = os.environ.get("DISCORD_ERRO")
        if webhook_erro:
            nl = chr(10)
            msg_erro = dict(content="[FALHA_SISTEMA] Interrupcao no ciclo preditivo:" + nl + str(err)[:1800])
            try: requests.post(webhook_erro, json=msg_erro)
            except: pass
        sys.exit(1)
