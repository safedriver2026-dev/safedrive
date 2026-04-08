import sys, os, json, requests, traceback, hashlib, gc, warnings, re, unicodedata
from pathlib import Path
from datetime import datetime
import polars as pl
import pandas as pd
import numpy as np
import h3, holidays, boto3
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor

print("[INICIALIZACAO] Motor SafeDriver Total-Vision v9.2 Ativado.", flush=True)
warnings.filterwarnings("ignore")

class SafeDriverTotalVision:
    def __init__(self):
        self.raiz = Path(".")
        self.hoje = datetime.now()
        
        # Credenciais Cloudflare R2
        def clean(key): return os.environ.get(key, "").strip()
        self.r2_cfg = {
            "url": clean("R2_ENDPOINT_URL"),
            "key": clean("R2_ACCESS_KEY_ID"),
            "secret": clean("R2_SECRET_ACCESS_KEY"),
            "bucket": clean("R2_BUCKET_NAME")
        }
        
        self.pastas = {p: self.raiz / "datalake" / p for p in ["raw", "prata", "ouro", "auditoria"]}
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)

        self.s3 = boto3.client('s3', endpoint_url=self.r2_cfg["url"], 
                               aws_access_key_id=self.r2_cfg["key"], 
                               aws_secret_access_key=self.r2_cfg["secret"], region_name="auto")
        
        self.cal_feriados = holidays.Brazil(subdiv='SP', years=range(2022, self.hoje.year + 1))
        self.datas_feriados = list(self.cal_feriados.keys())

    def classificar_natureza(self, rubrica):
        r = str(rubrica).upper()
        # Crimes contra a Pessoa (Foco na integridade física)
        if any(x in r for x in ["HOMICIDIO", "LESAO", "AMEACA", "ESTUPRO", "LATROCINIO", "RIXA"]):
            return "CONTRA_A_PESSOA"
        # Crimes contra o Patrimônio (Foco no objeto/bem)
        if any(x in r for x in ["ROUBO", "FURTO", "ESTELIONATO", "DANO", "RECEPTACAO"]):
            return "AO_PATRIMONIO"
        return "OUTROS"

    def categorizar_perfil(self, rubrica):
        r = str(rubrica).upper()
        if any(x in r for x in ["VEICULO", "CARGA", "AUTO", "MOTO", "ONIBUS"]): return "MOTORISTA"
        if any(x in r for x in ["BICICLETA", "BIKE"]): return "CICLISTA"
        return "PEDESTRE"

    def definir_turno(self, hora):
        try:
            h = int(str(hora)[:2])
            if 6 <= h < 12: return "MANHA"
            if 12 <= h < 18: return "TARDE"
            if 18 <= h < 24: return "NOITE"
            return "MADRUGADA"
        except: return "MADRUGADA"

    def executar_pipeline(self):
        # 1. COLETA (2022-2026)
        print("[COLETOR] Sincronizando Base Histórica...", flush=True)
        for ano in range(2022, self.hoje.year + 1):
            arq_raw = self.pastas["raw"] / f"ssp_{ano}.parquet"
            if arq_raw.exists() and ano < self.hoje.year: continue
            try:
                url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
                r = requests.get(url, timeout=120, verify=False)
                if r.status_code == 200:
                    temp = self.pastas["raw"] / "temp.xlsx"
                    with open(temp, "wb") as f: f.write(r.content)
                    
                    import fastexcel
                    excel = fastexcel.read_excel(str(temp))
                    df_ano = []
                    for aba in excel.sheet_names:
                        df_tmp = pl.read_excel(str(temp), sheet_name=aba, engine="calamine")
                        if len(df_tmp.columns) > 5:
                            df_tmp = df_tmp.rename({c: str(c).upper().strip() for c in df_tmp.columns})
                            df_ano.append(df_tmp.with_columns(pl.all().cast(pl.String)))
                    if df_ano: pl.concat(df_ano, how="diagonal").write_parquet(arq_raw)
                    os.remove(temp)
            except: print(f"[AVISO] Ano {ano} falhou.")

        # 2. REFINAMENTO COM DUPLA CLASSIFICAÇÃO
        print("[REFINADOR] Aplicando Classificação de Natureza e LGPD...", flush=True)
        arquivos = list(self.pastas["raw"].glob("*.parquet"))
        df = pl.concat([pl.read_parquet(str(f)) for f in arquivos], how="diagonal")
        
        df_prata = df.with_columns([
            pl.col("DATAOCORRENCIA").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).alias("DT"),
            pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LAT"),
            pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float32, strict=False).alias("LON"),
            pl.col("NATUREZA").map_elements(self.classificar_natureza, return_dtype=pl.String).alias("NATUREZA_CRIME"),
            pl.col("NATUREZA").map_elements(self.categorizar_perfil, return_dtype=pl.String).alias("PERFIL"),
            pl.col("HORAOCORRENCIA").map_elements(self.definir_turno, return_dtype=pl.String).alias("TURNO")
        ]).filter(pl.col("LAT").is_not_null() & pl.col("DT").is_not_null())

        df_prata = df_prata.with_columns([
            pl.col("DT").dt.date().is_in(self.datas_feriados).cast(pl.Int8).alias("IS_FERIADO"),
            ((pl.col("DT").dt.day().is_between(28, 31)) | (pl.col("DT").dt.day().is_between(1, 7))).cast(pl.Int8).alias("IS_PAGAMENTO")
        ])

        # 3. INTELIGÊNCIA GEOSPACIAL
        coords = df_prata.select(["LAT", "LON"]).unique().to_pandas()
        coords['H3'] = coords.apply(lambda r: h3.latlng_to_cell(r['LAT'], r['LON'], 8), axis=1)
        df_final = df_prata.join(pl.from_pandas(coords), on=["LAT", "LON"], how="left")

        # Fato para o Power BI: Agrupamento por todas as dimensões inteligentes
        fato = df_final.group_by(["H3", "PERFIL", "TURNO", "NATUREZA_CRIME", "IS_FERIADO", "IS_PAGAMENTO"]).agg([
            pl.len().alias("INCIDENTES"),
            pl.col("LAT").mean().alias("LAT_M"),
            pl.col("LON").mean().alias("LON_M")
        ])

        # IA Treino Multidimensional
        X_cols = ["LAT_M", "LON_M", "IS_FERIADO", "IS_PAGAMENTO"]
        X = fato.select(X_cols).to_pandas()
        y = np.log1p(fato.select("INCIDENTES").to_numpy().ravel())
        m1 = CatBoostRegressor(iterations=50, silent=True).fit(X, y)
        fato = fato.with_columns(pl.Series("RISCO_SCORE", np.round(np.expm1(m1.predict(X)), 2)))

        # 4. SYNC FINAL
        fato.write_parquet(self.pastas["ouro"] / "dashboard_final.parquet")
        for f in self.raiz.rglob("datalake/*/*"):
            if f.is_file():
                self.s3.upload_file(str(f), self.r2_cfg["bucket"], f.relative_to(self.raiz).as_posix())

if __name__ == "__main__":
    SafeDriverTotalVision().executar_pipeline()
