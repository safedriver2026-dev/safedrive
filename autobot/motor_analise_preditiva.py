import sys, os, requests, traceback, hashlib, gc, warnings, re, time, json
from pathlib import Path
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3, polars as pl, pandas as pd, numpy as np
import h3, holidays, boto3, shap
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

class Telemetria:
    def __init__(self):
        self.sucesso = os.environ.get("DISCORD_SUCESSO", "").strip()
        self.erro = os.environ.get("DISCORD_ERRO", "").strip()

    def notificar(self, url, titulo, corpo, cor):
        if not url: return
        payload = {"embeds": [{"title": titulo, "description": corpo, "color": cor, "footer": {"text": f"SafeDriver AI | {datetime.now().strftime('%H:%M')}"}}]}
        try: requests.post(url, json=payload, timeout=10)
        except: pass

class SafeDriver:
    def __init__(self):
        self.t_inicio = time.time()
        self.discord = Telemetria()
        self.pastas = {p: Path(f"datalake/{p}") for p in ["raw", "prata", "ouro"]}
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)
        
        cfg = {k: os.environ.get(k, "").strip() for k in ["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"]}
        self.s3 = boto3.client('s3', endpoint_url=cfg["R2_ENDPOINT_URL"], aws_access_key_id=cfg["R2_ACCESS_KEY_ID"], aws_secret_access_key=cfg["R2_SECRET_ACCESS_KEY"], region_name="auto")
        self.bucket = cfg["R2_BUCKET_NAME"]
        self.feriados = list(holidays.Brazil(subdiv='SP', years=range(2022, 2027)).keys())
        self.meta = self.pastas["raw"] / "meta.json"

    def cdc_check(self, ano, url):
        try:
            r = requests.head(url, timeout=30, verify=False)
            size = int(r.headers.get('Content-Length', 0))
            if self.meta.exists():
                with open(self.meta, 'r') as f:
                    if json.load(f).get(str(ano)) == size: return False, size
            return True, size
        except: return True, 0

    def wkt(self, h3_id):
        try:
            b = h3.h3_to_geo_boundary(h3_id, geo_json=True)
            pts = ", ".join([f"{ln} {lt}" for ln, lt in b])
            return f"POLYGON(({pts}, {b[0][0]} {b[0][1]}))"
        except: return None

    def processar(self):
        s = requests.Session()
        s.mount('https://', HTTPAdapter(max_retries=Retry(total=3)))
        novo, anos = False, []

        for ano in range(2022, datetime.now().year + 1):
            url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            run, sz = self.cdc_check(ano, url)
            path = self.pastas["raw"] / f"ssp_{ano}.parquet"
            if not run and path.exists(): anos.append(ano); continue

            r = s.get(url, timeout=300, verify=False)
            if r.status_code == 200:
                novo = True
                tmp = self.pastas["raw"] / "tmp.xlsx"
                with open(tmp, "wb") as f: f.write(r.content)
                import fastexcel
                ex = fastexcel.read_excel(str(tmp))
                abas = []
                for n in ex.sheet_names:
                    df = pl.read_excel(str(tmp), sheet_name=n, engine="calamine")
                    if len(df.columns) > 5:
                        df.columns = [c.upper().strip() for c in df.columns]
                        m = {'LAT':['LATITUDE','LAT'],'LON':['LONGITUDE','LON'],'D':['DATAOCORRENCIA','DATA_REF'],'H':['HORAOCORRENCIA','HORA_REF'],'N':['RUBRICA','NATUREZA']}
                        f_cols = {v[0]: k for k, v in m.items() if any(x in df.columns for x in v)}
                        if 'LAT' in f_cols.values(): abas.append(df.rename(f_cols).select(list(f_cols.values())).with_columns(pl.all().cast(pl.String)))
                if abas:
                    pl.concat(abas, how="diagonal").write_parquet(path)
                    anos.append(ano)
                    m_data = json.load(open(self.meta)) if self.meta.exists() else {}
                    m_data[str(ano)] = sz
                    json.dump(m_data, open(self.meta, 'w'))
                os.remove(tmp)

        if not novo and (self.pastas["ouro"] / "dashboard_final.parquet").exists():
            self.discord.notificar(self.sucesso, "SafeDriver Sync", "Sem novos dados na SSP.", 3066993)
            return

        lf = pl.concat([pl.scan_parquet(f) for f in self.pastas["raw"].glob("*.parquet")], how="diagonal")
        prata = lf.with_columns([
            pl.col("D").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).alias("DT"),
            pl.col("LAT").str.replace(",",".").cast(pl.Float32, strict=False),
            pl.col("LON").str.replace(",",".").cast(pl.Float32, strict=False),
            pl.col("H").str.slice(0,2).cast(pl.Int8, strict=False).alias("HR")
        ]).filter(pl.col("LAT").is_between(-25.5, -19.5)).with_columns([
            pl.when(pl.col("N").str.to_uppercase().str.contains("ROUBO|FURTO")).then(pl.lit("PATRIMONIO")).otherwise(pl.lit("PESSOA")).alias("NATUREZA_CRIME"),
            pl.when(pl.col("HR").is_between(6,18)).then(pl.lit("DIA")).otherwise(pl.lit("NOITE")).alias("TURNO"),
            pl.col("DT").dt.date().is_in(self.feriados).cast(pl.Int8).alias("IS_FERIADO"),
            ((pl.col("DT").dt.day().is_between(28,31))|(pl.col("DT").dt.day().is_between(1,7))).cast(pl.Int8).alias("IS_PAGAMENTO")
        ]).collect()

        prata.with_columns(pl.col("LAT").hash().alias("ID_ANONIMO")).drop(["D","H","HR","N"]).write_parquet(self.pastas["prata"] / "camada_prata.parquet")
        
        c = prata.select(["LAT","LON"]).unique().to_pandas()
        c['H3'] = c.apply(lambda r: h3.latlng_to_cell(r['LAT'], r['LON'], 8), axis=1)
        fato = prata.join(pl.from_pandas(c), on=["LAT","LON"]).group_by(["H3","TURNO","NATUREZA_CRIME","IS_FERIADO","IS_PAGAMENTO"]).agg([pl.len().alias("INCIDENTES"), pl.col("LAT").mean().alias("LAT_M"), pl.col("LON").mean().alias("LON_M")])
        
        X = fato.select(["LAT_M","LON_M","IS_FERIADO","IS_PAGAMENTO"]).to_pandas()
        y = np.log1p(fato.select("INCIDENTES").to_numpy().ravel())
        ens = VotingRegressor([('c', CatBoostRegressor(iterations=100, silent=True)), ('l', LGBMRegressor(n_estimators=100, verbose=-1))]).fit(X, y)
        
        fato.with_columns([
            pl.Series("RISCO_SCORE", np.round(np.expm1(ens.predict(X)), 2)),
            pl.col("H3").map_elements(self.wkt, return_dtype=pl.String).alias("GEOMETRIA_WKT")
        ]).write_parquet(self.pastas["ouro"] / "dashboard_final.parquet")

        sd = pd.DataFrame(shap.TreeExplainer(ens.estimators_[0]).shap_values(X), columns=X.columns).abs().mean().to_frame("IMPORTANCIA").reset_index()
        sd.columns = ["FEATURE", "IMPORTANCIA"]
        pl.from_pandas(sd).write_parquet(self.pastas["ouro"] / "shap_audit.parquet")

        for f in self.pastas["ouro"].glob("*.parquet"):
            self.s3.upload_file(str(f), self.bucket, f"ouro/{f.name}")
        
        self.discord.notificar(self.sucesso, "SafeDriver OK", f"Processados {prata.height:,} registros.", 3066993)

if __name__ == "__main__":
    app = SafeDriver()
    try: app.processar()
    except Exception:
        err = traceback.format_exc(); print(err, file=sys.stderr)
        app.discord.notificar(app.discord.erro, "SafeDriver FAIL", f"
http://googleusercontent.com/immersive_entry_chip/0
