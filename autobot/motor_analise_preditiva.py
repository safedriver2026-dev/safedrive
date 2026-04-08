import sys, os, json, requests, traceback, hashlib, gc, warnings, re, unicodedata
from pathlib import Path
from datetime import datetime
import polars as pl
import pandas as pd
import numpy as np
import h3, holidays, boto3
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
print("[SISTEMA] Motor SafeDriver iniciado.", flush=True)

class SafeDriverMotor:
    def __init__(self):
        self.raiz = Path(".")
        self.hoje = datetime.now()
        
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

    def normalizador_semantico(self, df):
        # Transforma tudo em Caps e remove espaços/pontos para comparação
        df = df.rename({c: re.sub(r'[^A-Z0-9]', '_', c.upper().strip()) for c in df.columns})
        
        # Mapeamento robusto para os nomes de coluna que variam entre 2022-2026
        mapeamento = {
            'LATITUDE': 'LAT', 'LAT': 'LAT',
            'LONGITUDE': 'LON', 'LON': 'LON',
            'DATAOCORRENCIA': 'DATA_REF', 'DATA_OCORRENCIA_BO': 'DATA_REF', 'DATA_OCORRENCIA': 'DATA_REF',
            'HORAOCORRENCIA': 'HORA_REF', 'HORA_OCORRENCIA_BO': 'HORA_REF', 'HORA_OCORRENCIA': 'HORA_REF',
            'RUBRICA': 'NATUREZA_RAW', 'NATUREZA_APURADA': 'NATUREZA_RAW'
        }
        
        colunas_atuais = df.columns
        renomear = {}
        for original, alvo in mapeamento.items():
            if original in colunas_atuais and alvo not in renomear.values():
                renomear[original] = alvo
        
        return df.rename(renomear)

    def classificar_natureza(self, rubrica):
        r = str(rubrica).upper()
        if any(x in r for x in ["HOMICIDIO", "LESAO", "AMEACA", "ESTUPRO", "LATROCINIO", "RIXA"]):
            return "CONTRA_A_PESSOA"
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

    def executar(self):
        # 1. Extração
        print("Sincronizando base histórica (2022-2026)...", flush=True)
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
                            # Normaliza cada aba individualmente antes de empilhar
                            df_tmp = self.normalizador_semantico(df_tmp)
                            df_ano.append(df_tmp.with_columns(pl.all().cast(pl.String)))
                    if df_ano:
                        pl.concat(df_ano, how="diagonal").write_parquet(arq_raw)
                    os.remove(temp)
            except: print(f"Aviso: Falha ao baixar ano {ano}.")

        # 2. Processamento
        print("Processando limpeza e anonimização...", flush=True)
        arquivos = list(self.pastas["raw"].glob("*.parquet"))
        df = pl.concat([pl.read_parquet(str(f)) for f in arquivos], how="diagonal")
        
        # Agora usamos os nomes normalizados (DATA_REF, LAT, LON, etc)
        df_prata = df.with_columns([
            pl.col("DATA_REF").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).alias("DT"),
            pl.col("LAT").str.replace(",", ".").cast(pl.Float32, strict=False),
            pl.col("LON").str.replace(",", ".").cast(pl.Float32, strict=False),
            pl.col("NATUREZA_RAW").map_elements(self.classificar_natureza, return_dtype=pl.String).alias("NATUREZA_CRIME"),
            pl.col("NATUREZA_RAW").map_elements(self.categorizar_perfil, return_dtype=pl.String).alias("PERFIL"),
            pl.col("HORA_REF").map_elements(self.definir_turno, return_dtype=pl.String).alias("TURNO"),
            pl.col("DATA_REF").str.strptime(pl.Date, "%Y-%m-%d", strict=False).is_in(self.datas_feriados).cast(pl.Int8).alias("IS_FERIADO"),
            ((pl.col("DATA_REF").str.slice(8, 2).cast(pl.Int8).is_between(28, 31)) | 
             (pl.col("DATA_REF").str.slice(8, 2).cast(pl.Int8).is_between(1, 7))).cast(pl.Int8).alias("IS_PAGAMENTO")
        ]).filter(pl.col("LAT").is_not_null() & pl.col("DT").is_not_null())

        df_prata = df_prata.with_columns(
            pl.col("LAT").hash(seed=100).alias("ID_ANONIMO")
        ).drop(["DATA_REF", "HORA_REF"])
        
        df_prata.write_parquet(self.pastas["prata"] / "camada_prata.parquet")

        # 3. Predição
        print("Gerando indicadores de risco...", flush=True)
        coords = df_prata.select(["LAT", "LON"]).unique().to_pandas()
        coords['H3'] = coords.apply(lambda r: h3.latlng_to_cell(r['LAT'], r['LON'], 8), axis=1)
        df_final = df_prata.join(pl.from_pandas(coords), on=["LAT", "LON"], how="left")

        fato = df_final.group_by(["H3", "PERFIL", "TURNO", "NATUREZA_CRIME", "IS_FERIADO", "IS_PAGAMENTO"]).agg([
            pl.len().alias("INCIDENTES"),
            pl.col("LAT").mean().alias("LAT_M"),
            pl.col("LON").mean().alias("LON_M")
        ])

        X = fato.select(["LAT_M", "LON_M", "IS_FERIADO", "IS_PAGAMENTO"]).to_pandas()
        y = np.log1p(fato.select("INCIDENTES").to_numpy().ravel())
        modelo = CatBoostRegressor(iterations=50, silent=True).fit(X, y)
        fato = fato.with_columns(pl.Series("RISCO_SCORE", np.round(np.expm1(modelo.predict(X)), 2)))

        # 4. Sincronização
        fato.write_parquet(self.pastas["ouro"] / "dashboard_final.parquet")
        for f in self.raiz.rglob("datalake/*/*"):
            if f.is_file():
                self.s3.upload_file(str(f), self.r2_cfg["bucket"], f.relative_to(self.raiz).as_posix())

if __name__ == "__main__":
    SafeDriverMotor().executar()
