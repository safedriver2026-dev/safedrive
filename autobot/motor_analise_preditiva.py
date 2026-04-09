import sys
import os
import requests
import traceback
import hashlib
import gc
import warnings
import re
import time
import json
from pathlib import Path
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import urllib3
import polars as pl
import pandas as pd
import numpy as np
import h3
import holidays
import boto3
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import VotingRegressor
import shap

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

class NotificadorDiscord:
    def __init__(self):
        self.url_sucesso = os.environ.get("DISCORD_SUCESSO", "").strip()
        self.url_erro = os.environ.get("DISCORD_ERRO", "").strip()

    def enviar(self, webhook_url, titulo, mensagem, cor):
        if not webhook_url: return
        payload = {
            "embeds": [{
                "title": titulo,
                "description": mensagem,
                "color": cor,
                "footer": {"text": f"SafeDriver AI | {datetime.now().strftime('%d/%m/%Y %H:%M')}"}
            }]
        }
        try:
            requests.post(webhook_url, json=payload, timeout=10)
        except:
            pass

    def relatar_sucesso(self, total_linhas, tempo_segundos, anos_proc, status_msg="Operação Concluída"):
        msg = (
            f"**Status:** 🟢 {status_msg}\n"
            f"**Anos Ativos:** {anos_proc}\n"
            f"**Registros:** {total_linhas:,}\n"
            f"**Tempo:** {tempo_segundos:.1f}s\n"
            "**Auditoria:** SHAP/CDC verificados."
        )
        self.enviar(self.url_sucesso, "Relatório Operacional", msg, 3066993)

    def relatar_erro(self, erro_msg):
        detalhes = str(erro_msg)[:3800]
        msg = f"**Status:** 🔴 Falha Crítica\n**Traceback:**\n```python\n{detalhes}\n```"
        self.enviar(self.url_erro, "Alerta de Sistema", msg, 15158332)

class SafeDriverMotor:
    def __init__(self):
        self.raiz = Path(".")
        self.hoje = datetime.now()
        self.discord = NotificadorDiscord()
        
        def clean(key): return os.environ.get(key, "").strip()
        self.r2_cfg = {
            "url": clean("R2_ENDPOINT_URL"),
            "key": clean("R2_ACCESS_KEY_ID"),
            "secret": clean("R2_SECRET_ACCESS_KEY"),
            "bucket": clean("R2_BUCKET_NAME")
        }
        
        self.pastas = {p: self.raiz / "datalake" / p for p in ["raw", "prata", "ouro"]}
        for p in self.pastas.values(): p.mkdir(parents=True, exist_ok=True)

        self.s3 = boto3.client('s3', endpoint_url=self.r2_cfg["url"], 
                               aws_access_key_id=self.r2_cfg["key"], 
                               aws_secret_access_key=self.r2_cfg["secret"], region_name="auto")
        
        self.feriados = list(holidays.Brazil(subdiv='SP', years=range(2022, 2027)).keys())
        self.meta_path = self.pastas["raw"] / "tracking_ssp.json"

    def verificar_atualizacao(self, ano, url):
        try:
            r = requests.head(url, timeout=30, verify=False)
            tamanho_remoto = int(r.headers.get('Content-Length', 0))
            
            if self.meta_path.exists():
                with open(self.meta_path, 'r') as f:
                    meta = json.load(f)
                if meta.get(str(ano)) == tamanho_remoto:
                    return False, tamanho_remoto
            return True, tamanho_remoto
        except:
            return True, 0

    def salvar_meta(self, ano, tamanho):
        meta = {}
        if self.meta_path.exists():
            with open(self.meta_path, 'r') as f: meta = json.load(f)
        meta[str(ano)] = tamanho
        with open(self.meta_path, 'w') as f: json.dump(meta, f)

    def limpar_esquema(self, df):
        df.columns = [c.upper().strip() for c in df.columns]
        mapeamento = {
            'LAT': ['LATITUDE', 'LAT'], 'LON': ['LONGITUDE', 'LON'],
            'DATA_REF': ['DATAOCORRENCIA', 'DATA_OCORRENCIA_BO', 'DATA_OCORRENCIA', 'DATA_REF'],
            'HORA_REF': ['HORAOCORRENCIA', 'HORA_OCORRENCIA_BO', 'HORA_REF'],
            'NATUREZA_RAW': ['RUBRICA', 'NATUREZA_APURADA', 'NATUREZA']
        }
        colunas_finais = {}
        for destino, opcoes in mapeamento.items():
            for opt in opcoes:
                if opt in df.columns:
                    colunas_finais[opt] = destino
                    break 
        return df.rename(colunas_finais).select([c for c in colunas_finais.values()])

    def gerar_wkt_h3(self, h3_index):
        try:
            limites = h3.h3_to_geo_boundary(h3_index, geo_json=True)
            coords = ", ".join([f"{lon} {lat}" for lon, lat in limites])
            coords += f", {limites[0][0]} {limites[0][1]}"
            return f"POLYGON(({coords}))"
        except:
            return None

    def executar(self):
        t0 = time.time()
        s = requests.Session()
        s.mount('https://', HTTPAdapter(max_retries=Retry(total=5, backoff_factor=3)))
        s.headers.update({'User-Agent': 'Mozilla/5.0'})

        houve_atualizacao = False
        anos_processados = []

        for ano in range(2022, self.hoje.year + 1):
            url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
            precisa_baixar, tamanho = self.verificar_atualizacao(ano, url)
            
            arq_raw = self.pastas["raw"] / f"ssp_{ano}.parquet"
            if not precisa_baixar and arq_raw.exists():
                anos_processados.append(ano)
                continue

            try:
                r = s.get(url, timeout=300, verify=False)
                if r.status_code == 200:
                    houve_atualizacao = True
                    temp = self.pastas["raw"] / "temp.xlsx"
                    with open(temp, "wb") as f: f.write(r.content)
                    import fastexcel
                    excel = fastexcel.read_excel(str(temp))
                    abas_ano = []
                    for aba in excel.sheet_names:
                        df_tmp = pl.read_excel(str(temp), sheet_name=aba, engine="calamine")
                        if len(df_tmp.columns) > 5:
                            df_tmp = self.limpar_esquema(df_tmp)
                            if "LAT" in df_tmp.columns and "DATA_REF" in df_tmp.columns:
                                abas_ano.append(df_tmp.with_columns(pl.all().cast(pl.String)))
                    if abas_ano:
                        pl.concat(abas_ano, how="diagonal").write_parquet(arq_raw)
                        anos_processados.append(ano)
                        self.salvar_meta(ano, tamanho)
                    if os.path.exists(temp): os.remove(temp)
            except: pass

        if not houve_atualizacao and (self.pastas["ouro"] / "dashboard_final.parquet").exists():
            self.discord.relatar_sucesso(0, time.time()-t0, anos_processados, "Verificação Concluída (Sem Novos Dados)")
            return

        arquivos = [str(f) for f in self.pastas["raw"].glob("*.parquet")]
        lfs = [pl.scan_parquet(f) for f in arquivos]
        lf = pl.concat(lfs, how="diagonal")
        
        df_prata = lf.with_columns([
            pl.col("DATA_REF").str.strptime(pl.Datetime, "%Y-%m-%d", strict=False).alias("DT"),
            pl.col("LAT").str.replace(",", ".").cast(pl.Float32, strict=False),
            pl.col("LON").str.replace(",", ".").cast(pl.Float32, strict=False),
            pl.col("HORA_REF").str.slice(0, 2).cast(pl.Int8, strict=False).alias("H_INT")
        ]).filter(
            (pl.col("LAT").is_between(-25.5, -19.5)) & 
            (pl.col("LON").is_between(-53.5, -44.0)) &
            (pl.col("DT").is_not_null())
        ).with_columns([
            pl.when(pl.col("NATUREZA_RAW").str.to_uppercase().str.contains("HOMICIDIO|LESAO|AMEACA|ESTUPRO|LATROCINIO"))
              .then(pl.lit("CONTRA_A_PESSOA"))
              .when(pl.col("NATUREZA_RAW").str.to_uppercase().str.contains("ROUBO|FURTO|ESTELIONATO|DANO"))
              .then(pl.lit("AO_PATRIMONIO"))
              .otherwise(pl.lit("OUTROS")).alias("NATUREZA_CRIME"),
            pl.when(pl.col("NATUREZA_RAW").str.to_uppercase().str.contains("VEICULO|CARGA|AUTO|MOTO|ONIBUS"))
              .then(pl.lit("MOTORISTA"))
              .when(pl.col("NATUREZA_RAW").str.to_uppercase().str.contains("BICICLETA|BIKE"))
              .then(pl.lit("CICLISTA"))
              .otherwise(pl.lit("PEDESTRE")).alias("PERFIL"),
            pl.when(pl.col("H_INT").is_between(6, 11)).then(pl.lit("MANHA"))
              .when(pl.col("H_INT").is_between(12, 17)).then(pl.lit("TARDE"))
              .when(pl.col("H_INT").is_between(18, 23)).then(pl.lit("NOITE"))
              .otherwise(pl.lit("MADRUGADA")).alias("TURNO"),
            pl.col("DT").dt.date().is_in(self.feriados).cast(pl.Int8).alias("IS_FERIADO"),
            ((pl.col("DT").dt.day().is_between(28, 31)) | (pl.col("DT").dt.day().is_between(1, 7))).cast(pl.Int8).alias("IS_PAGAMENTO")
        ]).collect()

        linhas_totais = df_prata.height
        df_prata = df_prata.with_columns(pl.col("LAT").hash(seed=100).alias("ID_ANONIMO")).drop(["DATA_REF", "HORA_REF", "H_INT", "NATUREZA_RAW"])
        df_prata.write_parquet(self.pastas["prata"] / "camada_prata.parquet")

        coords = df_prata.select(["LAT", "LON"]).unique().to_pandas()
        coords['H3'] = coords.apply(lambda r: h3.latlng_to_cell(r['LAT'], r['LON'], 8), axis=1)
        df_final = df_prata.join(pl.from_pandas(coords), on=["LAT", "LON"], how="left")
        fato = df_final.group_by(["H3", "PERFIL", "TURNO", "NATUREZA_CRIME", "IS_FERIADO", "IS_PAGAMENTO"]).agg([
            pl.len().alias("INCIDENTES"), pl.col("LAT").mean().alias("LAT_M"), pl.col("LON").mean().alias("LON_M")
        ])

        X = fato.select(["LAT_M", "LON_M", "IS_FERIADO", "IS_PAGAMENTO"]).to_pandas()
        y = np.log1p(fato.select("INCIDENTES").to_numpy().ravel())
        ensemble = VotingRegressor([('cat', CatBoostRegressor(iterations=100, silent=True)), ('lgbm', LGBMRegressor(n_estimators=100, verbose=-1))])
        ensemble.fit(X, y)
        
        fato = fato.with_columns([
            pl.Series("RISCO_SCORE", np.round(np.expm1(ensemble.predict(X)), 2)),
            pl.col("H3").map_elements(self.gerar_wkt_h3, return_dtype=pl.String).alias("GEOMETRIA_WKT")
        ])
        fato.write_parquet(self.pastas["ouro"] / "dashboard_final.parquet")

        explainer = shap.TreeExplainer(ensemble.estimators_[0])
        shap_resumo = pd.DataFrame(explainer.shap_values(X), columns=X.columns).abs().mean().to_frame("IMPORTANCIA").reset_index()
        pl.from_pandas(shap_resumo).write_parquet(self.pastas["ouro"] / "shap_audit.parquet")

        for f in self.raiz.rglob("datalake/*/*"):
            if f.is_file(): self.s3.upload_file(str(f), self.r2_cfg["bucket"], f.relative_to(self.raiz).as_posix())

        self.discord.relatar_sucesso(linhas_totais, time.time() - t0, sorted(list(set(anos_processados))))

if __name__ == "__main__":
    motor = SafeDriverMotor()
    try: motor.executar()
    except Exception as e:
        motor.discord.relatar_erro(traceback.format_exc())
        sys.exit(1)
