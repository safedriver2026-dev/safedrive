# autobot/motor_analise_preditiva.py
import sys, os, requests, traceback, hashlib, warnings, time, json, unicodedata
from pathlib import Path
from datetime import datetime, timedelta
from io import BytesIO
import urllib3
import polars as pl
import pandas as pd
import numpy as np
import h3
import holidays
import boto3
import joblib
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
import shap

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES GLOBAIS
# ══════════════════════════════════════════════════════════════════════════════

VERSAO_PIPELINE    = "4.2.0"
VERSAO_FEATURES    = "v4"
H3_RESOLUCAO       = 8
MIN_REGISTROS_OURO = 500

# Prefixo R2 exato conforme estrutura: safedriver/safedriver/datalake/raw/
R2_PREFIXO_RAW     = "safedriver/datalake/raw/"
R2_PREFIXO_PRATA   = "safedriver/datalake/prata/"
R2_PREFIXO_OURO    = "safedriver/datalake/ouro/"
R2_PREFIXO_MODELO  = "safedriver/datalake/modelos/"
R2_TRACKING        = "safedriver/datalake/raw/tracking_ssp.json"

PESO_PENAL_BASE = {
    "HOMICIDIO DOLOSO":            10.0,
    "LATROCINIO":                  10.0,
    "EXTORSAO MEDIANTE SEQUESTRO": 10.0,
    "ROUBO DE VEICULO":            10.0,
    "ROUBO DE MOTOCICLETA":        10.0,
    "ROUBO DE CARGA":              10.0,
    "ATROPELAMENTO":                9.0,
    "ESTUPRO":                      9.0,
    "FURTO DE VEICULO":             8.0,
    "FURTO DE MOTOCICLETA":         8.0,
    "ROUBO":                        7.0,
    "ACIDENTE COM MOTOCICLETA":     7.0,
    "LESAO CORPORAL DOLOSA":        6.0,
    "PORTE ILEGAL DE ARMA":         6.0,
    "TRAFICO DE ENTORPECENTES":     5.0,
    "ACIDENTE DE TRANSITO":         5.0,
    "FURTO":                        4.0,
}

MULTIPLICADOR_PERFIL = {
    "MOTORISTA":    {"ROUBO DE VEICULO": 1.5, "FURTO DE VEICULO": 1.5, "ROUBO DE CARGA": 1.4, "LATROCINIO": 1.3},
    "MOTOCICLISTA": {"ROUBO DE MOTOCICLETA": 1.5, "FURTO DE MOTOCICLETA": 1.5, "ACIDENTE COM MOTOCICLETA": 1.5},
    "PEDESTRE":     {"ROUBO": 1.4, "LESAO CORPORAL DOLOSA": 1.4, "ATROPELAMENTO": 1.5, "ESTUPRO": 1.5},
    "CICLISTA":     {"ROUBO": 1.3, "ATROPELAMENTO": 1.8, "ACIDENTE DE TRANSITO": 1.5},
}

SP_LAT_MIN, SP_LAT_MAX = -25.3, -19.8
SP_LON_MIN, SP_LON_MAX = -53.2, -44.0

SINONIMOS = {
    "NOME_DEPARTAMENTO":  ["DEPARTAMENTO", "DEPTO"],
    "NOME_SECCIONAL":     ["SECCIONAL"],
    "NOME_DELEGACIA":     ["DELEGACIA"],
    "NOME_MUNICIPIO":     ["MUNICIPIO", "MUN", "CIDADE"],
    "LOGRADOURO":         ["RUA", "ENDERECO"],
    "NUMERO_LOGRADOURO":  ["NUMERO", "NUM_LOGRADOURO"],
    "BAIRRO":             ["NOME_BAIRRO"],
    "LATITUDE":           ["LAT", "COORD_LAT", "LATITUDE_BO"],
    "LONGITUDE":          ["LON", "LNG", "COORD_LON", "LONGITUDE_BO"],
    "DATA_OCORRENCIA_BO": ["DATA_OCORRENCIA", "DT_OCORRENCIA"],
    "HORA_OCORRENCIA_BO": ["HORA_OCORRENCIA", "HR_OCORRENCIA"],
    "RUBRICA":            ["TIPO_CRIME", "NATUREZA_CRIMINAL"],
    "DESCR_CONDUTA":      ["CONDUTA", "DESCRICAO_CONDUTA"],
    "NATUREZA_APURADA":   ["NATUREZA", "NAT_APURADA"],
    "DATA_REGISTRO":      ["DT_REGISTRO", "DATA_BO"],
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════════════

def hora_brasilia() -> datetime:
    return datetime.utcnow() - timedelta(hours=3)


def normalizar_texto(s):
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.upper().strip()


def sanitizar_secret(valor: str) -> str:
    if not valor:
        return valor
    valor = valor.replace("\n", "").replace("\r", "").strip()
    return "".join(c for c in valor if c.isprintable())


def gerar_run_id() -> str:
    ts = hora_brasilia().isoformat()
    return hashlib.sha256(ts.encode()).hexdigest()[:12]


def criar_cliente_bq(projeto: str, cred_json: str) -> bigquery.Client:
    info = json.loads(cred_json)
    creds = service_account.Credentials.from_service_account_info(info)
    return bigquery.Client(project=projeto, credentials=creds)


def enviar_para_bigquery(df_pl: pl.DataFrame, tabela: str, projeto: str,
                          dataset: str, cred_json: str, modo: str = "WRITE_TRUNCATE"):
    client = criar_cliente_bq(projeto, cred_json)
    tabela_id = f"{projeto}.{dataset}.{tabela}"
    job = client.load_table_from_dataframe(
        df_pl.to_pandas(),
        tabela_id,
        job_config=bigquery.LoadJobConfig(write_disposition=modo)
    )
    job.result()
    print(f"  ✅ BQ → {tabela_id}: {len(df_pl):,} registros")


# ══════════════════════════════════════════════════════════════════════════════
# TELEMETRIA DISCORD
# ══════════════════════════════════════════════════════════════════════════════

class Telemetria:
    def __init__(self):
        self.url_sucesso = sanitizar_secret(os.environ.get("DISCORD_SUCESSO", ""))
        self.url_erro    = sanitizar_secret(os.environ.get("DISCORD_ERRO", ""))

    def _post(self, url: str, payload: dict):
        if not url:
            return
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"  ⚠️  Discord falhou: {e}")

    def notificar_sucesso(self, run_id, tempo, n_prata, n_ouro,
                          mae, r2, mae_anterior, melhoria, status_bq):
        emoji_r2  = "🟢" if r2 >= 0.85 else "🟡" if r2 >= 0.70 else "🔴"
        emoji_mae = "📈" if melhoria > 0 else "📉"
        self._post(self.url_sucesso, {"embeds": [{"title": "✅ SafeDriver Pipeline Concluído",
            "color": 3066993,
            "fields": [
                {"name": "Run ID",        "value": run_id,                         "inline": True},
                {"name": "Versão",        "value": VERSAO_PIPELINE,                "inline": True},
                {"name": "Tempo",         "value": f"{tempo:.0f}s",                "inline": True},
                {"name": "Prata",         "value": f"{n_prata:,} registros",       "inline": True},
                {"name": "Ouro",          "value": f"{n_ouro:,} hexágonos",        "inline": True},
                {"name": "BigQuery",      "value": status_bq,                      "inline": True},
                {"name": f"{emoji_r2} R²","value": f"{r2:.4f}",                    "inline": True},
                {"name": "MAE",           "value": f"{mae:.2f}",                   "inline": True},
                {"name": f"{emoji_mae} vs Anterior", "value": f"{melhoria:+.2f}", "inline": True},
            ],
            "timestamp": hora_brasilia().isoformat()
        }]})

    def notificar_erro(self, run_id, titulo, detalhe):
        self._post(self.url_erro, {"embeds": [{"title": f"🚨 {titulo}",
            "color": 15158332,
            "fields": [
                {"name": "Run ID",  "value": run_id},
                {"name": "Detalhe", "value": str(detalhe)[:1000]},
            ],
            "timestamp": hora_brasilia().isoformat()
        }]})


# ══════════════════════════════════════════════════════════════════════════════
# MEMÓRIA DO MODELO — residual learning entre execuções
# ══════════════════════════════════════════════════════════════════════════════

class MemoriaModelo:
    CHAVE_HISTORICO = "safedriver/datalake/modelos/historico_erros.parquet"
    CHAVE_PTR       = "safedriver/datalake/modelos/modelo_atual.json"

    def __init__(self, s3):
        self.s3 = s3
        self.bucket = sanitizar_secret(os.environ.get("R2_BUCKET_NAME", ""))

    def _existe(self, chave: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.bucket, Key=chave)
            return True
        except Exception:
            return False

    def carregar_historico(self) -> pd.DataFrame:
        if not self._existe(self.CHAVE_HISTORICO):
            return pd.DataFrame(columns=["H3_R8", "ERRO_MEDIO", "TENDENCIA", "EXECUCOES"])
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.CHAVE_HISTORICO)
        return pd.read_parquet(BytesIO(obj["Body"].read()))

    def salvar_historico(self, df: pd.DataFrame):
        buf = BytesIO()
        df.to_parquet(buf, index=False)
        buf.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=self.CHAVE_HISTORICO, Body=buf.getvalue())

    def carregar_ponteiro(self) -> dict:
        if not self._existe(self.CHAVE_PTR):
            return {"run_id": None, "mae": None, "r2": None}
        obj = self.s3.get_object(Bucket=self.bucket, Key=self.CHAVE_PTR)
        return json.loads(obj["Body"].read())

    def salvar_ponteiro(self, run_id: str, mae: float, r2: float):
        payload = json.dumps({"run_id": run_id, "mae": mae, "r2": r2,
                               "versao": VERSAO_PIPELINE,
                               "timestamp": hora_brasilia().isoformat()})
        self.s3.put_object(Bucket=self.bucket, Key=self.CHAVE_PTR,
                           Body=payload.encode())

    def salvar_modelo(self, modelo, run_id: str):
        buf = BytesIO()
        joblib.dump(modelo, buf)
        buf.seek(0)
        chave = f"safedriver/datalake/modelos/modelo_{run_id}.pkl"
        self.s3.put_object(Bucket=self.bucket, Key=chave, Body=buf.getvalue())


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SafeDriver:

    def __init__(self):
        self.run_id   = gerar_run_id()
        self.t_inicio = time.time()
        self.discord  = Telemetria()
        self._log("pipeline_iniciado", {"versao": VERSAO_PIPELINE, "features": VERSAO_FEATURES})

        # Sanitizar todos os secrets antes de usar
        endpoint   = sanitizar_secret(os.environ.get("R2_ENDPOINT_URL", ""))
        access_key = sanitizar_secret(os.environ.get("R2_ACCESS_KEY_ID", ""))
        secret_key = sanitizar_secret(os.environ.get("R2_SECRET_ACCESS_KEY", ""))
        self.bucket = sanitizar_secret(os.environ.get("R2_BUCKET_NAME", ""))

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )
        self.memoria   = MemoriaModelo(self.s3)
        self._parquet_chaves: list[str] = []

    def _log(self, evento: str, dados: dict = {}):
        print(f"  [{self.run_id}] {evento}: {dados}", flush=True)

    def _upload_bytes(self, chave: str, conteudo: bytes):
        self.s3.put_object(Bucket=self.bucket, Key=chave, Body=conteudo)

    def _baixar_bytes(self, chave: str) -> bytes:
        obj = self.s3.get_object(Bucket=self.bucket, Key=chave)
        return obj["Body"].read()

    # ── 1. Sincronizar RAW ────────────────────────────────────────────────────

    def sincronizar_raw(self):
        """
        Lista arquivos parquet em safedriver/datalake/raw/
        Estrutura real conforme imagem R2:
          ssp_2022.parquet, ssp_2023.parquet, ssp_2024.parquet,
          ssp_2025.parquet, ssp_2026.parquet, tracking_ssp.json
        """
        self._log("sincronizar_raw_inicio")

        paginator = self.s3.get_paginator("list_objects_v2")
        chaves = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=R2_PREFIXO_RAW):
            for obj in page.get("Contents", []):
                chave = obj["Key"]
                if chave.endswith(".parquet"):
                    chaves.append(chave)

        if not chaves:
            raise RuntimeError(
                f"Nenhum arquivo .parquet encontrado em '{R2_PREFIXO_RAW}' "
                f"no bucket '{self.bucket}'. Verifique o prefixo e o bucket."
            )

        self._parquet_chaves = sorted(chaves)
        self._log("sincronizar_raw_concluido", {
            "arquivos": [Path(c).name for c in self._parquet_chaves]
        })

    # ── 2. Construir Prata ────────────────────────────────────────────────────

    def _resolver_colunas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resolve sinônimos históricos de colunas renomeadas pela SSP."""
        mapa = {}
        cols_upper = {c.upper(): c for c in df.columns}
        for canonical, sinonimos in SINONIMOS.items():
            if canonical not in df.columns:
                for s in sinonimos:
                    if s in cols_upper:
                        mapa[cols_upper[s]] = canonical
                        break
        return df.rename(columns=mapa)

    def construir_prata(self) -> pl.DataFrame:
        self._log("prata_inicio")
        frames = []

        for chave in self._parquet_chaves:
            self._log("prata_lendo", {"arquivo": Path(chave).name})
            conteudo = self._baixar_bytes(chave)

            df = pd.read_parquet(BytesIO(conteudo))
            df.columns = [normalizar_texto(str(c)) for c in df.columns]
            df = self._resolver_colunas(df)

            # Converter coordenadas para float
            for col in ["LATITUDE", "LONGITUDE"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Descartar coordenadas inválidas — conforme documento SSP página 4
            n_antes = len(df)
            df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
            df = df[(df["LATITUDE"] != 0.0) & (df["LONGITUDE"] != 0.0)]
            df = df[
                df["LATITUDE"].between(SP_LAT_MIN, SP_LAT_MAX) &
                df["LONGITUDE"].between(SP_LON_MIN, SP_LON_MAX)
            ]
            self._log("prata_filtro_coordenadas", {
                "arquivo": Path(chave).name,
                "descartados": n_antes - len(df),
                "mantidos": len(df)
            })

            if df.empty:
                continue

            # Normalizar texto
            for col in ["NOME_DEPARTAMENTO", "NOME_MUNICIPIO", "LOGRADOURO",
                        "BAIRRO", "RUBRICA", "DESCR_CONDUTA", "NATUREZA_APURADA"]:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: normalizar_texto(x) if pd.notna(x) else x
                    )

            # Datas
            for col_data in ["DATA_OCORRENCIA_BO", "DATA_REGISTRO"]:
                if col_data in df.columns:
                    df[col_data] = pd.to_datetime(df[col_data], errors="coerce", dayfirst=True)

            # Features temporais
            if "DATA_OCORRENCIA_BO" in df.columns:
                df["ANO"]            = df["DATA_OCORRENCIA_BO"].dt.year
                df["MES"]            = df["DATA_OCORRENCIA_BO"].dt.month
                df["DIA_SEMANA"]     = df["DATA_OCORRENCIA_BO"].dt.dayofweek
                df["DIA_MES"]        = df["DATA_OCORRENCIA_BO"].dt.day

            # Atraso de registro — proxy de subnotificação
            if "DATA_OCORRENCIA_BO" in df.columns and "DATA_REGISTRO" in df.columns:
                df["ATRASO_REGISTRO_DIAS"] = (
                    df["DATA_REGISTRO"] - df["DATA_OCORRENCIA_BO"]
                ).dt.days.clip(lower=0)
            else:
                df["ATRASO_REGISTRO_DIAS"] = 0

            # Hora
            if "HORA_OCORRENCIA_BO" in df.columns:
                df["HORA_INT"] = pd.to_numeric(
                    df["HORA_OCORRENCIA_BO"].astype(str).str[:2], errors="coerce"
                ).fillna(0).astype(int)
                df["TURNO"] = pd.cut(
                    df["HORA_INT"],
                    bins=[-1, 5, 11, 17, 23],
                    labels=["MADRUGADA", "MANHA", "TARDE", "NOITE"]
                ).astype(str)
            else:
                df["HORA_INT"] = 0
                df["TURNO"]    = "DESCONHECIDO"

            # Feriados
            feriados_sp = set()
            for ano in range(2022, 2027):
                feriados_sp.update(holidays.Brazil(state="SP", years=ano).keys())
            df["FERIADO"] = df["DATA_OCORRENCIA_BO"].dt.date.apply(
                lambda d: 1 if d in feriados_sp else 0
            ) if "DATA_OCORRENCIA_BO" in df.columns else 0
            df["FIM_SEMANA"] = df["DIA_SEMANA"].apply(lambda d: 1 if d >= 5 else 0) \
                if "DIA_SEMANA" in df.columns else 0

            # H3
            df["H3_R8"] = df.apply(
                lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], H3_RESOLUCAO)
                if pd.notna(r["LATITUDE"]) and pd.notna(r["LONGITUDE"]) else None,
                axis=1
            )
            df = df.dropna(subset=["H3_R8"])

            # Flag crime veículo
            CRIMES_VEICULO = {
                "ROUBO DE VEICULO", "FURTO DE VEICULO",
                "ROUBO DE MOTOCICLETA", "FURTO DE MOTOCICLETA", "ROUBO DE CARGA"
            }
            df["CRIME_VEICULO"] = df["RUBRICA"].apply(
                lambda x: 1 if isinstance(x, str) and x in CRIMES_VEICULO else 0
            ) if "RUBRICA" in df.columns else 0

            # Peso penal
            df["PESO_PENAL"] = df["RUBRICA"].map(PESO_PENAL_BASE).fillna(1.0) \
                if "RUBRICA" in df.columns else 1.0

            frames.append(df)

        if not frames:
            raise RuntimeError("Nenhum dado válido encontrado nos arquivos parquet.")

        df_prata = pd.concat(frames, ignore_index=True)
        self._log("prata_concluida", {"total_registros": len(df_prata)})

        # Salvar prata no R2
        buf = BytesIO()
        df_prata.to_parquet(buf, index=False)
        buf.seek(0)
        chave_prata = f"{R2_PREFIXO_PRATA}prata_{self.run_id}.parquet"
        self._upload_bytes(chave_prata, buf.getvalue())
        self._log("prata_salva_r2", {"chave": chave_prata})

        return pl.from_pandas(df_prata)

    # ── 3. Construir Ouro ─────────────────────────────────────────────────────

    def _engenharia_features(self, df: pd.DataFrame,
                              historico: pd.DataFrame) -> pd.DataFrame:
        """Agrega por hexágono × mês e constrói todas as features do modelo."""

        agg = df.groupby(["H3_R8", "ANO", "MES"]).agg(
            TOTAL_CRIMES          = ("RUBRICA",          "count"),
            PESO_TOTAL            = ("PESO_PENAL",        "sum"),
            CRIMES_VEICULO        = ("CRIME_VEICULO",     "sum"),
            ATRASO_MEDIO          = ("ATRASO_REGISTRO_DIAS", "mean"),
            PCT_NOITE             = ("TURNO",             lambda x: (x == "NOITE").mean()),
            PCT_MADRUGADA         = ("TURNO",             lambda x: (x == "MADRUGADA").mean()),
            PCT_FIM_SEMANA        = ("FIM_SEMANA",        "mean"),
            PCT_FERIADO           = ("FERIADO",           "mean"),
            HORA_MEDIA            = ("HORA_INT",          "mean"),
            UNIQUE_RUBRICAS       = ("RUBRICA",           "nunique"),
        ).reset_index()

        agg["PCT_CRIMES_VEICULO"] = agg["CRIMES_VEICULO"] / agg["TOTAL_CRIMES"].clip(lower=1)
        agg["RISCO_PONDERADO"]    = agg["PESO_TOTAL"] / agg["TOTAL_CRIMES"].clip(lower=1)

        # Vizinhança espacial H3
        agg["MEDIA_VIZINHOS"] = agg["H3_R8"].apply(
            lambda hex_id: np.mean([
                agg.loc[agg["H3_R8"] == viz, "TOTAL_CRIMES"].values[0]
                if viz in agg["H3_R8"].values else 0
                for viz in h3.grid_disk(hex_id, 1)
                if viz != hex_id
            ])
        )

        # Lags temporais por hexágono
        agg = agg.sort_values(["H3_R8", "ANO", "MES"])
        agg["LAG1"] = agg.groupby("H3_R8")["TOTAL_CRIMES"].shift(1).fillna(0)
        agg["LAG2"] = agg.groupby("H3_R8")["TOTAL_CRIMES"].shift(2).fillna(0)
        agg["MM3"]  = agg.groupby("H3_R8")["TOTAL_CRIMES"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        agg["TENDENCIA"] = agg["TOTAL_CRIMES"] - agg["LAG1"]

        # Memória do modelo — residual learning
        if not historico.empty:
            agg = agg.merge(
                historico[["H3_R8", "ERRO_MEDIO", "TENDENCIA", "EXECUCOES"]].rename(
                    columns={"TENDENCIA": "TENDENCIA_HISTORICA"}
                ),
                on="H3_R8", how="left"
            )
            agg["ERRO_MEDIO"]         = agg["ERRO_MEDIO"].fillna(0)
            agg["TENDENCIA_HISTORICA"] = agg["TENDENCIA_HISTORICA"].fillna(0)
            agg["EXECUCOES"]          = agg["EXECUCOES"].fillna(0)
        else:
            agg["ERRO_MEDIO"]         = 0.0
            agg["TENDENCIA_HISTORICA"] = 0.0
            agg["EXECUCOES"]          = 0.0

        return agg

    def _calcular_risco_por_perfil(self, df_agg: pd.DataFrame) -> pl.DataFrame:
        """Gera tabela long com risco ponderado por perfil de usuário."""
        registros = []
        for _, row in df_agg.iterrows():
            for perfil, multiplicadores in MULTIPLICADOR_PERFIL.items():
                rubrica = str(row.get("RUBRICA_DOMINANTE", ""))
                mult    = multiplicadores.get(rubrica, 1.0)
                registros.append({
                    "H3_R8":           row["H3_R8"],
                    "ANO":             int(row["ANO"]),
                    "MES":             int(row["MES"]),
                    "PERFIL":          perfil,
                    "RISCO_PERFIL":    float(row["RISCO_PONDERADO"]) * mult,
                    "TOTAL_CRIMES":    int(row["TOTAL_CRIMES"]),
                    "PESO_TOTAL":      float(row["PESO_TOTAL"]),
                    "PCT_CRIMES_VEICULO": float(row["PCT_CRIMES_VEICULO"]),
                    "PCT_NOITE":       float(row["PCT_NOITE"]),
                    "PCT_FIM_SEMANA":  float(row["PCT_FIM_SEMANA"]),
                    "LAG1":            float(row["LAG1"]),
                    "LAG2":            float(row["LAG2"]),
                    "MM3":             float(row["MM3"]),
                    "TENDENCIA":       float(row["TENDENCIA"]),
                    "MEDIA_VIZINHOS":  float(row["MEDIA_VIZINHOS"]),
                })
        return pl.DataFrame(registros)

    def construir_ouro(self, df_prata: pl.DataFrame,
                       bq_project: str, bq_dataset: str, bq_cred: str):
        self._log("ouro_inicio")

        df = df_prata.to_pandas()
        if len(df) < MIN_REGISTROS_OURO:
            self._log("ouro_insuficiente", {"registros": len(df)})
            return None

        historico  = self.memoria.carregar_historico()
        ponteiro   = self.memoria.carregar_ponteiro()
        mae_anterior = ponteiro.get("mae") or 999.0

        df_agg = self._engenharia_features(df, historico)

        # Features e alvo
        FEATURES = [
            "LAG1", "LAG2", "MM3", "TENDENCIA",
            "MEDIA_VIZINHOS", "PCT_NOITE", "PCT_MADRUGADA",
            "PCT_FIM_SEMANA", "PCT_FERIADO", "HORA_MEDIA",
            "PCT_CRIMES_VEICULO", "RISCO_PONDERADO",
            "UNIQUE_RUBRICAS", "ATRASO_MEDIO",
            "ERRO_MEDIO", "TENDENCIA_HISTORICA", "EXECUCOES",
            "MES",
        ]
        ALVO = "TOTAL_CRIMES"

        df_modelo = df_agg[FEATURES + [ALVO, "H3_R8", "ANO", "MES"]].dropna()
        if len(df_modelo) < MIN_REGISTROS_OURO:
            self._log("ouro_modelo_insuficiente", {"registros": len(df_modelo)})
            return None

        X = df_modelo[FEATURES].values
        y = df_modelo[ALVO].values

        # TimeSeriesSplit — zero data leakage
        tscv  = TimeSeriesSplit(n_splits=5)
        maes, r2s = [], []

        lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05,
                              num_leaves=63, random_state=42, verbose=-1)
        cat  = CatBoostRegressor(iterations=500, learning_rate=0.05,
                                  depth=6, random_state=42, verbose=0)
        ensemble = VotingRegressor([("lgbm", lgbm), ("cat", cat)])

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            ensemble.fit(X_tr, y_tr)
            pred = ensemble.predict(X_te)
            maes.append(mean_absolute_error(y_te, pred))
            r2s.append(r2_score(y_te, pred))

        mae = float(np.mean(maes))
        r2  = float(np.mean(r2s))
        self._log("ouro_metricas", {"mae": mae, "r2": r2, "mae_anterior": mae_anterior})

        # Treinar modelo final com todos os dados
        ensemble.fit(X, y)
        df_modelo["PREV_CRIMES"] = ensemble.predict(X)
        df_modelo["RESIDUO"]     = df_modelo[ALVO] - df_modelo["PREV_CRIMES"]

        # Atualizar histórico de erros por hexágono
        novo_historico = df_modelo.groupby("H3_R8").agg(
            ERRO_MEDIO = ("RESIDUO", lambda x: float(np.mean(np.abs(x)))),
            TENDENCIA  = ("RESIDUO", lambda x: float(x.iloc[-1] - x.iloc[0]) if len(x) > 1 else 0.0),
            EXECUCOES  = ("RESIDUO", "count"),
        ).reset_index()
        self.memoria.salvar_historico(novo_historico)
        self.memoria.salvar_modelo(ensemble, self.run_id)
        self.memoria.salvar_ponteiro(self.run_id, mae, r2)

        # SHAP para auditoria
        try:
            explainer   = shap.TreeExplainer(lgbm)
            shap_values = explainer.shap_values(X)
            df_shap = pd.DataFrame(shap_values, columns=FEATURES)
            df_shap["H3_R8"]  = df_modelo["H3_R8"].values
            df_shap["RUN_ID"] = self.run_id
            buf_shap = BytesIO()
            df_shap.to_parquet(buf_shap, index=False)
            buf_shap.seek(0)
            self._upload_bytes(
                f"safedriver/datalake/auditoria/shap_{self.run_id}.parquet",
                buf_shap.getvalue()
            )
        except Exception as e:
            self._log("shap_falhou", {"erro": str(e)})

        # Camada ouro — tabela fato por perfil para Looker
        df_perfil = self._calcular_risco_por_perfil(df_modelo)

        # Dimensões para Star Schema
        dim_tempo = pl.DataFrame({
            "SK_TEMPO": list(range(len(df_modelo))),
            "ANO":      df_modelo["ANO"].tolist(),
            "MES":      df_modelo["MES"].tolist(),
        }).unique(["ANO", "MES"])

        dim_local = pl.DataFrame({
            "SK_LOCAL": list(range(len(df_modelo))),
            "H3_R8":    df_modelo["H3_R8"].tolist(),
        }).unique(["H3_R8"])

        # Tabela fato principal
        df_ouro = df_perfil.with_columns([
            pl.lit(self.run_id).alias("RUN_ID"),
            pl.lit(VERSAO_PIPELINE).alias("VERSAO_PIPELINE"),
            pl.lit(VERSAO_FEATURES).alias("VERSAO_FEATURES"),
            pl.lit(hora_brasilia().isoformat()).alias("PROCESSADO_EM"),
            pl.lit(mae).alias("MAE_MODELO"),
            pl.lit(r2).alias("R2_MODELO"),
        ])

        # Salvar ouro no R2
        buf_ouro = BytesIO()
        df_ouro.to_pandas().to_parquet(buf_ouro, index=False)
        buf_ouro.seek(0)
        chave_ouro = f"{R2_PREFIXO_OURO}ouro_{self.run_id}.parquet"
        self._upload_bytes(chave_ouro, buf_ouro.getvalue())
        self._log("ouro_salvo_r2", {"chave": chave_ouro, "registros": len(df_ouro)})

        # BigQuery
        status_bq = "não configurado"
        if bq_project and bq_dataset and bq_cred:
            try:
                enviar_para_bigquery(df_ouro,    "fato_risco_hexagono", bq_project, bq_dataset, bq_cred)
                enviar_para_bigquery(dim_tempo,  "dim_tempo",            bq_project, bq_dataset, bq_cred)
                enviar_para_bigquery(dim_local,  "dim_local",            bq_project, bq_dataset, bq_cred)

                # Log de auditoria no BQ
                df_audit = pl.DataFrame([{
                    "RUN_ID":         self.run_id,
                    "VERSAO":         VERSAO_PIPELINE,
                    "MAE":            mae,
                    "R2":             r2,
                    "MAE_ANTERIOR":   mae_anterior,
                    "MELHORIA":       mae_anterior - mae,
                    "N_REGISTROS":    len(df_ouro),
                    "PROCESSADO_EM":  hora_brasilia().isoformat(),
                }])
                enviar_para_bigquery(df_audit, "auditoria_log", bq_project, bq_dataset, bq_cred,
                                     modo="WRITE_APPEND")
                status_bq = "✅ ok"
            except Exception as e:
                status_bq = f"⚠️ {str(e)[:80]}"
                self._log("bq_erro", {"erro": str(e)})

        melhoria = mae_anterior - mae
        return df_ouro, mae, r2, mae_anterior, melhoria, status_bq

    # ── Orquestrador ──────────────────────────────────────────────────────────

    def processar(self):
        bq_project = sanitizar_secret(os.environ.get("BQ_PROJECT_ID", ""))
        bq_dataset = sanitizar_secret(os.environ.get("BQ_DATASET", ""))
        bq_cred    = sanitizar_secret(os.environ.get("BQ_CREDENTIALS_JSON", ""))

        self.sincronizar_raw()
        df_prata  = self.construir_prata()
        resultado = self.construir_ouro(df_prata, bq_project, bq_dataset, bq_cred)

        tempo = time.time() - self.t_inicio

        if resultado:
            df_ouro, mae, r2, mae_ant, melhoria, status_bq = resultado
            self.discord.notificar_sucesso(
                self.run_id, tempo, len(df_prata), len(df_ouro),
                mae, r2, mae_ant, melhoria, status_bq
            )
        else:
            self.discord.notificar_erro(
                self.run_id, "Pipeline sem ouro", "Dados insuficientes para treino."
            )


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = SafeDriver()
    try:
        app.processar()
    except Exception:
        err = traceback.format_exc()
        print("\n" + "=" * 60, file=sys.stderr)
        print("ERRO FATAL", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(err, file=sys.stderr)
        app.discord.notificar_erro(app.run_id, "Falha Sistêmica SafeDriver", err)
        sys.exit(1)
