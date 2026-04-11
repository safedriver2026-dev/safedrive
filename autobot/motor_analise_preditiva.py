# autobot/motor_analise_preditiva.py
import sys, os, traceback, hashlib, warnings, time, json, unicodedata
from datetime import datetime, timedelta
from io import BytesIO
import urllib3
import polars as pl
import pandas as pd
import numpy as np
import h3
import holidays
import boto3
from botocore.exceptions import ClientError
import joblib
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
import shap
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES GLOBAIS
# ══════════════════════════════════════════════════════════════════════════════

VERSAO_PIPELINE  = "4.5.0"
VERSAO_FEATURES  = "v4"
H3_RESOLUCAO     = 8
MIN_REGISTROS    = 500
ANO_ATUAL        = datetime.utcnow().year
ANOS_DISPONIVEIS = list(range(2022, ANO_ATUAL + 1))

R2_PREFIXO_RAW    = "safedriver/datalake/raw/"
R2_PREFIXO_PRATA  = "safedriver/datalake/prata/"
R2_PREFIXO_OURO   = "safedriver/datalake/ouro/"
R2_PREFIXO_MODELO = "safedriver/datalake/modelos/"
R2_TRACKING       = "safedriver/datalake/raw/tracking_ssp.json"

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

COLUNAS_CRITICAS = [
    "NOME_DEPARTAMENTO", "NOME_MUNICIPIO",
    "LOGRADOURO", "LATITUDE", "LONGITUDE", "DATA_OCORRENCIA_BO"
]


# ══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS
# ══════════════════════════════════════════════════════════════════════════════

def hora_brasilia() -> datetime:
    return datetime.utcnow() - timedelta(hours=3)

def sanitizar_secret(valor: str) -> str:
    if not valor:
        return ""
    return "".join(c for c in valor if c.isprintable()).strip()

def normalizar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return texto
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.upper().strip()

def run_id_curto() -> str:
    return hashlib.md5(str(time.time()).encode()).hexdigest()[:12]

def calcular_escore(row: dict, perfil: str = "MOTORISTA") -> float:
    rubrica  = str(row.get("RUBRICA", "")).upper().strip()
    peso     = PESO_PENAL_BASE.get(rubrica, 1.0)
    mult     = MULTIPLICADOR_PERFIL.get(perfil, {}).get(rubrica, 1.0)
    hora     = row.get("HORA_OCORRENCIA_BO", "")
    try:
        h = int(str(hora).split(":")[0])
        fator_noite = 1.3 if (h >= 22 or h < 6) else 1.0
    except Exception:
        fator_noite = 1.0
    return round(peso * mult * fator_noite, 4)


# ══════════════════════════════════════════════════════════════════════════════
# DISCORD
# ══════════════════════════════════════════════════════════════════════════════

class Discord:
    def __init__(self, webhook_sucesso: str, webhook_erro: str):
        self.ws = webhook_sucesso
        self.we = webhook_erro

    def _enviar(self, webhook: str, payload: dict):
        if not webhook:
            return
        try:
            requests.post(webhook, json=payload, timeout=10)
        except Exception as e:
            print(f"[Discord] Falha ao enviar: {e}")

    def notificar_sucesso(self, run_id, tempo, n_prata, n_ouro,
                          mae, r2, mae_ant, melhoria, status_bq):
        self._enviar(self.ws, {"embeds": [{"title": "✅ SafeDriver Concluído",
            "color": 3066993,
            "fields": [
                {"name": "Run ID",       "value": run_id,                    "inline": True},
                {"name": "Tempo",        "value": f"{tempo:.1f}s",           "inline": True},
                {"name": "Prata",        "value": f"{n_prata:,} registros",  "inline": True},
                {"name": "Ouro",         "value": f"{n_ouro:,} hexágonos",   "inline": True},
                {"name": "MAE",          "value": f"{mae:.4f}",              "inline": True},
                {"name": "R²",           "value": f"{r2:.4f}",               "inline": True},
                {"name": "MAE Anterior", "value": f"{mae_ant:.4f}",          "inline": True},
                {"name": "Melhoria",     "value": f"{melhoria:+.2f}%",       "inline": True},
                {"name": "BigQuery",     "value": status_bq,                 "inline": True},
            ],
            "timestamp": hora_brasilia().isoformat(),
        }]})

    def notificar_erro(self, run_id, titulo, detalhe):
        self._enviar(self.we, {"embeds": [{"title": f"❌ {titulo}",
            "color": 15158332,
            "fields": [
                {"name": "Run ID",  "value": run_id,           "inline": True},
                {"name": "Detalhe", "value": str(detalhe)[:500]},
            ],
            "timestamp": hora_brasilia().isoformat(),
        }]})


# ══════════════════════════════════════════════════════════════════════════════
# TRACKING — controle incremental por ano
# ══════════════════════════════════════════════════════════════════════════════

class Tracking:
    def __init__(self, s3, bucket: str):
        self.s3     = s3
        self.bucket = bucket
        self._estado: dict = {}
        self._carregar()

    def _carregar(self):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=R2_TRACKING)
            self._estado = json.loads(obj["Body"].read().decode("utf-8"))
            print(f"[Tracking] Carregado: {list(self._estado.keys())}")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                print("[Tracking] tracking_ssp.json não encontrado — iniciando do zero.")
                self._estado = {}
            else:
                raise

    def _salvar(self):
        corpo = json.dumps(self._estado, ensure_ascii=False, indent=2).encode("utf-8")
        self.s3.put_object(
            Bucket=self.bucket, Key=R2_TRACKING,
            Body=corpo, ContentType="application/json"
        )
        print(f"[Tracking] Salvo: {list(self._estado.keys())}")

    def tamanho_conhecido(self, ano: int) -> int:
        return self._estado.get(str(ano), {}).get("tamanho_bytes", 0)

    def ultima_data_conhecida(self, ano: int):
        v = self._estado.get(str(ano), {}).get("ultima_data")
        if v:
            try:
                return datetime.strptime(v[:10], "%Y-%m-%d")
            except Exception:
                return None
        return None

    def e_ano_fechado(self, ano: int) -> bool:
        return ano < ANO_ATUAL

    def precisa_processar(self, ano: int, tamanho_atual: int) -> bool:
        conhecido = self.tamanho_conhecido(ano)
        if self.e_ano_fechado(ano):
            resultado = conhecido == 0
            print(f"[Tracking] {ano} fechado — "
                  f"{'processar (nunca visto)' if resultado else 'pular (já processado)'}")
            return resultado
        else:
            resultado = tamanho_atual != conhecido
            print(f"[Tracking] {ano} atual — R2={tamanho_atual:,}B "
                  f"conhecido={conhecido:,}B → "
                  f"{'ATUALIZADO' if resultado else 'sem mudança'}")
            return resultado

    def atualizar(self, ano: int, tamanho_bytes: int, ultima_data, n_registros: int):
        self._estado[str(ano)] = {
            "tamanho_bytes": tamanho_bytes,
            "ultima_data":   str(ultima_data)[:10] if ultima_data else None,
            "registros":     n_registros,
            "atualizado_em": hora_brasilia().isoformat(),
        }
        self._salvar()


# ══════════════════════════════════════════════════════════════════════════════
# MEMÓRIA DO MODELO
# ══════════════════════════════════════════════════════════════════════════════

class MemoriaModelo:
    CHAVE = "safedriver/datalake/modelos/historico_mae.json"

    def __init__(self, s3, bucket: str):
        self.s3     = s3
        self.bucket = bucket
        self._hist: list = []
        self._carregar()

    def _carregar(self):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.CHAVE)
            self._hist = json.loads(obj["Body"].read().decode("utf-8"))
            print(f"[Memória] {len(self._hist)} execuções históricas carregadas.")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                print("[Memória] Sem histórico de erros — primeira execução.")
            else:
                raise

    def mae_anterior(self) -> float:
        if not self._hist:
            return float("inf")
        return self._hist[-1].get("mae", float("inf"))

    def registrar(self, mae: float, r2: float, n: int):
        self._hist.append({
            "ts":  hora_brasilia().isoformat(),
            "mae": mae,
            "r2":  r2,
            "n":   n,
        })
        corpo = json.dumps(self._hist[-50:], ensure_ascii=False, indent=2).encode("utf-8")
        self.s3.put_object(
            Bucket=self.bucket, Key=self.CHAVE,
            Body=corpo, ContentType="application/json"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SafeDriver:

    def __init__(self):
        self.run_id  = run_id_curto()
        self.t_inicio = time.time()
        self.df_raw  = pl.DataFrame()

        endpoint   = sanitizar_secret(os.environ.get("R2_ENDPOINT_URL", ""))
        access_key = sanitizar_secret(os.environ.get("R2_ACCESS_KEY_ID", ""))
        secret_key = sanitizar_secret(os.environ.get("R2_SECRET_ACCESS_KEY", ""))
        self.bucket = sanitizar_secret(os.environ.get("R2_BUCKET_NAME", ""))

        if not self.bucket:
            raise RuntimeError("Secret R2_BUCKET_NAME não definido ou vazio.")

        print(f"[R2] Bucket   : {self.bucket}")
        print(f"[R2] Endpoint : {endpoint}")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )

        self.tracking = Tracking(self.s3, self.bucket)
        self.memoria  = MemoriaModelo(self.s3, self.bucket)
        self.discord  = Discord(
            sanitizar_secret(os.environ.get("DISCORD_SUCESSO", "")),
            sanitizar_secret(os.environ.get("DISCORD_ERRO", ""))
        )

    def _log(self, evento: str, dados: dict):
        ts = hora_brasilia().strftime("%H:%M:%S")
        print(f"  [{self.run_id}] {evento}: {dados} [{ts}]")

    # ── SINCRONIZAR RAW ──────────────────────────────────────────────────────

    def sincronizar_raw(self):
        self._log("sincronizar_raw_inicio", {})
        frames = []

        for ano in ANOS_DISPONIVEIS:
            chave = f"{R2_PREFIXO_RAW}ssp_{ano}.parquet"

            try:
                meta = self.s3.head_object(Bucket=self.bucket, Key=chave)
                tamanho_atual = meta["ContentLength"]
                print(f"[R2] {chave} — {tamanho_atual:,} bytes encontrado")
            except ClientError as e:
                codigo = e.response["Error"]["Code"]
                if codigo in ("404", "NoSuchKey"):
                    print(f"[R2] {chave} não existe — pulando.")
                    continue
                raise

            if not self.tracking.precisa_processar(ano, tamanho_atual):
                continue

            print(f"[R2] Baixando {chave}...")
            obj = self.s3.get_object(Bucket=self.bucket, Key=chave)
            buf = BytesIO(obj["Body"].read())
            df  = pl.read_parquet(buf)
            print(f"[R2] {ano}: {len(df):,} registros | colunas: {df.columns[:6]}")

            # Filtro incremental para ano atual
            ultima = self.tracking.ultima_data_conhecida(ano)
            if not self.tracking.e_ano_fechado(ano) and ultima is not None:
                if "DATA_OCORRENCIA_BO" in df.columns:
                    antes = len(df)
                    df = df.filter(pl.col("DATA_OCORRENCIA_BO") > pl.lit(ultima))
                    print(f"[R2] Incremental {ano}: {antes:,} → {len(df):,} novos")

            if df.is_empty():
                print(f"[R2] {ano} sem registros novos.")
                continue

            frames.append(df)

            ultima_data = None
            if "DATA_OCORRENCIA_BO" in df.columns:
                ultima_data = df["DATA_OCORRENCIA_BO"].max()

            self.tracking.atualizar(ano, tamanho_atual, ultima_data, len(df))

        if not frames:
            self._log("sincronizar_raw_sem_dados_novos", {})
            print("[R2] Nenhum dado novo detectado.")
            return

        self.df_raw = pl.concat(frames, how="diagonal_relaxed")
        self._log("sincronizar_raw_fim", {"total_registros": len(self.df_raw)})
        print(f"[R2] Total consolidado: {len(self.df_raw):,} registros")

    # ── CONSTRUIR PRATA ──────────────────────────────────────────────────────

    def construir_prata(self) -> pl.DataFrame:
        if self.df_raw.is_empty():
            print("[Prata] df_raw vazio — nada a processar.")
            return pl.DataFrame()

        self._log("prata_inicio", {"registros_raw": len(self.df_raw)})
        df = self.df_raw.to_pandas()

        # Renomear sinônimos
        for nome_canonical, aliases in SINONIMOS.items():
            for alias in aliases:
                if alias in df.columns and nome_canonical not in df.columns:
                    df.rename(columns={alias: nome_canonical}, inplace=True)

        # Converter coordenadas
        for col in ["LATITUDE", "LONGITUDE"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remover coordenadas inválidas
        n_antes = len(df)
        df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
        df = df[(df["LATITUDE"] != 0.0) & (df["LONGITUDE"] != 0.0)]
        df = df[
            df["LATITUDE"].between(SP_LAT_MIN, SP_LAT_MAX) &
            df["LONGITUDE"].between(SP_LON_MIN, SP_LON_MAX)
        ]
        print(f"[Prata] Coordenadas: {n_antes:,} → {len(df):,} registros válidos")

        if df.empty:
            print("[Prata] Nenhum registro válido após filtro de coordenadas.")
            return pl.DataFrame()

        # Normalizar texto
        for col in ["NOME_DEPARTAMENTO", "NOME_MUNICIPIO", "LOGRADOURO",
                    "BAIRRO", "RUBRICA", "DESCR_CONDUTA", "NATUREZA_APURADA"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: normalizar_texto(x) if pd.notna(x) else x
                )

        # Datas
        for col in ["DATA_OCORRENCIA_BO", "DATA_REGISTRO"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

        # Features temporais
        if "DATA_OCORRENCIA_BO" in df.columns:
            df["ANO"]         = df["DATA_OCORRENCIA_BO"].dt.year
            df["MES"]         = df["DATA_OCORRENCIA_BO"].dt.month
            df["DIA_SEMANA"]  = df["DATA_OCORRENCIA_BO"].dt.dayofweek
            df["DIA_ANO"]     = df["DATA_OCORRENCIA_BO"].dt.dayofyear
            feriados_sp       = holidays.Brazil(state="SP")
            df["FERIADO"]     = df["DATA_OCORRENCIA_BO"].dt.date.apply(
                lambda d: 1 if d in feriados_sp else 0
            )

        if "HORA_OCORRENCIA_BO" in df.columns:
            df["HORA"] = df["HORA_OCORRENCIA_BO"].apply(
                lambda x: int(str(x).split(":")[0]) if pd.notna(x) else -1
            )
            df["TURNO"] = df["HORA"].apply(
                lambda h: "MADRUGADA" if 0 <= h < 6
                else "MANHA" if 6 <= h < 12
                else "TARDE" if 12 <= h < 18
                else "NOITE" if 18 <= h < 24
                else "DESCONHECIDO"
            )

        # H3
        try:
            df["H3_INDEX"] = df.apply(
                lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], H3_RESOLUCAO)
                if pd.notna(r["LATITUDE"]) and pd.notna(r["LONGITUDE"]) else None,
                axis=1
            )
        except Exception as e:
            print(f"[Prata] H3 falhou: {e}")
            df["H3_INDEX"] = None

        # Escore de risco
        df["ESCORE_RISCO"] = df.apply(
            lambda r: calcular_escore(r.to_dict(), "MOTORISTA"), axis=1
        )

        result = pl.from_pandas(df)
        self._log("prata_fim", {"registros": len(result)})
        print(f"[Prata] {len(result):,} registros processados.")
        return result

    # ── CONSTRUIR OURO ───────────────────────────────────────────────────────

    def construir_ouro(self, df_prata: pl.DataFrame,
                       bq_project: str, bq_dataset: str, bq_cred: str):
        if df_prata.is_empty():
            print("[Ouro] prata vazia — abortando.")
            return None

        self._log("ouro_inicio", {"registros_prata": len(df_prata)})
        df = df_prata.to_pandas()

        if "H3_INDEX" not in df.columns or df["H3_INDEX"].isna().all():
            print("[Ouro] H3_INDEX ausente — abortando.")
            return None

        # Agregação por hexágono
        agg = df.groupby("H3_INDEX").agg(
            LATITUDE_MEDIA=("LATITUDE",    "mean"),
            LONGITUDE_MEDIA=("LONGITUDE",  "mean"),
            QTD_CRIMES=("ESCORE_RISCO",    "count"),
            ESCORE_TOTAL=("ESCORE_RISCO",  "sum"),
            ESCORE_MEDIO=("ESCORE_RISCO",  "mean"),
            PROP_NOITE=("HORA",            lambda x: (x >= 22).mean() + (x < 6).mean()),
            PROP_PATRIMONIO=("RUBRICA",    lambda x: x.str.contains(
                "ROUBO|FURTO", na=False).mean()),
            PROP_MOTORISTA=("RUBRICA",     lambda x: x.str.contains(
                "VEICULO|CARGA|LATROCINIO", na=False).mean()),
            PROP_MOTOCICLETA=("RUBRICA",   lambda x: x.str.contains(
                "MOTOCICLETA", na=False).mean()),
        ).reset_index()

        # Vizinhança H3
        def media_vizinhos(h3_idx, col):
            try:
                vizinhos = list(h3.grid_disk(h3_idx, 1))
                vals = agg.loc[agg["H3_INDEX"].isin(vizinhos), col]
                return vals.mean() if not vals.empty else 0.0
            except Exception:
                return 0.0

        agg["ESCORE_VIZ_1"]   = agg["H3_INDEX"].apply(lambda x: media_vizinhos(x, "ESCORE_MEDIO"))
        agg["QTD_CRIMES_VIZ"] = agg["H3_INDEX"].apply(lambda x: media_vizinhos(x, "QTD_CRIMES"))

        if len(agg) < MIN_REGISTROS:
            print(f"[Ouro] {len(agg)} hexágonos < mínimo {MIN_REGISTROS} — abortando treino.")
            return None

        # Features e target
        features = [
            "LATITUDE_MEDIA", "LONGITUDE_MEDIA",
            "PROP_NOITE", "PROP_PATRIMONIO", "PROP_MOTORISTA", "PROP_MOTOCICLETA",
            "ESCORE_VIZ_1", "QTD_CRIMES_VIZ",
        ]
        target = "ESCORE_TOTAL"
        X = agg[features].fillna(0)
        y = agg[target]

        # Treino com TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        maes, r2s = [], []

        lgbm = LGBMRegressor(n_estimators=400, learning_rate=0.05,
                              max_depth=6, random_state=42, verbose=-1)
        cat  = CatBoostRegressor(iterations=400, learning_rate=0.05,
                                 depth=6, random_seed=42, verbose=0)
        modelo = VotingRegressor([("lgbm", lgbm), ("cat", cat)])

        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            modelo.fit(X_tr, y_tr)
            pred = modelo.predict(X_val)
            maes.append(mean_absolute_error(y_val, pred))
            r2s.append(r2_score(y_val, pred))

        mae = float(np.mean(maes))
        r2  = float(np.mean(r2s))
        print(f"[Ouro] MAE={mae:.4f} R²={r2:.4f}")

        # Predição final
        modelo.fit(X, y)
        agg["ESCORE_PREDITO"] = modelo.predict(X)

        # SHAP
        try:
            explainer   = shap.TreeExplainer(modelo.estimators_[0])
            shap_values = explainer.shap_values(X)
            importancias = dict(zip(features, np.abs(shap_values).mean(axis=0)))
            print(f"[Ouro] SHAP top3: {sorted(importancias.items(), key=lambda x: -x[1])[:3]}")
        except Exception as e:
            print(f"[Ouro] SHAP falhou: {e}")

        mae_ant  = self.memoria.mae_anterior()
        melhoria = ((mae_ant - mae) / mae_ant * 100) if mae_ant != float("inf") else 0.0
        self.memoria.registrar(mae, r2, len(agg))

        # Salvar modelo no R2
        buf_modelo = BytesIO()
        joblib.dump(modelo, buf_modelo)
        buf_modelo.seek(0)
        chave_modelo = f"{R2_PREFIXO_MODELO}modelo_v{VERSAO_PIPELINE}.pkl"
        self.s3.put_object(Bucket=self.bucket, Key=chave_modelo, Body=buf_modelo.read())
        print(f"[Ouro] Modelo salvo em {chave_modelo}")

        # Salvar ouro no R2
        ouro_pl  = pl.from_pandas(agg)
        buf_ouro = BytesIO()
        ouro_pl.write_parquet(buf_ouro)
        buf_ouro.seek(0)
        chave_ouro = f"{R2_PREFIXO_OURO}ouro_atual.parquet"
        self.s3.put_object(Bucket=self.bucket, Key=chave_ouro, Body=buf_ouro.read())
        print(f"[Ouro] Dados salvos em {chave_ouro}")

        # BigQuery
        status_bq = "skipped"
        if bq_project and bq_dataset and bq_cred:
            try:
                cred_info = json.loads(bq_cred)
                creds     = service_account.Credentials.from_service_account_info(
                    cred_info,
                    scopes=["https://www.googleapis.com/auth/bigquery"]
                )
                client    = bigquery.Client(project=bq_project, credentials=creds)
                tabela    = f"{bq_project}.{bq_dataset}.ouro_h3"
                job_cfg   = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
                job       = client.load_table_from_dataframe(agg, tabela, job_config=job_cfg)
                job.result()
                status_bq = f"ok — {len(agg):,} linhas → {tabela}"
                print(f"[BigQuery] {status_bq}")
            except Exception as e:
                status_bq = f"erro: {e}"
                print(f"[BigQuery] {status_bq}")

        self._log("ouro_fim", {"hexagonos": len(agg), "mae": mae, "r2": r2})
        return ouro_pl, mae, r2, mae_ant, melhoria, status_bq

    # ── PROCESSAR ────────────────────────────────────────────────────────────

    def processar(self):
        self._log("pipeline_iniciado", {
            "versao": VERSAO_PIPELINE, "features": VERSAO_FEATURES
        })

        # ← NOMES CORRETOS DOS SECRETS conforme imagem GitHub
        bq_project = sanitizar_secret(os.environ.get("BQ_PROJECT_ID", ""))
        bq_dataset = sanitizar_secret(os.environ.get("BQ_DATASET_ID", ""))
        bq_cred    = sanitizar_secret(os.environ.get("BQ_SERVICE_ACCOUNT_JSON", ""))

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
                self.run_id, "Pipeline sem dados novos",
                "Nenhum arquivo SSP foi atualizado desde a última execução."
            )
        self._log("pipeline_fim", {"tempo_segundos": round(tempo, 1)})


# ══════════════════════════════════════════════════════════════════════════════
# ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════

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
