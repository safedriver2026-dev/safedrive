# autobot/motor_analise_preditiva.py
"""
SafeDriver_Motor_V1.0.0
Pipeline preditivo de segurança urbana — Estado de São Paulo
Estratégia de descoberta R2: listagem → fallback por ano → download forçado.
Escore calculado por perfil: MOTORISTA, MOTOCICLISTA, PEDESTRE, CICLISTA.
"""
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
# IDENTIDADE
# ══════════════════════════════════════════════════════════════════════════════

NOME_SISTEMA     = "SafeDriver_Motor_V1.0.0"
VERSAO_PIPELINE  = "5.3.0"
VERSAO_FEATURES  = "v5"
H3_RESOLUCAO     = 8
MIN_REGISTROS    = 500
ANO_ATUAL        = datetime.utcnow().year
ANOS_DISPONIVEIS = list(range(2022, ANO_ATUAL + 1))
FUSO_BRASILIA    = timedelta(hours=3)
PERFIS           = ["MOTORISTA", "MOTOCICLISTA", "PEDESTRE", "CICLISTA"]

# Candidatos de prefixo raw para tentativa direta — ordem de prioridade
PREFIXOS_CANDIDATOS = [
    "safedriver/datalake/raw/",
    "safedriver/safedriver/datalake/raw/",
    "datalake/raw/",
    "raw/",
    "",
]


# ══════════════════════════════════════════════════════════════════════════════
# PESOS E MULTIPLICADORES
# ══════════════════════════════════════════════════════════════════════════════

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

FATOR_PERIODO = {
    "MADRUGADA": 1.4,
    "MANHA":     1.0,
    "TARDE":     1.0,
    "NOITE":     1.3,
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
    return datetime.utcnow() - FUSO_BRASILIA

def sanitizar_secret(valor: str) -> str:
    if not valor:
        return ""
    return "".join(c for c in valor if c.isprintable()).strip()

def normalizar_texto(texto) -> str:
    if not isinstance(texto, str):
        return str(texto) if texto is not None else ""
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return texto.upper().strip()

def run_id_curto() -> str:
    return hashlib.md5(str(time.time()).encode()).hexdigest()[:12]

def classificar_periodo(hora_str) -> str:
    try:
        h = int(str(hora_str).split(":")[0])
    except Exception:
        return "MANHA"
    if 0 <= h < 6:
        return "MADRUGADA"
    elif 6 <= h < 12:
        return "MANHA"
    elif 12 <= h < 18:
        return "TARDE"
    else:
        return "NOITE"

def fator_periodo(hora_str) -> float:
    return FATOR_PERIODO[classificar_periodo(hora_str)]

def anonimizar_campo(valor: str, salt: str) -> str:
    raw = f"{salt}:{normalizar_texto(str(valor))}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def calcular_escore(rubrica: str, hora_str: str, perfil: str = "MOTORISTA") -> float:
    rubrica = normalizar_texto(rubrica)
    peso    = PESO_PENAL_BASE.get(rubrica, 1.0)
    mult    = MULTIPLICADOR_PERFIL.get(perfil, {}).get(rubrica, 1.0)
    fator   = fator_periodo(hora_str)
    return round(peso * mult * fator, 4)

def calcular_escores_todos_perfis(rubrica: str, hora_str: str) -> dict:
    return {perfil: calcular_escore(rubrica, hora_str, perfil) for perfil in PERFIS}

def renomear_sinonimos(df: pl.DataFrame) -> pl.DataFrame:
    mapa = {}
    colunas_upper = {c.upper(): c for c in df.columns}
    for nome_oficial, sinonimos in SINONIMOS.items():
        if nome_oficial.upper() not in colunas_upper:
            for sin in sinonimos:
                if sin.upper() in colunas_upper:
                    mapa[colunas_upper[sin.upper()]] = nome_oficial
                    break
    if mapa:
        print(f"[Prata] Renomeando colunas: {mapa}")
        df = df.rename(mapa)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DESCOBERTA DE PREFIXO R2 — 3 ESTRATÉGIAS EM CASCATA
# ══════════════════════════════════════════════════════════════════════════════

def _chave_existe(s3, bucket: str, chave: str) -> bool:
    """Verifica se um objeto existe no R2 via HEAD."""
    try:
        s3.head_object(Bucket=bucket, Key=chave)
        return True
    except ClientError:
        return False

def descobrir_prefixo_raw(s3_client, bucket: str) -> str | None:
    """
    Estratégia 1: ListObjectsV2 completo.
    Estratégia 2: Tentativa direta com prefixos candidatos + ano 2022.
    Estratégia 3: Retorna None — sincronizar_raw fará download forçado por ano.
    """

    # ── Estratégia 1: listagem ────────────────────────────────────────────────
    print(f"[R2] Estratégia 1 — listando bucket '{bucket}'...")
    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            for obj in page.get("Contents", []):
                chave = obj["Key"]
                nome  = chave.split("/")[-1]
                if nome.startswith("ssp_") and nome.endswith(".parquet"):
                    prefixo = chave[: chave.rfind("/") + 1]
                    print(f"[R2] Prefixo encontrado via listagem: '{prefixo}'")
                    return prefixo
    except Exception as e:
        print(f"[R2] Listagem falhou ({type(e).__name__}): {e}")

    # ── Estratégia 2: tentativa direta por candidatos ─────────────────────────
    print("[R2] Estratégia 2 — tentando prefixos candidatos com ssp_2022.parquet...")
    for candidato in PREFIXOS_CANDIDATOS:
        chave_teste = f"{candidato}ssp_2022.parquet"
        print(f"[R2]   HEAD {chave_teste} ...")
        if _chave_existe(s3_client, bucket, chave_teste):
            print(f"[R2] Prefixo encontrado via HEAD: '{candidato}'")
            return candidato

    # ── Estratégia 3: nenhum prefixo encontrado ───────────────────────────────
    print("[R2] Estratégia 3 — prefixo não descoberto. Download forçado será feito por ano.")
    return None

def derivar_prefixos(prefixo_raw: str) -> dict:
    base = prefixo_raw.rsplit("raw/", 1)[0] if "raw/" in prefixo_raw else prefixo_raw
    return {
        "raw":    prefixo_raw,
        "prata":  f"{base}prata/",
        "ouro":   f"{base}ouro/",
        "modelo": f"{base}modelos/",
        "track":  f"{prefixo_raw}tracking_ssp.json",
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRACKING SSP
# ══════════════════════════════════════════════════════════════════════════════

class TrackingSSP:
    def __init__(self, s3, bucket: str, chave_track: str):
        self.s3     = s3
        self.bucket = bucket
        self.chave  = chave_track
        self.dados  = self._carregar()

    def _carregar(self) -> dict:
        try:
            obj   = self.s3.get_object(Bucket=self.bucket, Key=self.chave)
            dados = json.loads(obj["Body"].read().decode("utf-8"))
            print(f"[Tracking] Carregado: {list(dados.keys())}")
            return dados
        except Exception:
            print("[Tracking] tracking_ssp.json não encontrado — iniciando do zero.")
            return {}

    def _salvar(self):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.chave,
            Body=json.dumps(self.dados, ensure_ascii=False, default=str).encode("utf-8"),
        )

    def precisa_processar(self, ano: int, tamanho_atual: int) -> bool:
        entrada = self.dados.get(str(ano), {})
        return entrada.get("tamanho_bytes", 0) != tamanho_atual

    def ultima_data_conhecida(self, ano: int):
        entrada = self.dados.get(str(ano), {})
        val     = entrada.get("ultima_data")
        if val:
            try:
                return pd.Timestamp(val)
            except Exception:
                return None
        return None

    def e_ano_fechado(self, ano: int) -> bool:
        return ano < ANO_ATUAL

    def atualizar(self, ano: int, tamanho: int, ultima_data, n_registros: int):
        self.dados[str(ano)] = {
            "tamanho_bytes": tamanho,
            "ultima_data":   str(ultima_data) if ultima_data else None,
            "n_registros":   n_registros,
            "atualizado_em": hora_brasilia().isoformat(),
        }
        self._salvar()


# ══════════════════════════════════════════════════════════════════════════════
# MEMÓRIA DE PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

class MemoriaPerformance:
    def __init__(self, s3, bucket: str, prefixo_ouro: str):
        self.s3     = s3
        self.bucket = bucket
        self.chave  = f"{prefixo_ouro}memoria_performance.json"
        self.dados  = self._carregar()

    def _carregar(self) -> dict:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.chave)
            return json.loads(obj["Body"].read().decode("utf-8"))
        except Exception:
            print("[Memória] Sem histórico — primeira execução.")
            return {}

    def mae_anterior(self) -> float:
        return float(self.dados.get("mae", float("inf")))

    def registrar(self, mae: float, r2: float, n_hex: int):
        self.dados = {
            "mae":          mae,
            "r2":           r2,
            "n_hexagonos":  n_hex,
            "atualizado_em": hora_brasilia().isoformat(),
        }
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.chave,
            Body=json.dumps(self.dados).encode("utf-8"),
        )


# ══════════════════════════════════════════════════════════════════════════════
# DISCORD
# ══════════════════════════════════════════════════════════════════════════════

class Discord:
    def __init__(self):
        self.url_sucesso = sanitizar_secret(os.environ.get("DISCORD_SUCESSO", ""))
        self.url_erro    = sanitizar_secret(os.environ.get("DISCORD_ERRO", ""))

    def _post(self, url: str, payload: dict):
        if not url:
            return
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"[Discord] Falha ao enviar: {e}")

    def relatorio_executivo(self, run_id, tempo, n_hex, mae, r2,
                            melhoria, top_municipio, top_shap):
        sinal = "📈" if melhoria > 0 else "📉"
        self._post(self.url_sucesso, {"embeds": [{"title": f"🚗 SafeDriver — Relatório Executivo",
            "color": 3066993,
            "fields": [
                {"name": "Run ID",             "value": run_id,                        "inline": True},
                {"name": "Tempo",              "value": f"{tempo:.1f}s",               "inline": True},
                {"name": "Hexágonos",          "value": f"{n_hex:,}",                  "inline": True},
                {"name": "MAE",                "value": f"{mae:.4f}",                  "inline": True},
                {"name": "R²",                 "value": f"{r2:.4f}",                   "inline": True},
                {"name": f"Melhoria {sinal}",  "value": f"{melhoria:+.1f}%",           "inline": True},
                {"name": "Município Crítico",  "value": top_municipio,                 "inline": True},
                {"name": "Feature SHAP #1",    "value": top_shap,                      "inline": True},
            ],
            "footer": {"text": f"{NOME_SISTEMA} v{VERSAO_PIPELINE}"},
        }]})

    def relatorio_operacional(self, run_id, n_raw, n_prata, n_ouro,
                               mae, r2, mae_ant, anos, status_bq, shap_top3, prefixo_raw):
        shap_str = " | ".join([f"{f}={v:.4f}" for f, v in shap_top3]) if shap_top3 else "N/A"
        self._post(self.url_sucesso, {"embeds": [{"title": "🔧 SafeDriver — Operacional",
            "color": 15105570,
            "fields": [
                {"name": "Run ID",       "value": run_id,             "inline": True},
                {"name": "Raw",          "value": f"{n_raw:,}",       "inline": True},
                {"name": "Prata",        "value": f"{n_prata:,}",     "inline": True},
                {"name": "Ouro (hex)",   "value": f"{n_ouro:,}",      "inline": True},
                {"name": "MAE atual",    "value": f"{mae:.4f}",       "inline": True},
                {"name": "MAE anterior", "value": f"{mae_ant:.4f}",   "inline": True},
                {"name": "R²",           "value": f"{r2:.4f}",        "inline": True},
                {"name": "Anos",         "value": str(anos),          "inline": True},
                {"name": "BigQuery",     "value": status_bq,          "inline": False},
                {"name": "SHAP top3",    "value": shap_str,           "inline": False},
                {"name": "Prefixo R2",   "value": f"`{prefixo_raw}`", "inline": False},
            ],
            "footer": {"text": f"{NOME_SISTEMA} v{VERSAO_PIPELINE}"},
        }]})

    def sem_novidades(self, run_id: str, tempo: float):
        self._post(self.url_sucesso, {"embeds": [{"title": "ℹ️ SafeDriver — Sem dados novos",
            "color": 9807270,
            "fields": [
                {"name": "Run ID", "value": run_id,          "inline": True},
                {"name": "Tempo",  "value": f"{tempo:.1f}s", "inline": True},
            ],
        }]})

    def alerta_erro(self, run_id: str, titulo: str, detalhe: str):
        self._post(self.url_erro, {"embeds": [{"title": f"🚨 {titulo}",
            "color": 15158332,
            "fields": [
                {"name": "Run ID",  "value": run_id,            "inline": True},
                {"name": "Detalhe", "value": detalhe[:1000],    "inline": False},
            ],
        }]})

    def alerta_bucket_vazio(self, run_id: str, bucket: str):
        self._post(self.url_erro, {"embeds": [{"title": "🚨 SafeDriver — Bucket sem dados SSP",
            "color": 15158332,
            "description": (
                f"Nenhum arquivo `ssp_*.parquet` encontrado no bucket `{bucket}` "
                f"após todas as estratégias de descoberta.\n\n"
                f"Verifique se os arquivos foram carregados corretamente no R2."
            ),
            "fields": [{"name": "Run ID", "value": run_id, "inline": True}],
        }]})


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SafeDriver:
    def __init__(self):
        self.run_id  = run_id_curto()
        self.t_inicio = time.time()
        self.discord  = Discord()

        print(f"[{NOME_SISTEMA}] Iniciando run {self.run_id}")

        self.bucket   = sanitizar_secret(os.environ.get("R2_BUCKET_NAME", ""))
        endpoint      = sanitizar_secret(os.environ.get("R2_ENDPOINT_URL", ""))
        access_key    = sanitizar_secret(os.environ.get("R2_ACCESS_KEY_ID", ""))
        secret_key    = sanitizar_secret(os.environ.get("R2_SECRET_ACCESS_KEY", ""))
        self.lgpd_salt = sanitizar_secret(os.environ.get("LGPD_SALT", "safedriver_default_salt"))

        print(f"[R2] Bucket   : {self.bucket}")
        print(f"[R2] Endpoint : {endpoint[:40]}...")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )

        # Descoberta automática de prefixo em cascata
        prefixo_raw = descobrir_prefixo_raw(self.s3, self.bucket)

        if prefixo_raw is None:
            # Nenhum arquivo existe ainda — usaremos prefixo padrão
            # e sincronizar_raw fará o download forçado de todos os anos
            print("[R2] Usando prefixo padrão 'safedriver/datalake/raw/' para primeira carga.")
            prefixo_raw = "safedriver/datalake/raw/"

        self.prefixos      = derivar_prefixos(prefixo_raw)
        self.tracking      = TrackingSSP(self.s3, self.bucket, self.prefixos["track"])
        self.memoria       = MemoriaPerformance(self.s3, self.bucket, self.prefixos["ouro"])
        self.df_raw        = pl.DataFrame()
        self.anos_processados = []

    def _log(self, evento: str, dados: dict):
        ts  = hora_brasilia().strftime("%H:%M:%S")
        msg = json.dumps(dados, ensure_ascii=False, default=str)
        print(f"  [{self.run_id}] {evento}: ***{msg}*** [{ts}]")

    # ── SINCRONIZAR RAW ──────────────────────────────────────────────────────

    def sincronizar_raw(self):
        self._log("sincronizar_raw_inicio", {})
        frames = []

        for ano in ANOS_DISPONIVEIS:
            chave = f"{self.prefixos['raw']}ssp_{ano}.parquet"

            # Verificar tamanho atual
            tamanho_atual = 0
            try:
                meta          = self.s3.head_object(Bucket=self.bucket, Key=chave)
                tamanho_atual = meta["ContentLength"]
            except ClientError as e:
                codigo = e.response["Error"]["Code"]
                if codigo in ("404", "NoSuchKey"):
                    print(f"[R2] {chave} não existe — pulando.")
                    continue
                raise

            if not self.tracking.precisa_processar(ano, tamanho_atual):
                print(f"[R2] {ano} sem alterações — usando cache.")
                # Carregar mesmo assim para ter o df_raw completo
                try:
                    obj = self.s3.get_object(Bucket=self.bucket, Key=chave)
                    buf = BytesIO(obj["Body"].read())
                    frames.append(pl.read_parquet(buf))
                except Exception as e:
                    print(f"[R2] Erro ao carregar cache {ano}: {e}")
                continue

            print(f"[R2] Baixando {chave}...")
            obj = self.s3.get_object(Bucket=self.bucket, Key=chave)
            buf = BytesIO(obj["Body"].read())
            df  = pl.read_parquet(buf)
            print(f"[R2] {ano} — {len(df):,} registros | colunas: {df.columns[:6]}")

            ultima_data = self.tracking.ultima_data_conhecida(ano)
            if not self.tracking.e_ano_fechado(ano) and ultima_data is not None:
                if "DATA_OCORRENCIA_BO" in df.columns:
                    antes = len(df)
                    df    = df.filter(pl.col("DATA_OCORRENCIA_BO") > pl.lit(ultima_data))
                    print(f"[R2] Incremental {ano}: {antes:,} → {len(df):,} registros novos")

            if df.is_empty():
                print(f"[R2] {ano} sem registros novos — pulando.")
                continue

            ultima = df["DATA_OCORRENCIA_BO"].max() if "DATA_OCORRENCIA_BO" in df.columns else None
            self.tracking.atualizar(ano, tamanho_atual, ultima, len(df))
            self.anos_processados.append(ano)
            frames.append(df)

        if frames:
            self.df_raw = pl.concat(frames, how="diagonal")
            print(f"[R2] Total consolidado: {len(self.df_raw):,} registros | {len(ANOS_DISPONIVEIS)} anos tentados")
            self._log("sincronizar_raw_fim", {"total_registros": len(self.df_raw)})
        else:
            print("[R2] Nenhum dado encontrado em nenhum ano.")
            self.discord.alerta_bucket_vazio(self.run_id, self.bucket)
            self._log("sincronizar_raw_sem_dados", {})

    # ── CONSTRUIR PRATA ──────────────────────────────────────────────────────

    def construir_prata(self) -> pd.DataFrame:
        if self.df_raw.is_empty():
            print("[Prata] df_raw vazio — nada a processar.")
            return pd.DataFrame()

        print(f"[Prata] Iniciando — {len(self.df_raw):,} registros...")
        df = renomear_sinonimos(self.df_raw)

        # Normalizar colunas de texto
        for col in ["RUBRICA", "NOME_MUNICIPIO", "LOGRADOURO", "NOME_DEPARTAMENTO"]:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).map_elements(normalizar_texto, return_dtype=pl.Utf8).alias(col)
                )

        # Coordenadas
        for col in ["LATITUDE", "LONGITUDE"]:
            if col in df.columns:
                df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

        df = df.with_columns([
            pl.col("LATITUDE").alias("LATITUDE_F"),
            pl.col("LONGITUDE").alias("LONGITUDE_F"),
        ])

        # Filtro geográfico SP
        if "LATITUDE_F" in df.columns and "LONGITUDE_F" in df.columns:
            antes = len(df)
            df = df.filter(
                (pl.col("LATITUDE_F")  >= SP_LAT_MIN) & (pl.col("LATITUDE_F")  <= SP_LAT_MAX) &
                (pl.col("LONGITUDE_F") >= SP_LON_MIN) & (pl.col("LONGITUDE_F") <= SP_LON_MAX) &
                pl.col("LATITUDE_F").is_not_null() & pl.col("LONGITUDE_F").is_not_null()
            )
            print(f"[Prata] Filtro geográfico: {antes:,} → {len(df):,}")

        if df.is_empty():
            print("[Prata] DataFrame vazio após filtros.")
            return pd.DataFrame()

        # Data e hora
        if "DATA_OCORRENCIA_BO" in df.columns:
            df = df.with_columns(
                pl.col("DATA_OCORRENCIA_BO").cast(pl.Utf8).str.slice(0, 10).alias("DATA_STR")
            )
        else:
            df = df.with_columns(pl.lit("2024-01-01").alias("DATA_STR"))

        hora_col = "HORA_OCORRENCIA_BO" if "HORA_OCORRENCIA_BO" in df.columns else None

        # H3 Index
        def safe_h3(lat, lon):
            try:
                return h3.latlng_to_cell(float(lat), float(lon), H3_RESOLUCAO)
            except Exception:
                return None

        print("[Prata] Calculando H3...")
        df_pd = df.to_pandas()
        df_pd["H3_INDEX"] = df_pd.apply(
            lambda r: safe_h3(r.get("LATITUDE_F"), r.get("LONGITUDE_F")), axis=1
        )
        df_pd = df_pd[df_pd["H3_INDEX"].notna()].copy()

        # LGPD — anonimizar campos sensíveis
        for campo in ["LOGRADOURO", "NUMERO_LOGRADOURO", "BAIRRO"]:
            if campo in df_pd.columns:
                df_pd[f"ID_AUDITORIA_{campo}"] = df_pd[campo].apply(
                    lambda v: anonimizar_campo(str(v), self.lgpd_salt)
                )
                df_pd.drop(columns=[campo], inplace=True)

        # Período do dia
        if hora_col and hora_col in df_pd.columns:
            df_pd["PERIODO_DIA"] = df_pd[hora_col].apply(classificar_periodo)
        else:
            df_pd["PERIODO_DIA"] = "MANHA"

        # Escore por perfil — coluna individual para cada perfil
        rubrica_col = "RUBRICA" if "RUBRICA" in df_pd.columns else None
        hora_serie  = df_pd[hora_col] if hora_col and hora_col in df_pd.columns else pd.Series(["08:00"] * len(df_pd))

        if rubrica_col:
            for perfil in PERFIS:
                df_pd[f"ESCORE_{perfil}"] = df_pd.apply(
                    lambda r: calcular_escore(str(r[rubrica_col]), str(r[hora_col]) if hora_col and hora_col in r else "08:00", perfil),
                    axis=1
                )
            # Escore base = MOTORISTA para compatibilidade
            df_pd["ESCORE"] = df_pd["ESCORE_MOTORISTA"]
        else:
            for perfil in PERFIS:
                df_pd[f"ESCORE_{perfil}"] = 1.0
            df_pd["ESCORE"] = 1.0

        # Flags auxiliares
        df_pd["IS_NOITE_MADRUGADA"] = df_pd["PERIODO_DIA"].isin(["NOITE", "MADRUGADA"]).astype(int)

        crimes_patrimonio = {"ROUBO DE VEICULO","FURTO DE VEICULO","ROUBO DE MOTOCICLETA",
                             "FURTO DE MOTOCICLETA","ROUBO","FURTO","ROUBO DE CARGA","LATROCINIO"}
        crimes_pessoa     = {"HOMICIDIO DOLOSO","LATROCINIO","LESAO CORPORAL DOLOSA",
                             "ESTUPRO","ATROPELAMENTO"}

        if rubrica_col:
            df_pd["IS_PATRIMONIO"]       = df_pd[rubrica_col].isin(crimes_patrimonio).astype(int)
            df_pd["IS_VIOLENCIA_PESSOA"] = df_pd[rubrica_col].isin(crimes_pessoa).astype(int)
        else:
            df_pd["IS_PATRIMONIO"]       = 0
            df_pd["IS_VIOLENCIA_PESSOA"] = 0

        # Feriados SP
        feriados_sp = set()
        for ano in ANOS_DISPONIVEIS:
            feriados_sp.update(str(d) for d in holidays.Brazil(state="SP", years=ano).keys())
        df_pd["IS_FERIADO"] = df_pd["DATA_STR"].isin(feriados_sp).astype(int)

        # ANO_MES
        df_pd["ANO_MES"] = df_pd["DATA_STR"].str[:7]

        # Município dominante
        if "NOME_MUNICIPIO" in df_pd.columns:
            df_pd["NOME_MUNICIPIO"] = df_pd["NOME_MUNICIPIO"].fillna("DESCONHECIDO")

        # Salvar prata
        prata_pl  = pl.from_pandas(df_pd)
        buf_prata = BytesIO()
        prata_pl.write_parquet(buf_prata)
        buf_prata.seek(0)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefixos['prata']}prata_atual.parquet",
            Body=buf_prata.read(),
        )

        print(f"[Prata] Concluída — {len(df_pd):,} registros | colunas escore: {[f'ESCORE_{p}' for p in PERFIS]}")
        self._log("prata_fim", {"registros": len(df_pd)})
        return df_pd

    # ── CONSTRUIR OURO ───────────────────────────────────────────────────────

    def construir_ouro(self, df_prata: pd.DataFrame,
                       bq_project: str, bq_dataset: str, bq_cred: str):
        if df_prata.empty:
            print("[Ouro] prata vazia — abortando.")
            return None

        print(f"[Ouro] Iniciando agregação por hexágono H3...")

        # Agregação base + por perfil
        agg_dict = {
            "QTD_CRIMES":             ("ESCORE",             "count"),
            "ESCORE_TOTAL":           ("ESCORE",             "sum"),
            "ESCORE_MEDIO":           ("ESCORE",             "mean"),
            "ESCORE_GRAVIDADE_MAX":   ("ESCORE",             "max"),
            "LATITUDE_MEDIA":         ("LATITUDE_F",         "mean"),
            "LONGITUDE_MEDIA":        ("LONGITUDE_F",        "mean"),
            "PROP_NOITE_MADRUGADA":   ("IS_NOITE_MADRUGADA", "mean"),
            "PROP_PATRIMONIO":        ("IS_PATRIMONIO",      "mean"),
            "PROP_VIOLENCIA_PESSOA":  ("IS_VIOLENCIA_PESSOA","mean"),
            "PROP_FERIADO":           ("IS_FERIADO",         "mean"),
        }

        # Escore médio por perfil como coluna individual no ouro
        for perfil in PERFIS:
            col = f"ESCORE_{perfil}"
            if col in df_prata.columns:
                agg_dict[f"ESCORE_MEDIO_{perfil}"] = (col, "mean")
                agg_dict[f"ESCORE_TOTAL_{perfil}"] = (col, "sum")

        if "NOME_MUNICIPIO" in df_prata.columns:
            agg_dict["MUNICIPIO_DOMINANTE"] = ("NOME_MUNICIPIO", lambda x: x.mode()[0] if not x.empty else "DESCONHECIDO")

        agg = df_prata.groupby("H3_INDEX").agg(**agg_dict).reset_index()

        # Lagged inference — shift(2) por hexágono
        agg = agg.sort_values("H3_INDEX").reset_index(drop=True)
        agg["ESCORE_LAG2"] = agg["ESCORE_TOTAL"].shift(2)
        agg["QTD_LAG2"]    = agg["QTD_CRIMES"].shift(2)
        agg["ESCORE_LAG2"] = agg["ESCORE_LAG2"].fillna(agg["ESCORE_TOTAL"])
        agg["QTD_LAG2"]    = agg["QTD_LAG2"].fillna(agg["QTD_CRIMES"])

        # Spatial Smoothing ring-1 e ring-2
        idx_map = dict(zip(agg["H3_INDEX"], agg.index))

        def media_vizinhos(h3_idx: str, col: str, ring: int = 1) -> float:
            try:
                vizinhos = [v for v in h3.grid_disk(h3_idx, ring) if v in idx_map]
                if not vizinhos:
                    return 0.0
                return float(agg.loc[[idx_map[v] for v in vizinhos], col].mean())
            except Exception:
                return 0.0

        print("[Ouro] Calculando Spatial Smoothing ring-1 e ring-2...")
        agg["ESCORE_VIZ_1"]   = agg["H3_INDEX"].apply(lambda x: media_vizinhos(x, "ESCORE_MEDIO", 1))
        agg["ESCORE_VIZ_2"]   = agg["H3_INDEX"].apply(lambda x: media_vizinhos(x, "ESCORE_MEDIO", 2))
        agg["QTD_CRIMES_VIZ"] = agg["H3_INDEX"].apply(lambda x: media_vizinhos(x, "QTD_CRIMES",   1))

        features = [
            "LATITUDE_MEDIA", "LONGITUDE_MEDIA",
            "PROP_NOITE_MADRUGADA", "PROP_PATRIMONIO", "PROP_VIOLENCIA_PESSOA",
            "ESCORE_GRAVIDADE_MAX", "ESCORE_VIZ_1", "ESCORE_VIZ_2",
            "QTD_CRIMES_VIZ", "ESCORE_LAG2", "QTD_LAG2",
        ]
        target = "ESCORE_TOTAL"
        X = agg[features].fillna(0)
        y = agg[target]

        if len(agg) < MIN_REGISTROS:
            print(f"[Ouro] {len(agg)} hexágonos insuficientes (mínimo {MIN_REGISTROS}).")
            return None

        print("[Ouro] Treinando Ensemble LightGBM + CatBoost...")
        tscv = TimeSeriesSplit(n_splits=5)
        maes, r2s = [], []

        lgbm = LGBMRegressor(
            n_estimators=500, learning_rate=0.04, max_depth=6,
            num_leaves=63, min_child_samples=20, random_state=42, verbose=-1
        )
        cat = CatBoostRegressor(
            iterations=500, learning_rate=0.04, depth=6,
            random_seed=42, verbose=0, loss_function="RMSE"
        )
        modelo = VotingRegressor([("lgbm", lgbm), ("cat", cat)])

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            modelo.fit(X_tr, y_tr)
            pred     = modelo.predict(X_val)
            fold_mae = mean_absolute_error(y_val, pred)
            fold_r2  = r2_score(y_val, pred)
            maes.append(fold_mae)
            r2s.append(fold_r2)
            print(f"[Ouro] Fold {fold+1}/5 — MAE={fold_mae:.4f} R²={fold_r2:.4f}")

        mae = float(np.mean(maes))
        r2  = float(np.mean(r2s))
        print(f"[Ouro] Final — MAE={mae:.4f}  R²={r2:.4f}")

        modelo.fit(X, y)
        agg["ESCORE_PREDITO"] = modelo.predict(X)

        # SHAP
        shap_top3, shap_col_names, top_shap_feature = [], [], "N/A"
        try:
            print("[Ouro] Calculando SHAP...")
            explainer        = shap.TreeExplainer(modelo.estimators_[0])
            shap_values      = explainer.shap_values(X)
            importancias     = list(zip(features, np.abs(shap_values).mean(axis=0)))
            importancias.sort(key=lambda x: -x[1])
            shap_top3        = importancias[:3]
            top_shap_feature = shap_top3[0][0]
            print(f"[Ouro] SHAP top3: {shap_top3}")
            for i, feat in enumerate(features):
                col_shap = f"SHAP_{feat}"
                agg[col_shap] = shap_values[:, i]
                shap_col_names.append(col_shap)
        except Exception as e:
            print(f"[Ouro] SHAP não crítico — continuando: {e}")

        # Município mais crítico
        top_municipio = "N/A"
        if "MUNICIPIO_DOMINANTE" in agg.columns:
            try:
                top_municipio = (
                    agg.groupby("MUNICIPIO_DOMINANTE")["ESCORE_TOTAL"]
                    .sum().sort_values(ascending=False).index[0]
                )
            except Exception:
                pass

        mae_ant  = self.memoria.mae_anterior()
        melhoria = ((mae_ant - mae) / mae_ant * 100) if mae_ant != float("inf") else 0.0
        self.memoria.registrar(mae, r2, len(agg))

        # Salvar modelo
        buf_modelo = BytesIO()
        joblib.dump(modelo, buf_modelo)
        buf_modelo.seek(0)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefixos['modelo']}modelo_{NOME_SISTEMA}.pkl",
            Body=buf_modelo.read(),
        )

        # Salvar ouro
        ouro_pl  = pl.from_pandas(agg)
        buf_ouro = BytesIO()
        ouro_pl.write_parquet(buf_ouro)
        buf_ouro.seek(0)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefixos['ouro']}ouro_atual.parquet",
            Body=buf_ouro.read(),
        )
        print(f"[Ouro] {len(agg):,} hexágonos | perfis: {PERFIS} | SHAP: {len(shap_col_names)} colunas")

        # BigQuery
        status_bq = "skipped"
        if bq_project and bq_dataset and bq_cred:
            try:
                cred_info = json.loads(bq_cred)
                creds     = service_account.Credentials.from_service_account_info(
                    cred_info, scopes=["https://www.googleapis.com/auth/bigquery"]
                )
                client  = bigquery.Client(project=bq_project, credentials=creds)
                tabela  = f"{bq_project}.{bq_dataset}.ouro_h3"
                job_cfg = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
                job     = client.load_table_from_dataframe(agg, tabela, job_config=job_cfg)
                job.result()
                status_bq = f"ok — {len(agg):,} linhas → {tabela}"
                print(f"[BigQuery] {status_bq}")
            except Exception as e:
                status_bq = f"erro: {str(e)[:300]}"
                print(f"[BigQuery] {status_bq}")

        self._log("ouro_fim", {
            "hexagonos": len(agg), "mae": round(mae, 4),
            "r2": round(r2, 4), "shap_top1": top_shap_feature,
            "top_municipio": top_municipio, "perfis": PERFIS,
        })

        return (ouro_pl, mae, r2, mae_ant, melhoria, status_bq,
                len(self.df_raw), len(df_prata), top_municipio,
                top_shap_feature, shap_top3)

    # ── PROCESSAR ────────────────────────────────────────────────────────────

    def processar(self):
        self._log("pipeline_iniciado", {
            "sistema": NOME_SISTEMA, "versao": VERSAO_PIPELINE, "features": VERSAO_FEATURES
        })

        bq_project = sanitizar_secret(os.environ.get("BQ_PROJECT_ID", ""))
        bq_dataset = sanitizar_secret(os.environ.get("BQ_DATASET_ID", ""))
        bq_cred    = sanitizar_secret(os.environ.get("BQ_SERVICE_ACCOUNT_JSON", ""))

        self.sincronizar_raw()
        df_prata  = self.construir_prata()
        resultado = self.construir_ouro(df_prata, bq_project, bq_dataset, bq_cred)

        tempo = time.time() - self.t_inicio

        if resultado:
            (df_ouro, mae, r2, mae_ant, melhoria, status_bq,
             n_raw, n_prata, top_municipio, top_shap_feature, shap_top3) = resultado

            self.discord.relatorio_executivo(
                self.run_id, tempo, len(df_ouro), mae, r2,
                melhoria, top_municipio, top_shap_feature
            )
            self.discord.relatorio_operacional(
                self.run_id, n_raw, n_prata, len(df_ouro),
                mae, r2, mae_ant, self.anos_processados,
                status_bq, shap_top3, self.prefixos["raw"]
            )
        else:
            self.discord.sem_novidades(self.run_id, tempo)

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
        try:
            app.discord.alerta_erro(app.run_id, "Falha Sistêmica", err)
        except Exception:
            pass
        sys.exit(1)
