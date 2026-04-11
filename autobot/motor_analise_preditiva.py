# autobot/motor_analise_preditiva.py
"""
SafeDriver_Motor_V1.0.0
Pipeline preditivo de segurança urbana — Estado de São Paulo
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

NOME_SISTEMA    = "SafeDriver_Motor_V1.0.0"
VERSAO_PIPELINE = "5.0.0"
VERSAO_FEATURES = "v5"
H3_RESOLUCAO    = 8
MIN_REGISTROS   = 500
ANO_ATUAL       = datetime.utcnow().year
ANOS_DISPONIVEIS = list(range(2022, ANO_ATUAL + 1))
FUSO_BRASILIA   = timedelta(hours=3)

# Estrutura R2 confirmada pelas imagens:
# bucket : safedriver-bucket  (secret R2_BUCKET_NAME)
# caminho: safedriver/safedriver/datalake/raw/ssp_XXXX.parquet
R2_PREFIXO_RAW    = "safedriver/safedriver/datalake/raw/"
R2_PREFIXO_PRATA  = "safedriver/safedriver/datalake/prata/"
R2_PREFIXO_OURO   = "safedriver/safedriver/datalake/ouro/"
R2_PREFIXO_MODELO = "safedriver/safedriver/datalake/modelos/"
R2_TRACKING       = "safedriver/safedriver/datalake/raw/tracking_ssp.json"


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

# Fatores de período: apenas noite e madrugada são agravantes
# Manhã (06-11) e Tarde (12-17) são neutros — não há consenso regional
FATOR_PERIODO = {
    "MADRUGADA": 1.4,  # 00h–05h — agravante forte
    "MANHA":     1.0,  # 06h–11h — neutro
    "TARDE":     1.0,  # 12h–17h — neutro
    "NOITE":     1.3,  # 18h–23h — agravante
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
    """Classifica a hora em período do dia."""
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
    """Retorna o fator multiplicador pelo período — só noite e madrugada são agravantes."""
    periodo = classificar_periodo(hora_str)
    return FATOR_PERIODO[periodo]

def anonimizar_campo(valor: str, salt: str) -> str:
    """
    LGPD Privacy by Design — gera ID_AUDITORIA_ANON via SHA-256.
    O campo original é destruído após o hash.
    """
    raw = f"{salt}:{normalizar_texto(str(valor))}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def calcular_escore(rubrica: str, hora_str: str, perfil: str = "MOTORISTA") -> float:
    """
    Escore = peso_penal × multiplicador_perfil × fator_periodo
    Apenas noite (1.3) e madrugada (1.4) são agravantes.
    Manhã e tarde são neutros (1.0).
    """
    rubrica = normalizar_texto(rubrica)
    peso    = PESO_PENAL_BASE.get(rubrica, 1.0)
    mult    = MULTIPLICADOR_PERFIL.get(perfil, {}).get(rubrica, 1.0)
    fator   = fator_periodo(hora_str)
    return round(peso * mult * fator, 4)

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
            r = requests.post(webhook, json=payload, timeout=10)
            r.raise_for_status()
        except Exception as e:
            print(f"[Discord] Falha: {e}")

    def relatorio_executivo(self, run_id, tempo, n_ouro, mae, r2,
                             melhoria, top_municipio, top_shap_feature):
        self._enviar(self.ws, {"embeds": [{
            "title": "📊 SafeDriver — Relatório Executivo",
            "color": 3066993,
            "description": (
                f"Pipeline processou **{n_ouro:,} hexágonos** do Estado de São Paulo "
                f"e atualizou o mapa preditivo de risco com dados da SSP-SP."
            ),
            "fields": [
                {"name": "🏙️ Município mais crítico",   "value": str(top_municipio),   "inline": True},
                {"name": "🔍 Fator de risco dominante", "value": str(top_shap_feature), "inline": True},
                {"name": "📈 Melhoria do modelo",       "value": f"{melhoria:+.2f}%",  "inline": True},
                {"name": "🎯 Precisão (R²)",            "value": f"{r2:.4f}",           "inline": True},
                {"name": "⏱️ Tempo total",              "value": f"{tempo:.1f}s",       "inline": True},
                {"name": "🔖 Run ID",                  "value": run_id,                "inline": True},
            ],
            "footer": {"text": f"{NOME_SISTEMA} • {hora_brasilia().strftime('%d/%m/%Y %H:%M')} (Brasília)"},
        }]})

    def relatorio_operacional(self, run_id, n_raw, n_prata, n_ouro,
                               mae, r2, mae_ant, anos_processados,
                               status_bq, shap_top3):
        shap_txt = "\n".join([f"`{f}` → {v:.4f}" for f, v in shap_top3]) or "n/a"
        self._enviar(self.ws, {"embeds": [{
            "title": "⚙️ SafeDriver — Relatório Operacional",
            "color": 3447003,
            "fields": [
                {"name": "Versão",           "value": VERSAO_PIPELINE,                        "inline": True},
                {"name": "Run ID",           "value": run_id,                                 "inline": True},
                {"name": "Anos processados", "value": str(anos_processados),                  "inline": True},
                {"name": "Raw → Prata",      "value": f"{n_raw:,} → {n_prata:,} registros",  "inline": False},
                {"name": "Prata → Ouro",     "value": f"{n_prata:,} → {n_ouro:,} hexágonos", "inline": False},
                {"name": "MAE atual",        "value": f"{mae:.4f}",                           "inline": True},
                {"name": "MAE anterior",     "value": f"{mae_ant:.4f}",                       "inline": True},
                {"name": "R²",               "value": f"{r2:.4f}",                            "inline": True},
                {"name": "SHAP top 3",       "value": shap_txt,                               "inline": False},
                {"name": "BigQuery",         "value": status_bq,                              "inline": False},
            ],
            "footer": {"text": f"{NOME_SISTEMA} • {hora_brasilia().strftime('%d/%m/%Y %H:%M')} (Brasília)"},
        }]})

    def alerta_erro(self, run_id, titulo, erro_completo):
        trecho = str(erro_completo)[-1800:]
        self._enviar(self.we, {"embeds": [{
            "title": f"🚨 SafeDriver — FALHA CRÍTICA: {titulo}",
            "color": 15158332,
            "fields": [
                {"name": "Run ID",    "value": run_id,                    "inline": True},
                {"name": "Traceback", "value": f"```\n{trecho}\n```",     "inline": False},
            ],
            "footer": {"text": f"{NOME_SISTEMA} • {hora_brasilia().strftime('%d/%m/%Y %H:%M')} (Brasília)"},
        }]})

    def sem_novidades(self, run_id, tempo):
        self._enviar(self.ws, {"embeds": [{
            "title": "ℹ️ SafeDriver — Sem dados novos",
            "color": 9807270,
            "fields": [
                {"name": "Run ID", "value": run_id,           "inline": True},
                {"name": "Tempo",  "value": f"{tempo:.1f}s",  "inline": True},
            ],
            "footer": {"text": f"{NOME_SISTEMA} • {hora_brasilia().strftime('%d/%m/%Y %H:%M')} (Brasília)"},
        }]})


# ══════════════════════════════════════════════════════════════════════════════
# TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class TrackingSSP:
    def __init__(self, s3, bucket: str):
        self.s3      = s3
        self.bucket  = bucket
        self._estado: dict = {}
        self._carregar()

    def _carregar(self):
        try:
            obj          = self.s3.get_object(Bucket=self.bucket, Key=R2_TRACKING)
            self._estado = json.loads(obj["Body"].read().decode("utf-8"))
            print(f"[Tracking] {len(self._estado)} anos conhecidos: {list(self._estado.keys())}")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                print("[Tracking] tracking_ssp.json não encontrado — iniciando do zero.")
                self._estado = {}
            else:
                raise

    def _salvar(self):
        body = json.dumps(self._estado, ensure_ascii=False, indent=2, default=str)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=R2_TRACKING,
            Body=body.encode("utf-8"),
            ContentType="application/json"
        )

    def tamanho_conhecido(self, ano: int) -> int:
        return self._estado.get(str(ano), {}).get("tamanho_bytes", 0)

    def ultima_data_conhecida(self, ano: int):
        val = self._estado.get(str(ano), {}).get("ultima_data")
        if val:
            try:
                return pd.to_datetime(val)
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
                  f"{'processar (inédito)' if resultado else 'pular (já processado)'}")
            return resultado
        else:
            resultado = tamanho_atual != conhecido
            print(f"[Tracking] {ano} atual — R2={tamanho_atual:,}B "
                  f"conhecido={conhecido:,}B → {'ATUALIZADO' if resultado else 'sem mudança'}")
            return resultado

    def atualizar(self, ano: int, tamanho: int, ultima_data, n: int):
        self._estado[str(ano)] = {
            "tamanho_bytes":    tamanho,
            "ultima_data":      str(ultima_data) if ultima_data is not None else None,
            "registros":        n,
            "atualizado_em":    hora_brasilia().isoformat(),
        }
        self._salvar()
        print(f"[Tracking] {ano} atualizado — {n:,} registros, {tamanho:,} bytes")


# ══════════════════════════════════════════════════════════════════════════════
# MEMÓRIA DO MODELO
# ══════════════════════════════════════════════════════════════════════════════

class MemoriaModelo:
    CHAVE = "safedriver/safedriver/datalake/modelos/historico_mae.json"

    def __init__(self, s3, bucket: str):
        self.s3     = s3
        self.bucket = bucket
        self.dados  = []
        self._carregar()

    def _carregar(self):
        try:
            obj        = self.s3.get_object(Bucket=self.bucket, Key=self.CHAVE)
            self.dados = json.loads(obj["Body"].read().decode("utf-8"))
            print(f"[Memória] {len(self.dados)} execuções no histórico.")
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                print("[Memória] Sem histórico — primeira execução.")
                self.dados = []
            else:
                raise

    def mae_anterior(self) -> float:
        if not self.dados:
            return float("inf")
        return self.dados[-1].get("mae", float("inf"))

    def registrar(self, mae: float, r2: float, n: int):
        self.dados.append({
            "mae":         mae,
            "r2":          r2,
            "n_registros": n,
            "timestamp":   hora_brasilia().isoformat(),
            "versao":      VERSAO_PIPELINE,
        })
        body = json.dumps(self.dados[-50:], ensure_ascii=False, indent=2)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.CHAVE,
            Body=body.encode("utf-8"),
            ContentType="application/json"
        )


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SafeDriver:
    def __init__(self):
        self.run_id          = run_id_curto()
        self.t_inicio        = time.time()
        self.df_raw          = pl.DataFrame()
        self.anos_processados = []

        # Secrets — todos lidos do ambiente, nenhum hardcoded
        self.bucket    = sanitizar_secret(os.environ.get("R2_BUCKET_NAME", ""))
        endpoint       = sanitizar_secret(os.environ.get("R2_ENDPOINT_URL", ""))
        ak             = sanitizar_secret(os.environ.get("R2_ACCESS_KEY_ID", ""))
        sk             = sanitizar_secret(os.environ.get("R2_SECRET_ACCESS_KEY", ""))
        self.lgpd_salt = sanitizar_secret(os.environ.get("LGPD_SALT", ""))

        if not self.bucket:
            raise RuntimeError("Secret R2_BUCKET_NAME não definido.")
        if not self.lgpd_salt:
            raise RuntimeError("Secret LGPD_SALT não definido.")

        print(f"[{NOME_SISTEMA}] Iniciando run {self.run_id}")
        print(f"[R2] Bucket: {self.bucket}")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            region_name="auto"
        )

        self.tracking = TrackingSSP(self.s3, self.bucket)
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
                meta          = self.s3.head_object(Bucket=self.bucket, Key=chave)
                tamanho_atual = meta["ContentLength"]
                print(f"[R2] {chave} — {tamanho_atual:,} bytes")
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    print(f"[R2] {chave} não existe — pulando.")
                    continue
                raise

            if not self.tracking.precisa_processar(ano, tamanho_atual):
                continue

            print(f"[R2] Baixando {chave}...")
            obj = self.s3.get_object(Bucket=self.bucket, Key=chave)
            buf = BytesIO(obj["Body"].read())
            df  = pl.read_parquet(buf)
            print(f"[R2] {ano} — {len(df):,} registros | colunas sample: {df.columns[:6]}")

            ultima_data = self.tracking.ultima_data_conhecida(ano)
            if not self.tracking.e_ano_fechado(ano) and ultima_data is not None:
                if "DATA_OCORRENCIA_BO" in df.columns:
                    antes = len(df)
                    df = df.filter(pl.col("DATA_OCORRENCIA_BO") > pl.lit(ultima_data))
                    print(f"[R2] Incremental {ano}: {antes:,} → {len(df):,} novos")

            if df.is_empty():
                print(f"[R2] {ano} sem registros novos — pulando.")
                continue

            ultima = df["DATA_OCORRENCIA_BO"].max() if "DATA_OCORRENCIA_BO" in df.columns else None
            self.tracking.atualizar(ano, tamanho_atual, ultima, len(df))
            self.anos_processados.append(ano)
            frames.append(df)

        if not frames:
            self._log("sincronizar_raw_sem_dados_novos", {})
            print("[R2] Nenhum dado novo detectado.")
            return

        self.df_raw = pl.concat(frames, how="diagonal_relaxed")
        self._log("sincronizar_raw_fim", {"total_registros": len(self.df_raw)})

    # ── CONSTRUIR PRATA ──────────────────────────────────────────────────────

    def construir_prata(self) -> pl.DataFrame:
        if self.df_raw.is_empty():
            print("[Prata] df_raw vazio — nada a processar.")
            return pl.DataFrame()

        self._log("prata_inicio", {"registros_raw": len(self.df_raw)})
        df = renomear_sinonimos(self.df_raw)

        # Normalizar colunas de texto
        for col in ["RUBRICA", "NOME_MUNICIPIO", "NOME_DEPARTAMENTO",
                    "NOME_DELEGACIA", "BAIRRO", "NATUREZA_APURADA"]:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).map_elements(normalizar_texto, return_dtype=pl.Utf8).alias(col)
                )

        # Converter coordenadas
        for col in ["LATITUDE", "LONGITUDE"]:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False).alias(col)
                )

        # Filtrar coordenadas válidas do Estado de SP
        if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
            antes = len(df)
            df = df.filter(
                pl.col("LATITUDE").is_not_null()  &
                pl.col("LONGITUDE").is_not_null() &
                pl.col("LATITUDE").is_between(SP_LAT_MIN, SP_LAT_MAX) &
                pl.col("LONGITUDE").is_between(SP_LON_MIN, SP_LON_MAX)
            )
            print(f"[Prata] Filtro geográfico SP: {antes:,} → {len(df):,}")

        # LGPD — anonimizar campos identificáveis antes de qualquer agregação
        campos_pii = ["LOGRADOURO", "NUMERO_LOGRADOURO", "BAIRRO", "NOME_DELEGACIA"]
        salt = self.lgpd_salt
        for campo in campos_pii:
            if campo in df.columns:
                df = df.with_columns(
                    pl.col(campo)
                    .map_elements(lambda v: anonimizar_campo(str(v), salt), return_dtype=pl.Utf8)
                    .alias(campo)
                )
        print(f"[Prata] LGPD — campos anonimizados: {[c for c in campos_pii if c in df.columns]}")

        # Calcular período do dia
        hora_col = "HORA_OCORRENCIA_BO"
        if hora_col in df.columns:
            df = df.with_columns(
                pl.col(hora_col)
                .map_elements(classificar_periodo, return_dtype=pl.Utf8)
                .alias("PERIODO_DIA")
            )
            df = df.with_columns(
                pl.col(hora_col)
                .map_elements(fator_periodo, return_dtype=pl.Float64)
                .alias("FATOR_PERIODO")
            )
        else:
            df = df.with_columns([
                pl.lit("MANHA").alias("PERIODO_DIA"),
                pl.lit(1.0).alias("FATOR_PERIODO"),
            ])

        # Escore individual: volume × gravidade × período
        rubrica_col = "RUBRICA" if "RUBRICA" in df.columns else None
        if rubrica_col:
            df = df.with_columns(
                pl.struct([rubrica_col, hora_col] if hora_col in df.columns else [rubrica_col])
                .map_elements(
                    lambda r: calcular_escore(
                        r.get(rubrica_col, ""),
                        r.get(hora_col, "06:00") if hora_col in df.columns else "06:00"
                    ),
                    return_dtype=pl.Float64
                )
                .alias("ESCORE")
            )
        else:
            df = df.with_columns(pl.lit(1.0).alias("ESCORE"))

        # Índice H3
        if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
            df = df.with_columns(
                pl.struct(["LATITUDE", "LONGITUDE"])
                .map_elements(
                    lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], H3_RESOLUCAO)
                    if r["LATITUDE"] is not None and r["LONGITUDE"] is not None else None,
                    return_dtype=pl.Utf8
                )
                .alias("H3_INDEX")
            )
            antes = len(df)
            df = df.filter(pl.col("H3_INDEX").is_not_null())
            print(f"[Prata] H3 gerado: {antes:,} → {len(df):,} registros com índice válido")

        # Salvar prata no R2
        buf = BytesIO()
        df.write_parquet(buf)
        buf.seek(0)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{R2_PREFIXO_PRATA}prata_atual.parquet",
            Body=buf.read()
        )

        self._log("prata_fim", {"registros": len(df)})
        return df

    # ── CONSTRUIR OURO ───────────────────────────────────────────────────────

    def construir_ouro(self, df_prata: pl.DataFrame,
                       bq_project: str, bq_dataset: str, bq_cred: str):
        if df_prata.is_empty():
            print("[Ouro] prata vazia — abortando.")
            return None

        self._log("ouro_inicio", {"registros_prata": len(df_prata)})
        df = df_prata.to_pandas()

        if "H3_INDEX" not in df.columns or "ESCORE" not in df.columns:
            print("[Ouro] Colunas H3_INDEX ou ESCORE ausentes — abortando.")
            return None

        df["LATITUDE_F"]  = pd.to_numeric(df.get("LATITUDE"),  errors="coerce")
        df["LONGITUDE_F"] = pd.to_numeric(df.get("LONGITUDE"), errors="coerce")

        # IS_NOITE_MADRUGADA — para feature binária no modelo
        df["IS_NOITE_MADRUGADA"] = df.get("PERIODO_DIA", pd.Series(dtype=str)).isin(
            ["NOITE", "MADRUGADA"]
        ).astype(int)

        df["IS_PATRIMONIO"] = df.get("RUBRICA", pd.Series(dtype=str)).str.contains(
            "VEICULO|CARGA|LATROCINIO|MOTOCICLETA", na=False
        ).astype(int)

        df["IS_VIOLENCIA_PESSOA"] = df.get("RUBRICA", pd.Series(dtype=str)).str.contains(
            "HOMICIDIO|ESTUPRO|LESAO|ATROPELAMENTO", na=False
        ).astype(int)

        # Agregação temporal por hexágono e mês para lag features
        if "DATA_OCORRENCIA_BO" in df.columns:
            df["ANO_MES"] = pd.to_datetime(
                df["DATA_OCORRENCIA_BO"], errors="coerce"
            ).dt.to_period("M").astype(str)
        else:
            df["ANO_MES"] = "2026-01"

        agg_temporal = df.groupby(["H3_INDEX", "ANO_MES"]).agg(
            QTD_CRIMES           = ("ESCORE",             "count"),
            ESCORE_TOTAL         = ("ESCORE",             "sum"),
            ESCORE_MEDIO         = ("ESCORE",             "mean"),
            ESCORE_GRAVIDADE_MAX = ("ESCORE",             "max"),
            LATITUDE_MEDIA       = ("LATITUDE_F",         "mean"),
            LONGITUDE_MEDIA      = ("LONGITUDE_F",        "mean"),
            PROP_NOITE_MADRUGADA = ("IS_NOITE_MADRUGADA", "mean"),
            PROP_PATRIMONIO      = ("IS_PATRIMONIO",      "mean"),
            PROP_VIOLENCIA_PESSOA= ("IS_VIOLENCIA_PESSOA","mean"),
        ).reset_index()

        agg_temporal.sort_values(["H3_INDEX", "ANO_MES"], inplace=True)

        # Lag de 2 meses para compensar defasagem de publicação da SSP
        agg_temporal["ESCORE_LAG2"] = (
            agg_temporal.groupby("H3_INDEX")["ESCORE_TOTAL"].shift(2)
        )
        agg_temporal["QTD_LAG2"] = (
            agg_temporal.groupby("H3_INDEX")["QTD_CRIMES"].shift(2)
        )

        mes_max = agg_temporal["ANO_MES"].max()
        agg = agg_temporal[agg_temporal["ANO_MES"] == mes_max].copy()
        agg.dropna(subset=["ESCORE_LAG2"], inplace=True)

        if len(agg) < MIN_REGISTROS:
            print(f"[Ouro] Mês corrente insuficiente ({len(agg)}), usando agregação total.")
            agg = df.groupby("H3_INDEX").agg(
                QTD_CRIMES            = ("ESCORE",             "count"),
                ESCORE_TOTAL          = ("ESCORE",             "sum"),
                ESCORE_MEDIO          = ("ESCORE",             "mean"),
                ESCORE_GRAVIDADE_MAX  = ("ESCORE",             "max"),
                LATITUDE_MEDIA        = ("LATITUDE_F",         "mean"),
                LONGITUDE_MEDIA       = ("LONGITUDE_F",        "mean"),
                PROP_NOITE_MADRUGADA  = ("IS_NOITE_MADRUGADA", "mean"),
                PROP_PATRIMONIO       = ("IS_PATRIMONIO",      "mean"),
                PROP_VIOLENCIA_PESSOA = ("IS_VIOLENCIA_PESSOA","mean"),
            ).reset_index()
            agg["ESCORE_LAG2"] = agg["ESCORE_TOTAL"]
            agg["QTD_LAG2"]    = agg["QTD_CRIMES"]

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
            "LATITUDE_MEDIA",
            "LONGITUDE_MEDIA",
            "PROP_NOITE_MADRUGADA",
            "PROP_PATRIMONIO",
            "PROP_VIOLENCIA_PESSOA",
            "ESCORE_GRAVIDADE_MAX",
            "ESCORE_VIZ_1",
            "ESCORE_VIZ_2",
            "QTD_CRIMES_VIZ",
            "ESCORE_LAG2",
            "QTD_LAG2",
        ]
        target = "ESCORE_TOTAL"
        X = agg[features].fillna(0)
        y = agg[target]

        if len(agg) < MIN_REGISTROS:
            print(f"[Ouro] {len(agg)} hexágonos insuficientes — abortando.")
            return None

        # Ensemble LightGBM + CatBoost
        print("[Ouro] Treinando Ensemble LightGBM + CatBoost...")
        tscv = TimeSeriesSplit(n_splits=5)
        maes, r2s = [], []

        lgbm   = LGBMRegressor(n_estimators=500, learning_rate=0.04,
                                max_depth=6, num_leaves=63,
                                min_child_samples=20, random_state=42, verbose=-1)
        cat    = CatBoostRegressor(iterations=500, learning_rate=0.04,
                                   depth=6, random_seed=42, verbose=0,
                                   loss_function="RMSE")
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

        # SHAP — explicabilidade por hexágono para Looker
        shap_top3      = []
        shap_col_names = []
        try:
            print("[Ouro] Calculando SHAP...")
            explainer   = shap.TreeExplainer(modelo.estimators_[0])
            shap_values = explainer.shap_values(X)
            importancias = list(zip(features, np.abs(shap_values).mean(axis=0)))
            importancias.sort(key=lambda x: -x[1])
            shap_top3 = importancias[:3]
            print(f"[Ouro] SHAP top3: {shap_top3}")

            for i, feat in enumerate(features):
                col_shap = f"SHAP_{feat}"
                agg[col_shap] = shap_values[:, i]
                shap_col_names.append(col_shap)

        except Exception as e:
            print(f"[Ouro] SHAP não crítico: {e}")

        # Município mais crítico
        top_municipio = "N/A"
        if "NOME_MUNICIPIO" in df.columns:
            try:
                top_municipio = (
                    df.groupby("NOME_MUNICIPIO")["ESCORE"]
                    .sum().sort_values(ascending=False).index[0]
                )
            except Exception:
                pass

        top_shap_feature = shap_top3[0][0] if shap_top3 else "N/A"
        mae_ant  = self.memoria.mae_anterior()
        melhoria = ((mae_ant - mae) / mae_ant * 100) if mae_ant != float("inf") else 0.0
        self.memoria.registrar(mae, r2, len(agg))

        # Salvar modelo
        buf_modelo = BytesIO()
        joblib.dump(modelo, buf_modelo)
        buf_modelo.seek(0)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{R2_PREFIXO_MODELO}modelo_{NOME_SISTEMA}.pkl",
            Body=buf_modelo.read()
        )

        # Salvar ouro com SHAP
        ouro_pl  = pl.from_pandas(agg)
        buf_ouro = BytesIO()
        ouro_pl.write_parquet(buf_ouro)
        buf_ouro.seek(0)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{R2_PREFIXO_OURO}ouro_atual.parquet",
            Body=buf_ouro.read()
        )
        print(f"[Ouro] {len(agg):,} hexágonos salvos com colunas SHAP: {shap_col_names}")

        # BigQuery — projeto e dataset 100% via secrets
        status_bq = "skipped"
        if bq_project and bq_dataset and bq_cred:
            try:
                cred_info = json.loads(bq_cred)
                creds     = service_account.Credentials.from_service_account_info(
                    cred_info,
                    scopes=["https://www.googleapis.com/auth/bigquery"]
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
            "hexagonos":     len(agg),
            "mae":           round(mae, 4),
            "r2":            round(r2, 4),
            "shap_top1":     top_shap_feature,
            "top_municipio": top_municipio,
        })

        return (ouro_pl, mae, r2, mae_ant, melhoria, status_bq,
                len(self.df_raw), len(df_prata), top_municipio,
                top_shap_feature, shap_top3)

    # ── PROCESSAR ────────────────────────────────────────────────────────────

    def processar(self):
        self._log("pipeline_iniciado", {
            "sistema":  NOME_SISTEMA,
            "versao":   VERSAO_PIPELINE,
            "features": VERSAO_FEATURES,
        })

        # Todos via secret — nenhum hardcoded
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
                status_bq, shap_top3
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
        app.discord.alerta_erro(app.run_id, "Falha Sistêmica", err)
        sys.exit(1)
