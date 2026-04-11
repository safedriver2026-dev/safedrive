# autobot/motor_analise_preditiva.py
"""
SafeDriver_Motor_V1.0.0
Pipeline preditivo de segurança urbana — Estado de São Paulo
Prefixo R2 descoberto automaticamente via listagem do bucket.
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
VERSAO_PIPELINE  = "5.1.0"
VERSAO_FEATURES  = "v5"
H3_RESOLUCAO     = 8
MIN_REGISTROS    = 500
ANO_ATUAL        = datetime.utcnow().year
ANOS_DISPONIVEIS = list(range(2022, ANO_ATUAL + 1))
FUSO_BRASILIA    = timedelta(hours=3)

# Sufixos fixos relativos ao prefixo raw descoberto dinamicamente
SUFIXO_PRATA  = "prata/"
SUFIXO_OURO   = "ouro/"
SUFIXO_MODELO = "modelos/"
SUFIXO_TRACK  = "tracking_ssp.json"

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

# Apenas noite e madrugada são agravantes — manhã e tarde são neutros
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
# DESCOBERTA AUTOMÁTICA DE PREFIXO R2
# ══════════════════════════════════════════════════════════════════════════════

def descobrir_prefixo_raw(s3_client, bucket: str) -> str | None:
    """
    Lista o bucket inteiro e encontra onde estão os arquivos ssp_XXXX.parquet.
    Retorna o prefixo (ex: 'safedriver/datalake/raw/') ou None se não encontrar.
    Nunca depende de um caminho hardcoded.
    """
    print(f"[R2] Descobrindo prefixo raw no bucket '{bucket}'...")
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            chave = obj["Key"]
            nome  = chave.split("/")[-1]
            # Procura qualquer arquivo ssp_XXXX.parquet
            if nome.startswith("ssp_") and nome.endswith(".parquet"):
                prefixo = chave[: chave.rfind("/") + 1]
                print(f"[R2] Prefixo raw encontrado automaticamente: '{prefixo}'")
                return prefixo

    print("[R2] AVISO: Nenhum arquivo ssp_*.parquet encontrado no bucket.")
    print("[R2] Listando tudo que existe no bucket para diagnóstico:")
    for page in paginator.paginate(Bucket=bucket):
        for obj in page.get("Contents", []):
            print(f"       {obj['Key']}  ({obj['Size']:,} bytes)")
    return None

def derivar_prefixos(prefixo_raw: str) -> dict:
    """
    A partir do prefixo raw descoberto, deriva todos os outros prefixos
    substituindo 'raw/' pelo sufixo desejado na mesma hierarquia.
    """
    base = prefixo_raw.rsplit("raw/", 1)[0]
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
            obj  = self.s3.get_object(Bucket=self.bucket, Key=self.chave)
            dados = json.loads(obj["Body"].read().decode("utf-8"))
            print(f"[Tracking] Carregado: {list(dados.keys())}")
            return dados
        except ClientError as e:
            if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                print("[Tracking] tracking_ssp.json não encontrado — iniciando do zero.")
                return {}
            raise

    def _salvar(self):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.chave,
            Body=json.dumps(self.dados, default=str).encode("utf-8"),
            ContentType="application/json",
        )

    def precisa_processar(self, ano: int, tamanho_atual: int) -> bool:
        entrada = self.dados.get(str(ano), {})
        tam_ant = entrada.get("tamanho_bytes", 0)
        if tamanho_atual != tam_ant:
            print(f"[Tracking] {ano}: {tam_ant:,} → {tamanho_atual:,} bytes — PROCESSAR")
            return True
        print(f"[Tracking] {ano}: sem alteração ({tamanho_atual:,} bytes) — pular")
        return False

    def e_ano_fechado(self, ano: int) -> bool:
        return ano < ANO_ATUAL

    def ultima_data_conhecida(self, ano: int):
        val = self.dados.get(str(ano), {}).get("ultima_data")
        if val:
            try:
                return datetime.fromisoformat(str(val))
            except Exception:
                return None
        return None

    def atualizar(self, ano: int, tamanho: int, ultima_data, n_registros: int):
        self.dados[str(ano)] = {
            "tamanho_bytes": tamanho,
            "ultima_data":   str(ultima_data) if ultima_data else None,
            "n_registros":   n_registros,
            "atualizado_em": hora_brasilia().isoformat(),
        }
        self._salvar()
        print(f"[Tracking] {ano} atualizado — {n_registros:,} registros")


# ══════════════════════════════════════════════════════════════════════════════
# MEMÓRIA DO MODELO
# ══════════════════════════════════════════════════════════════════════════════

class MemoriaModelo:
    def __init__(self, s3, bucket: str, prefixo_modelo: str):
        self.s3     = s3
        self.bucket = bucket
        self.chave  = f"{prefixo_modelo}historico_mae.json"
        self.dados  = self._carregar()

    def _carregar(self) -> dict:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.chave)
            return json.loads(obj["Body"].read().decode("utf-8"))
        except ClientError:
            print("[Memória] Sem histórico — primeira execução.")
            return {"historico": []}

    def _salvar(self):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.chave,
            Body=json.dumps(self.dados).encode("utf-8"),
            ContentType="application/json",
        )

    def mae_anterior(self) -> float:
        hist = self.dados.get("historico", [])
        return hist[-1]["mae"] if hist else float("inf")

    def registrar(self, mae: float, r2: float, n_hex: int):
        self.dados.setdefault("historico", []).append({
            "mae":      mae,
            "r2":       r2,
            "n_hex":    n_hex,
            "data":     hora_brasilia().isoformat(),
            "versao":   VERSAO_PIPELINE,
        })
        self._salvar()


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
                               status_bq, shap_top3, prefixo_raw):
        shap_txt = "\n".join([f"`{f}` → {v:.4f}" for f, v in shap_top3]) or "n/a"
        self._enviar(self.ws, {"embeds": [{
            "title": "⚙️ SafeDriver — Relatório Operacional",
            "color": 3447003,
            "fields": [
                {"name": "Versão",           "value": VERSAO_PIPELINE,                        "inline": True},
                {"name": "Run ID",           "value": run_id,                                 "inline": True},
                {"name": "Prefixo Raw",      "value": f"`{prefixo_raw}`",                     "inline": False},
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
                {"name": "Run ID",    "value": run_id,                "inline": True},
                {"name": "Traceback", "value": f"```\n{trecho}\n```", "inline": False},
            ],
            "footer": {"text": f"{NOME_SISTEMA} • {hora_brasilia().strftime('%d/%m/%Y %H:%M')} (Brasília)"},
        }]})

    def sem_novidades(self, run_id, tempo):
        self._enviar(self.ws, {"embeds": [{
            "title": "ℹ️ SafeDriver — Sem dados novos",
            "color": 9807270,
            "fields": [
                {"name": "Run ID", "value": run_id,          "inline": True},
                {"name": "Tempo",  "value": f"{tempo:.1f}s", "inline": True},
            ],
            "footer": {"text": f"{NOME_SISTEMA} • {hora_brasilia().strftime('%d/%m/%Y %H:%M')} (Brasília)"},
        }]})

    def alerta_prefixo_nao_encontrado(self, run_id, bucket):
        self._enviar(self.we, {"embeds": [{
            "title": "🚨 SafeDriver — Prefixo Raw Não Encontrado",
            "color": 15158332,
            "description": (
                f"Nenhum arquivo `ssp_*.parquet` encontrado no bucket `{bucket}`.\n"
                f"Verifique se os dados SSP foram carregados corretamente no R2."
            ),
            "fields": [
                {"name": "Run ID", "value": run_id, "inline": True},
                {"name": "Bucket", "value": bucket, "inline": True},
            ],
            "footer": {"text": f"{NOME_SISTEMA} • {hora_brasilia().strftime('%d/%m/%Y %H:%M')} (Brasília)"},
        }]})


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SafeDriver:
    def __init__(self):
        self.run_id   = run_id_curto()
        self.t_inicio = time.time()
        self.df_raw   = pl.DataFrame()
        self.anos_processados = []
        self.prefixos = {}

        print(f"[{NOME_SISTEMA}] Iniciando run {self.run_id}")

        # Secrets
        endpoint = sanitizar_secret(os.environ.get("R2_ENDPOINT_URL", ""))
        ak       = sanitizar_secret(os.environ.get("R2_ACCESS_KEY_ID", ""))
        sk       = sanitizar_secret(os.environ.get("R2_SECRET_ACCESS_KEY", ""))
        self.bucket   = sanitizar_secret(os.environ.get("R2_BUCKET_NAME", ""))
        self.lgpd_salt = sanitizar_secret(os.environ.get("LGPD_SALT", ""))

        if not self.lgpd_salt:
            raise RuntimeError("LGPD_SALT não definido — pipeline abortado por segurança.")

        print(f"[R2] Bucket   : {self.bucket}")
        print(f"[R2] Endpoint : {endpoint[:40]}...")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
            region_name="auto",
        )

        # Descoberta automática do prefixo
        prefixo_raw = descobrir_prefixo_raw(self.s3, self.bucket)
        if not prefixo_raw:
            self.discord = Discord(
                sanitizar_secret(os.environ.get("DISCORD_SUCESSO", "")),
                sanitizar_secret(os.environ.get("DISCORD_ERRO", "")),
            )
            self.discord.alerta_prefixo_nao_encontrado(self.run_id, self.bucket)
            raise RuntimeError(
                f"Nenhum arquivo ssp_*.parquet encontrado no bucket '{self.bucket}'. "
                f"Verifique se os dados SSP foram carregados corretamente no R2."
            )

        self.prefixos = derivar_prefixos(prefixo_raw)
        print(f"[R2] Prefixos derivados: {json.dumps(self.prefixos, indent=2)}")

        self.tracking = TrackingSSP(self.s3, self.bucket, self.prefixos["track"])
        self.memoria  = MemoriaModelo(self.s3, self.bucket, self.prefixos["modelo"])
        self.discord  = Discord(
            sanitizar_secret(os.environ.get("DISCORD_SUCESSO", "")),
            sanitizar_secret(os.environ.get("DISCORD_ERRO", "")),
        )

    def _log(self, evento: str, dados: dict):
        ts = hora_brasilia().strftime("%H:%M:%S")
        print(f"  [{self.run_id}] {evento}: {dados} [{ts}]")

    # ── SINCRONIZAR RAW ──────────────────────────────────────────────────────

    def sincronizar_raw(self):
        self._log("sincronizar_raw_inicio", {})
        frames = []

        for ano in ANOS_DISPONIVEIS:
            chave = f"{self.prefixos['raw']}ssp_{ano}.parquet"

            try:
                meta          = self.s3.head_object(Bucket=self.bucket, Key=chave)
                tamanho_atual = meta["ContentLength"]
                print(f"[R2] {chave} — {tamanho_atual:,} bytes")
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
            print(f"[R2] {ano} — {len(df):,} registros | colunas: {df.columns[:6]}")

            ultima_data = self.tracking.ultima_data_conhecida(ano)
            if not self.tracking.e_ano_fechado(ano) and ultima_data is not None:
                if "DATA_OCORRENCIA_BO" in df.columns:
                    antes = len(df)
                    df = df.filter(pl.col("DATA_OCORRENCIA_BO") > pl.lit(ultima_data))
                    print(f"[R2] Incremental {ano}: {antes:,} → {len(df):,} registros novos")

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
        print(f"[R2] Total consolidado: {len(self.df_raw):,} registros")

    # ── CONSTRUIR PRATA ──────────────────────────────────────────────────────

    def construir_prata(self) -> pl.DataFrame:
        if self.df_raw.is_empty():
            print("[Prata] df_raw vazio — nada a processar.")
            return pl.DataFrame()

        print(f"[Prata] Iniciando limpeza — {len(self.df_raw):,} registros")
        df = self.df_raw.clone()

        # Renomear sinônimos
        df = renomear_sinonimos(df)

        # Normalizar textos com Polars
        for col in ["NOME_MUNICIPIO", "LOGRADOURO", "BAIRRO", "RUBRICA",
                    "NOME_DEPARTAMENTO", "NOME_SECCIONAL", "NOME_DELEGACIA"]:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Utf8).fill_null("").map_elements(
                        normalizar_texto, return_dtype=pl.Utf8
                    ).alias(col)
                )

        # Coordenadas válidas dentro de SP
        if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
            antes = len(df)
            df = df.with_columns([
                pl.col("LATITUDE").cast(pl.Float64,  strict=False),
                pl.col("LONGITUDE").cast(pl.Float64, strict=False),
            ])
            df = df.filter(
                pl.col("LATITUDE").is_not_null()  &
                pl.col("LONGITUDE").is_not_null() &
                pl.col("LATITUDE").is_between(SP_LAT_MIN, SP_LAT_MAX) &
                pl.col("LONGITUDE").is_between(SP_LON_MIN, SP_LON_MAX)
            )
            print(f"[Prata] Coordenadas válidas SP: {antes:,} → {len(df):,}")

        # Data de ocorrência
        if "DATA_OCORRENCIA_BO" in df.columns:
            df = df.with_columns(
                pl.col("DATA_OCORRENCIA_BO").cast(pl.Utf8).str.to_datetime(
                    format=None, strict=False
                ).alias("DATA_OCORRENCIA_BO")
            )
            df = df.filter(pl.col("DATA_OCORRENCIA_BO").is_not_null())

        # LGPD — anonimizar campos sensíveis antes de qualquer agregação
        salt = self.lgpd_salt
        for col in ["LOGRADOURO", "NUMERO_LOGRADOURO", "BAIRRO"]:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).map_elements(
                        lambda v: anonimizar_campo(str(v), salt),
                        return_dtype=pl.Utf8
                    ).alias(f"ID_AUDITORIA_{col}")
                ).drop(col)

        # H3 Index
        if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
            df = df.with_columns(
                pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                    lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], H3_RESOLUCAO),
                    return_dtype=pl.Utf8
                ).alias("H3_INDEX")
            )
            df = df.filter(pl.col("H3_INDEX").is_not_null())

        # Período do dia e flags de agravante
        if "HORA_OCORRENCIA_BO" in df.columns:
            df = df.with_columns(
                pl.col("HORA_OCORRENCIA_BO").cast(pl.Utf8).fill_null("12:00").map_elements(
                    classificar_periodo, return_dtype=pl.Utf8
                ).alias("PERIODO_DIA")
            )
            df = df.with_columns(
                pl.col("PERIODO_DIA").map_elements(
                    lambda p: 1 if p in ("NOITE", "MADRUGADA") else 0,
                    return_dtype=pl.Int8
                ).alias("IS_NOITE_MADRUGADA")
            )
        else:
            df = df.with_columns([
                pl.lit("MANHA").alias("PERIODO_DIA"),
                pl.lit(0).cast(pl.Int8).alias("IS_NOITE_MADRUGADA"),
            ])

        # Flags de categoria de crime
        crimes_patrimonio = {
            "ROUBO DE VEICULO", "FURTO DE VEICULO", "ROUBO DE MOTOCICLETA",
            "FURTO DE MOTOCICLETA", "ROUBO DE CARGA", "ROUBO", "FURTO"
        }
        crimes_violencia = {
            "HOMICIDIO DOLOSO", "LATROCINIO", "LESAO CORPORAL DOLOSA",
            "ESTUPRO", "EXTORSAO MEDIANTE SEQUESTRO", "ATROPELAMENTO"
        }

        if "RUBRICA" in df.columns:
            df = df.with_columns([
                pl.col("RUBRICA").map_elements(
                    lambda r: 1 if r in crimes_patrimonio else 0,
                    return_dtype=pl.Int8
                ).alias("IS_PATRIMONIO"),
                pl.col("RUBRICA").map_elements(
                    lambda r: 1 if r in crimes_violencia else 0,
                    return_dtype=pl.Int8
                ).alias("IS_VIOLENCIA_PESSOA"),
            ])
        else:
            df = df.with_columns([
                pl.lit(0).cast(pl.Int8).alias("IS_PATRIMONIO"),
                pl.lit(0).cast(pl.Int8).alias("IS_VIOLENCIA_PESSOA"),
            ])

        # Escore individual
        rubrica_col = "RUBRICA" if "RUBRICA" in df.columns else None
        hora_col    = "HORA_OCORRENCIA_BO" if "HORA_OCORRENCIA_BO" in df.columns else None

        if rubrica_col and hora_col:
            df = df.with_columns(
                pl.struct([rubrica_col, hora_col]).map_elements(
                    lambda r: calcular_escore(r[rubrica_col], r[hora_col]),
                    return_dtype=pl.Float64
                ).alias("ESCORE")
            )
        else:
            df = df.with_columns(pl.lit(1.0).alias("ESCORE"))

        # ANO_MES para série temporal
        if "DATA_OCORRENCIA_BO" in df.columns:
            df = df.with_columns(
                pl.col("DATA_OCORRENCIA_BO").dt.strftime("%Y-%m").alias("ANO_MES")
            )

        # Coordenadas finais como float limpo
        if "LATITUDE" in df.columns:
            df = df.with_columns([
                pl.col("LATITUDE").alias("LATITUDE_F"),
                pl.col("LONGITUDE").alias("LONGITUDE_F"),
            ])

        # Salvar prata no R2
        buf = BytesIO()
        df.write_parquet(buf)
        buf.seek(0)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefixos['prata']}prata_atual.parquet",
            Body=buf.read(),
        )
        print(f"[Prata] {len(df):,} registros salvos em {self.prefixos['prata']}prata_atual.parquet")
        return df

    # ── CONSTRUIR OURO ───────────────────────────────────────────────────────

    def construir_ouro(self, df_prata: pl.DataFrame,
                       bq_project: str, bq_dataset: str, bq_cred: str):
        if df_prata.is_empty():
            print("[Ouro] prata vazia — abortando.")
            return None

        self._log("ouro_inicio", {"registros_prata": len(df_prata)})
        df = df_prata.to_pandas()

        # Agregação por hexágono — volume + gravidade
        agg = df.groupby("H3_INDEX").agg(
            QTD_CRIMES            =("ESCORE",             "count"),
            ESCORE_TOTAL          =("ESCORE",             "sum"),
            ESCORE_MEDIO          =("ESCORE",             "mean"),
            ESCORE_GRAVIDADE_MAX  =("ESCORE",             "max"),
            LATITUDE_MEDIA        =("LATITUDE_F",         "mean"),
            LONGITUDE_MEDIA       =("LONGITUDE_F",        "mean"),
            PROP_NOITE_MADRUGADA  =("IS_NOITE_MADRUGADA", "mean"),
            PROP_PATRIMONIO       =("IS_PATRIMONIO",      "mean"),
            PROP_VIOLENCIA_PESSOA =("IS_VIOLENCIA_PESSOA","mean"),
        ).reset_index()

        # Município mais crítico por hexágono
        if "NOME_MUNICIPIO" in df.columns:
            mun_por_hex = (
                df.groupby("H3_INDEX")["NOME_MUNICIPIO"]
                .agg(lambda x: x.value_counts().index[0])
                .reset_index()
                .rename(columns={"NOME_MUNICIPIO": "MUNICIPIO_DOMINANTE"})
            )
            agg = agg.merge(mun_por_hex, on="H3_INDEX", how="left")

        # Lagged inference — shift de 2 períodos por hexágono
        if "ANO_MES" in df.columns:
            serie = (
                df.groupby(["H3_INDEX", "ANO_MES"])
                .agg(ESCORE_LAG_SRC=("ESCORE", "sum"), QTD_LAG_SRC=("ESCORE", "count"))
                .reset_index()
                .sort_values(["H3_INDEX", "ANO_MES"])
            )
            serie["ESCORE_LAG2"] = serie.groupby("H3_INDEX")["ESCORE_LAG_SRC"].shift(2)
            serie["QTD_LAG2"]    = serie.groupby("H3_INDEX")["QTD_LAG_SRC"].shift(2)
            lag_agg = serie.groupby("H3_INDEX")[["ESCORE_LAG2", "QTD_LAG2"]].mean().reset_index()
            agg = agg.merge(lag_agg, on="H3_INDEX", how="left")
        else:
            agg["ESCORE_LAG2"] = agg["ESCORE_TOTAL"]
            agg["QTD_LAG2"]    = agg["QTD_CRIMES"]

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
            print(f"[Ouro] {len(agg)} hexágonos insuficientes (mínimo {MIN_REGISTROS}) — abortando.")
            return None

        # Ensemble LightGBM + CatBoost
        print("[Ouro] Treinando Ensemble LightGBM + CatBoost...")
        tscv = TimeSeriesSplit(n_splits=5)
        maes, r2s = [], []

        lgbm = LGBMRegressor(
            n_estimators=500, learning_rate=0.04,
            max_depth=6, num_leaves=63,
            min_child_samples=20, random_state=42, verbose=-1
        )
        cat = CatBoostRegressor(
            iterations=500, learning_rate=0.04,
            depth=6, random_seed=42, verbose=0,
            loss_function="RMSE"
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

        # SHAP — explicabilidade por hexágono para Looker
        shap_top3      = []
        shap_col_names = []
        top_shap_feature = "N/A"
        try:
            print("[Ouro] Calculando SHAP...")
            explainer   = shap.TreeExplainer(modelo.estimators_[0])
            shap_values = explainer.shap_values(X)
            importancias = list(zip(features, np.abs(shap_values).mean(axis=0)))
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

        # Município mais crítico global
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
        print(f"[Ouro] {len(agg):,} hexágonos salvos | SHAP cols: {shap_col_names}")

        # BigQuery — 100% via secrets
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
