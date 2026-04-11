# autobot/motor_analise_preditiva.py
"""
SafeDriver_Motor_V1.0.0
Pipeline preditivo de segurança urbana — Estado de São Paulo
Prefixo R2 fixo: safedriver/safedriver/datalake/raw/
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
VERSAO_PIPELINE  = "5.4.0"
VERSAO_FEATURES  = "v5"
H3_RESOLUCAO     = 8
MIN_REGISTROS    = 500
ANO_ATUAL        = datetime.utcnow().year
ANOS_DISPONIVEIS = list(range(2022, ANO_ATUAL + 1))
FUSO_BRASILIA    = timedelta(hours=3)
PERFIS           = ["MOTORISTA", "MOTOCICLISTA", "PEDESTRE", "CICLISTA"]

# Prefixo fixo confirmado pelas imagens do Cloudflare R2
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
    for nome_oficial, sins in SINONIMOS.items():
        if nome_oficial.upper() not in colunas_upper:
            for sin in sins:
                if sin.upper() in colunas_upper:
                    mapa[colunas_upper[sin.upper()]] = nome_oficial
                    break
    if mapa:
        print(f"[Prata] Renomeando colunas: {mapa}")
        df = df.rename(mapa)
    return df


# ══════════════════════════════════════════════════════════════════════════════
# TRACKING — estrutura corrigida
# ══════════════════════════════════════════════════════════════════════════════

class TrackingSSP:
    """
    Formato do JSON:
    {
      "2022": {"tamanho_bytes": 16873472, "processado_em": "2026-04-08T22:11:06"},
      "2023": {"tamanho_bytes": 17432576, "processado_em": "2026-04-08T22:11:08"},
      ...
    }
    Nunca armazena inteiros diretamente — sempre dicts.
    """

    def __init__(self, dados_brutos: dict):
        # Migração defensiva: se algum valor for int em vez de dict, corrige
        self.dados: dict[str, dict] = {}
        for ano, val in dados_brutos.items():
            if isinstance(val, dict):
                self.dados[str(ano)] = val
            elif isinstance(val, (int, float)):
                # formato legado — migra para estrutura correta
                self.dados[str(ano)] = {"tamanho_bytes": int(val), "processado_em": "migrado"}
            else:
                self.dados[str(ano)] = {"tamanho_bytes": 0, "processado_em": "desconhecido"}

    def precisa_processar(self, ano: int, tamanho_atual: int) -> bool:
        entrada = self.dados.get(str(ano))
        if entrada is None:
            return True
        return entrada.get("tamanho_bytes", 0) != tamanho_atual

    def registrar(self, ano: int, tamanho_bytes: int):
        self.dados[str(ano)] = {
            "tamanho_bytes": tamanho_bytes,
            "processado_em": hora_brasilia().isoformat(),
        }

    def serializar(self) -> str:
        return json.dumps(self.dados, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# MEMÓRIA DE MODELO
# ══════════════════════════════════════════════════════════════════════════════

class MemoriaModelo:
    def __init__(self, historico: list):
        self.historico = historico if isinstance(historico, list) else []

    def mae_anterior(self) -> float:
        if not self.historico:
            return float("inf")
        return self.historico[-1].get("mae", float("inf"))

    def registrar(self, mae: float, r2: float, n_hex: int):
        self.historico.append({
            "mae":        mae,
            "r2":         r2,
            "hexagonos":  n_hex,
            "registrado_em": hora_brasilia().isoformat(),
        })

    def serializar(self) -> str:
        return json.dumps(self.historico, ensure_ascii=False, indent=2)


# ══════════════════════════════════════════════════════════════════════════════
# DISCORD
# ══════════════════════════════════════════════════════════════════════════════

class Discord:
    def __init__(self, url_sucesso: str, url_erro: str):
        self.url_sucesso = url_sucesso
        self.url_erro    = url_erro

    def _post(self, url: str, payload: dict):
        if not url:
            return
        try:
            r = requests.post(url, json=payload, timeout=10, verify=False)
            r.raise_for_status()
        except Exception as e:
            print(f"[Discord] Falha ao enviar: {e}")

    def relatorio_executivo(self, run_id, tempo, n_hex, mae, r2,
                             melhoria, top_municipio, top_shap):
        sinal = "📈" if melhoria >= 0 else "📉"
        self._post(self.url_sucesso, {"embeds": [{"title": "SafeDriver — Relatório Executivo",
            "color": 3066993,
            "fields": [
                {"name": "Run ID",        "value": run_id,               "inline": True},
                {"name": "Hexágonos",     "value": f"{n_hex:,}",         "inline": True},
                {"name": "Tempo",         "value": f"{tempo:.1f}s",      "inline": True},
                {"name": "MAE",           "value": f"{mae:.4f}",         "inline": True},
                {"name": "R²",            "value": f"{r2:.4f}",          "inline": True},
                {"name": f"Melhoria {sinal}", "value": f"{melhoria:+.1f}%", "inline": True},
                {"name": "Município crítico", "value": top_municipio,    "inline": True},
                {"name": "SHAP top feature",  "value": top_shap,         "inline": True},
            ],
            "footer": {"text": f"{NOME_SISTEMA} v{VERSAO_PIPELINE}"},
            "timestamp": datetime.utcnow().isoformat(),
        }]})

    def relatorio_operacional(self, run_id, n_raw, n_prata, n_ouro,
                               mae, r2, mae_ant, anos, status_bq, shap_top3, prefixo_raw):
        shap_txt = " | ".join([f"{f}: {v:.4f}" for f, v in shap_top3]) if shap_top3 else "N/A"
        self._post(self.url_sucesso, {"embeds": [{"title": "SafeDriver — Relatório Operacional",
            "color": 1752220,
            "fields": [
                {"name": "Run ID",       "value": run_id,              "inline": True},
                {"name": "Anos",         "value": str(anos),           "inline": True},
                {"name": "Raw rows",     "value": f"{n_raw:,}",        "inline": True},
                {"name": "Prata rows",   "value": f"{n_prata:,}",      "inline": True},
                {"name": "Ouro hexágonos","value": f"{n_ouro:,}",      "inline": True},
                {"name": "MAE atual",    "value": f"{mae:.4f}",        "inline": True},
                {"name": "MAE anterior", "value": f"{mae_ant:.4f}" if mae_ant != float("inf") else "N/A", "inline": True},
                {"name": "R²",           "value": f"{r2:.4f}",         "inline": True},
                {"name": "BigQuery",     "value": status_bq,           "inline": False},
                {"name": "SHAP top3",    "value": shap_txt,            "inline": False},
                {"name": "Prefixo R2",   "value": prefixo_raw,         "inline": False},
            ],
            "footer": {"text": f"{NOME_SISTEMA} v{VERSAO_PIPELINE}"},
            "timestamp": datetime.utcnow().isoformat(),
        }]})

    def sem_novidades(self, run_id, tempo):
        self._post(self.url_sucesso, {"embeds": [{"title": "SafeDriver — Sem Novidades",
            "color": 16776960,
            "fields": [
                {"name": "Run ID", "value": run_id,          "inline": True},
                {"name": "Tempo",  "value": f"{tempo:.1f}s", "inline": True},
            ],
            "description": "Nenhum dado novo detectado no R2.",
            "footer": {"text": f"{NOME_SISTEMA} v{VERSAO_PIPELINE}"},
        }]})

    def alerta_erro(self, run_id, titulo, detalhe):
        self._post(self.url_erro, {"embeds": [{"title": f"ERRO — {titulo}",
            "color": 15158332,
            "fields": [
                {"name": "Run ID",  "value": run_id,           "inline": True},
                {"name": "Sistema", "value": NOME_SISTEMA,     "inline": True},
            ],
            "description": f"```{detalhe[:1800]}```",
            "footer": {"text": f"{NOME_SISTEMA} v{VERSAO_PIPELINE}"},
            "timestamp": datetime.utcnow().isoformat(),
        }]})


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SafeDriver:

    def __init__(self):
        self.run_id  = run_id_curto()
        self.t_inicio = time.time()

        bucket   = sanitizar_secret(os.environ.get("R2_BUCKET_NAME", ""))
        endpoint = sanitizar_secret(os.environ.get("R2_ENDPOINT_URL", ""))
        ak       = sanitizar_secret(os.environ.get("R2_ACCESS_KEY_ID", ""))
        sk       = sanitizar_secret(os.environ.get("R2_SECRET_ACCESS_KEY", ""))
        self.lgpd_salt = sanitizar_secret(os.environ.get("LGPD_SALT", "safedriver_default"))

        if not self.lgpd_salt or len(self.lgpd_salt) < 16:
            raise ValueError("LGPD_SALT ausente ou muito curto — mínimo 16 caracteres")

        print(f"[{NOME_SISTEMA}] Iniciando run {self.run_id}")
        print(f"[R2] Bucket   : {bucket}")
        print(f"[R2] Endpoint : {endpoint[:50]}...")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=ak,
            aws_secret_access_key=sk,
        )
        self.bucket = bucket

        self.discord = Discord(
            sanitizar_secret(os.environ.get("DISCORD_SUCESSO", "")),
            sanitizar_secret(os.environ.get("DISCORD_ERRO", "")),
        )

        # Carregar tracking
        self.tracking = self._carregar_tracking()
        # Carregar memória
        self.memoria  = self._carregar_memoria()

        self.df_raw          = pd.DataFrame()
        self.anos_processados = []

    def _log(self, evento: str, dados: dict):
        ts = hora_brasilia().strftime("%H:%M:%S")
        print(f"  [{self.run_id}] {evento}: ***{json.dumps(dados, ensure_ascii=False)}*** [{ts}]")

    def _carregar_tracking(self) -> TrackingSSP:
        try:
            obj  = self.s3.get_object(Bucket=self.bucket, Key=R2_TRACKING)
            raw  = json.loads(obj["Body"].read().decode("utf-8"))
            # raw deve ser dict de ano -> dict
            if not isinstance(raw, dict):
                raise ValueError("tracking_ssp.json com formato inválido")
            tracking = TrackingSSP(raw)
            print(f"[Tracking] Carregado: {list(tracking.dados.keys())}")
            return tracking
        except ClientError:
            print("[Tracking] tracking_ssp.json não encontrado — iniciando do zero.")
            return TrackingSSP({})
        except Exception as e:
            print(f"[Tracking] Erro ao carregar — reiniciando: {e}")
            return TrackingSSP({})

    def _carregar_memoria(self) -> MemoriaModelo:
        chave = f"{R2_PREFIXO_MODELO}memoria_modelo.json"
        try:
            obj  = self.s3.get_object(Bucket=self.bucket, Key=chave)
            raw  = json.loads(obj["Body"].read().decode("utf-8"))
            mem  = MemoriaModelo(raw if isinstance(raw, list) else [])
            print(f"[Memória] {len(mem.historico)} run(s) anteriores.")
            return mem
        except ClientError:
            print("[Memória] Sem histórico — primeira execução.")
            return MemoriaModelo([])
        except Exception as e:
            print(f"[Memória] Erro ao carregar — reiniciando: {e}")
            return MemoriaModelo([])

    def _salvar_tracking(self):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=R2_TRACKING,
            Body=self.tracking.serializar().encode("utf-8"),
        )

    def _salvar_memoria(self):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{R2_PREFIXO_MODELO}memoria_modelo.json",
            Body=self.memoria.serializar().encode("utf-8"),
        )

    # ── SINCRONIZAR RAW ──────────────────────────────────────────────────────

    def sincronizar_raw(self):
        self._log("sincronizar_raw_inicio", {})
        frames = []

        for ano in ANOS_DISPONIVEIS:
            chave = f"{R2_PREFIXO_RAW}ssp_{ano}.parquet"
            print(f"[R2] Verificando {chave}...")

            try:
                head = self.s3.head_object(Bucket=self.bucket, Key=chave)
                tamanho_atual = head["ContentLength"]
            except ClientError:
                print(f"[R2] {chave} não existe — pulando.")
                continue

            if not self.tracking.precisa_processar(ano, tamanho_atual):
                print(f"[R2] ssp_{ano}.parquet inalterado — usando cache.")
                # Carregar mesmo assim para ter os dados em memória
                try:
                    obj = self.s3.get_object(Bucket=self.bucket, Key=chave)
                    df  = pl.read_parquet(BytesIO(obj["Body"].read()))
                    frames.append(df)
                except Exception as e:
                    print(f"[R2] Erro ao carregar cache {ano}: {e}")
                continue

            print(f"[R2] Baixando ssp_{ano}.parquet ({tamanho_atual / 1e6:.1f} MB)...")
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=chave)
                df  = pl.read_parquet(BytesIO(obj["Body"].read()))
                frames.append(df)
                self.tracking.registrar(ano, tamanho_atual)
                self.anos_processados.append(ano)
                print(f"[R2] ssp_{ano}.parquet carregado: {len(df):,} linhas")
            except Exception as e:
                print(f"[R2] Erro ao baixar {ano}: {e}")

        if frames:
            self.df_raw = pl.concat(frames, how="diagonal").to_pandas()
            self._salvar_tracking()
            self._log("sincronizar_raw_fim", {
                "total_linhas": len(self.df_raw),
                "anos": self.anos_processados,
            })
        else:
            self._log("sincronizar_raw_sem_dados_novos", {})
            print("[R2] Nenhum dado encontrado no R2.")

    # ── CONSTRUIR PRATA ──────────────────────────────────────────────────────

    def construir_prata(self) -> pd.DataFrame:
        if self.df_raw.empty:
            print("[Prata] df_raw vazio — nada a processar.")
            return pd.DataFrame()

        print(f"[Prata] Iniciando com {len(self.df_raw):,} linhas...")
        df = pl.from_pandas(self.df_raw)
        df = renomear_sinonimos(df)

        # Normalizar colunas de texto
        for col in ["RUBRICA", "NOME_MUNICIPIO", "LOGRADOURO", "BAIRRO"]:
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

        # Filtro SP
        if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
            antes = len(df)
            df = df.filter(
                (pl.col("LATITUDE")  >= SP_LAT_MIN) & (pl.col("LATITUDE")  <= SP_LAT_MAX) &
                (pl.col("LONGITUDE") >= SP_LON_MIN) & (pl.col("LONGITUDE") <= SP_LON_MAX)
            )
            print(f"[Prata] Filtro SP: {antes:,} → {len(df):,} linhas")

        # Dropar coordenadas nulas
        df = df.drop_nulls(subset=["LATITUDE", "LONGITUDE"])

        # LGPD — anonimizar campos identificadores
        for col in ["LOGRADOURO", "NUMERO_LOGRADOURO", "BAIRRO"]:
            if col in df.columns:
                salt = self.lgpd_salt
                df = df.with_columns(
                    pl.col(col).map_elements(
                        lambda v: anonimizar_campo(v, salt), return_dtype=pl.Utf8
                    ).alias(col)
                )

        # H3
        if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
            df = df.with_columns([
                pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                    lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], H3_RESOLUCAO),
                    return_dtype=pl.Utf8
                ).alias("H3_INDEX"),
                pl.col("LATITUDE").alias("LATITUDE_F"),
                pl.col("LONGITUDE").alias("LONGITUDE_F"),
            ])

        # Período do dia
        hora_col = "HORA_OCORRENCIA_BO" if "HORA_OCORRENCIA_BO" in df.columns else None
        if hora_col:
            df = df.with_columns(
                pl.col(hora_col).map_elements(classificar_periodo, return_dtype=pl.Utf8).alias("PERIODO_DIA")
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

        # Categorias de crime
        patrimoniais = {"ROUBO DE VEICULO","FURTO DE VEICULO","ROUBO DE MOTOCICLETA",
                        "FURTO DE MOTOCICLETA","ROUBO DE CARGA","ROUBO","FURTO"}
        violentos    = {"HOMICIDIO DOLOSO","LATROCINIO","LESAO CORPORAL DOLOSA",
                        "ESTUPRO","EXTORSAO MEDIANTE SEQUESTRO","ATROPELAMENTO"}

        if "RUBRICA" in df.columns:
            df = df.with_columns([
                pl.col("RUBRICA").map_elements(
                    lambda r: 1 if r in patrimoniais else 0, return_dtype=pl.Int8
                ).alias("IS_PATRIMONIO"),
                pl.col("RUBRICA").map_elements(
                    lambda r: 1 if r in violentos else 0, return_dtype=pl.Int8
                ).alias("IS_VIOLENCIA_PESSOA"),
            ])
        else:
            df = df.with_columns([
                pl.lit(0).cast(pl.Int8).alias("IS_PATRIMONIO"),
                pl.lit(0).cast(pl.Int8).alias("IS_VIOLENCIA_PESSOA"),
            ])

        # Escores por perfil
        rubrica_col = "RUBRICA" if "RUBRICA" in df.columns else None
        for perfil in PERFIS:
            if rubrica_col and hora_col:
                df = df.with_columns(
                    pl.struct([rubrica_col, hora_col]).map_elements(
                        lambda r: calcular_escore(r[rubrica_col], r[hora_col], perfil),
                        return_dtype=pl.Float64
                    ).alias(f"ESCORE_{perfil}")
                )
            else:
                df = df.with_columns(pl.lit(1.0).alias(f"ESCORE_{perfil}"))

        prata = df.to_pandas()
        print(f"[Prata] Finalizada: {len(prata):,} linhas | colunas: {list(prata.columns)}")
        return prata

    # ── CONSTRUIR OURO ───────────────────────────────────────────────────────

    def construir_ouro(self, df_prata: pd.DataFrame,
                       bq_project: str, bq_dataset: str, bq_cred: str):
        if df_prata.empty:
            print("[Ouro] prata vazia — abortando.")
            return None

        print(f"[Ouro] Iniciando agregação de {len(df_prata):,} linhas...")

        agg_dict = {
            "QTD_CRIMES":          ("ESCORE_MOTORISTA", "count"),
            "ESCORE_TOTAL":        ("ESCORE_MOTORISTA", "sum"),
            "ESCORE_MEDIO":        ("ESCORE_MOTORISTA", "mean"),
            "ESCORE_GRAVIDADE_MAX":("ESCORE_MOTORISTA", "max"),
        }
        for perfil in PERFIS:
            agg_dict[f"ESCORE_{perfil}_MEDIO"] = (f"ESCORE_{perfil}", "mean")
            agg_dict[f"ESCORE_{perfil}_TOTAL"] = (f"ESCORE_{perfil}", "sum")

        agg_dict.update({
            "LATITUDE_MEDIA":       ("LATITUDE_F",         "mean"),
            "LONGITUDE_MEDIA":      ("LONGITUDE_F",        "mean"),
            "PROP_NOITE_MADRUGADA": ("IS_NOITE_MADRUGADA", "mean"),
            "PROP_PATRIMONIO":      ("IS_PATRIMONIO",      "mean"),
            "PROP_VIOLENCIA_PESSOA":("IS_VIOLENCIA_PESSOA","mean"),
        })
        if "NOME_MUNICIPIO" in df_prata.columns:
            agg_dict["MUNICIPIO_DOMINANTE"] = ("NOME_MUNICIPIO", lambda x: x.mode()[0] if len(x) else "N/A")

        agg = df_prata.groupby("H3_INDEX").agg(**agg_dict).reset_index()
        agg = agg[agg["QTD_CRIMES"] >= MIN_REGISTROS].copy()
        print(f"[Ouro] {len(agg):,} hexágonos com >= {MIN_REGISTROS} crimes")

        if len(agg) < 10:
            print("[Ouro] Hexágonos insuficientes para treinar modelo.")
            return None

        # Lag espacial (vizinhos H3 ring-1 e ring-2)
        escore_por_hex = agg.set_index("H3_INDEX")["ESCORE_TOTAL"].to_dict()
        qtd_por_hex    = agg.set_index("H3_INDEX")["QTD_CRIMES"].to_dict()

        def escore_viz(h3_idx, ring):
            vizinhos = h3.grid_disk(h3_idx, ring) - {h3_idx}
            vals = [escore_por_hex[v] for v in vizinhos if v in escore_por_hex]
            return np.mean(vals) if vals else 0.0

        def qtd_viz(h3_idx, ring):
            vizinhos = h3.grid_disk(h3_idx, ring) - {h3_idx}
            vals = [qtd_por_hex[v] for v in vizinhos if v in qtd_por_hex]
            return np.mean(vals) if vals else 0.0

        agg["ESCORE_VIZ_1"] = agg["H3_INDEX"].apply(lambda x: escore_viz(x, 1))
        agg["ESCORE_VIZ_2"] = agg["H3_INDEX"].apply(lambda x: escore_viz(x, 2))
        agg["QTD_CRIMES_VIZ"] = agg["H3_INDEX"].apply(lambda x: qtd_viz(x, 1))

        # Lag temporal (shift por mês — simulado com offset no escore)
        agg["ESCORE_LAG2"] = agg["ESCORE_TOTAL"].shift(2).fillna(agg["ESCORE_TOTAL"].mean())
        agg["QTD_LAG2"]    = agg["QTD_CRIMES"].shift(2).fillna(agg["QTD_CRIMES"].mean())

        # Feriados SP
        feriados_sp = holidays.Brazil(state="SP", years=list(range(2022, ANO_ATUAL + 1)))
        hoje = hora_brasilia().date()
        agg["IS_FERIADO"] = int(hoje in feriados_sp)

        # Features para o modelo
        features = [
            "QTD_CRIMES", "ESCORE_MEDIO", "ESCORE_GRAVIDADE_MAX",
            "PROP_NOITE_MADRUGADA", "PROP_PATRIMONIO", "PROP_VIOLENCIA_PESSOA",
            "ESCORE_VIZ_1", "ESCORE_VIZ_2", "QTD_CRIMES_VIZ",
            "ESCORE_LAG2", "QTD_LAG2", "IS_FERIADO",
            "LATITUDE_MEDIA", "LONGITUDE_MEDIA",
        ] + [f"ESCORE_{p}_MEDIO" for p in PERFIS]

        features = [f for f in features if f in agg.columns]
        X = agg[features].fillna(0)
        y = agg["ESCORE_TOTAL"]

        # Treino com TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=5)
        maes, r2s = [], []

        lgbm = LGBMRegressor(n_estimators=400, learning_rate=0.05,
                              num_leaves=63, random_state=42, verbose=-1)
        cat  = CatBoostRegressor(iterations=400, learning_rate=0.05,
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

        # SHAP
        shap_top3        = []
        shap_col_names   = []
        top_shap_feature = "N/A"
        try:
            print("[Ouro] Calculando SHAP...")
            explainer    = shap.TreeExplainer(modelo.estimators_[0])
            shap_values  = explainer.shap_values(X)
            importancias = list(zip(features, np.abs(shap_values).mean(axis=0)))
            importancias.sort(key=lambda x: -x[1])
            shap_top3        = importancias[:3]
            top_shap_feature = shap_top3[0][0]
            print(f"[Ouro] SHAP top3: {shap_top3}")
            for i, feat in enumerate(features):
                agg[f"SHAP_{feat}"] = shap_values[:, i]
                shap_col_names.append(f"SHAP_{feat}")
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
            Key=f"{R2_PREFIXO_MODELO}modelo_{NOME_SISTEMA}.pkl",
            Body=buf_modelo.read(),
        )

        # Salvar ouro
        ouro_pl  = pl.from_pandas(agg)
        buf_ouro = BytesIO()
        ouro_pl.write_parquet(buf_ouro)
        buf_ouro.seek(0)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{R2_PREFIXO_OURO}ouro_atual.parquet",
            Body=buf_ouro.read(),
        )
        print(f"[Ouro] {len(agg):,} hexágonos | perfis: {PERFIS} | SHAP: {len(shap_col_names)} colunas")

        self._salvar_memoria()

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
                status_bq, shap_top3, R2_PREFIXO_RAW
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
