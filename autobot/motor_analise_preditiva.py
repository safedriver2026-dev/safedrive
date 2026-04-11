# autobot/motor_analise_preditiva.py
"""
SafeDriver_Motor_V1.0.0
Pipeline preditivo de segurança urbana — Estado de São Paulo
Estratégia: R2 → fallback download SSP-SP → salva no R2.
Escore por perfil: MOTORISTA, MOTOCICLISTA, PEDESTRE, CICLISTA.
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
VERSAO_PIPELINE  = "5.5.0"
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

# URL base da SSP-SP — só o ano muda
SSP_URL_TEMPLATE  = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
SSP_TIMEOUT       = 300   # segundos — arquivo pode ter 100+ MB
SSP_MAX_TENTATIVAS = 3


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
    colunas_atuais = set(df.columns)
    for nome_canonico, sinonimos in SINONIMOS.items():
        if nome_canonico not in colunas_atuais:
            for sin in sinonimos:
                if sin in colunas_atuais:
                    mapa[sin] = nome_canonico
                    break
    return df.rename(mapa) if mapa else df


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD SSP-SP
# ══════════════════════════════════════════════════════════════════════════════

def baixar_ssp(ano: int) -> pd.DataFrame | None:
    """
    Baixa o Excel da SSP-SP para o ano informado.
    Tenta SSP_MAX_TENTATIVAS vezes com backoff.
    Retorna DataFrame pandas ou None se o ano não existir.
    """
    url = SSP_URL_TEMPLATE.format(ano=ano)
    print(f"[SSP] Baixando {ano} → {url}")

    for tentativa in range(1, SSP_MAX_TENTATIVAS + 1):
        try:
            resp = requests.get(
                url,
                timeout=SSP_TIMEOUT,
                verify=False,
                headers={"User-Agent": "SafeDriver-Pipeline/5.5.0"},
                stream=True,
            )
            if resp.status_code == 404:
                print(f"[SSP] {ano} não disponível ainda (404) — pulando.")
                return None
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}")

            # Lê em memória — evita arquivo temporário no runner
            buf = BytesIO()
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                buf.write(chunk)
            buf.seek(0)
            tamanho_mb = buf.getbuffer().nbytes / (1024 * 1024)
            print(f"[SSP] {ano} baixado — {tamanho_mb:.1f} MB")

            # Tenta as duas engines disponíveis no ambiente
            for engine in ("openpyxl", "xlrd"):
                try:
                    buf.seek(0)
                    df = pd.read_excel(buf, engine=engine, dtype=str)
                    print(f"[SSP] {ano} lido com engine={engine} — {len(df):,} linhas")
                    return df
                except Exception as e_eng:
                    print(f"[SSP] engine {engine} falhou: {e_eng}")

            print(f"[SSP] {ano} — nenhuma engine conseguiu ler o arquivo.")
            return None

        except Exception as e:
            print(f"[SSP] tentativa {tentativa}/{SSP_MAX_TENTATIVAS} falhou: {e}")
            if tentativa < SSP_MAX_TENTATIVAS:
                time.sleep(10 * tentativa)

    print(f"[SSP] {ano} — todas as tentativas esgotadas.")
    return None


def converter_e_salvar_r2(df_raw: pd.DataFrame, ano: int, s3, bucket: str) -> int:
    """Converte DataFrame pandas → parquet e salva no R2. Retorna tamanho em bytes."""
    df_pl  = pl.from_pandas(df_raw.astype(str))
    buf    = BytesIO()
    df_pl.write_parquet(buf)
    buf.seek(0)
    dados  = buf.read()
    chave  = f"{R2_PREFIXO_RAW}ssp_{ano}.parquet"
    s3.put_object(Bucket=bucket, Key=chave, Body=dados)
    print(f"[R2] ssp_{ano}.parquet salvo — {len(dados) / (1024*1024):.1f} MB")
    return len(dados)


# ══════════════════════════════════════════════════════════════════════════════
# TRACKING
# ══════════════════════════════════════════════════════════════════════════════

class TrackingSSP:
    def __init__(self, s3, bucket: str):
        self.s3     = s3
        self.bucket = bucket
        self.dados  = {}
        self._carregar()

    def _carregar(self):
        try:
            obj  = self.s3.get_object(Bucket=self.bucket, Key=R2_TRACKING)
            raw  = json.loads(obj["Body"].read().decode("utf-8"))
            # Migração defensiva: garante que cada valor é dict
            for ano, val in raw.items():
                if isinstance(val, dict):
                    self.dados[str(ano)] = val
                elif isinstance(val, (int, float)):
                    self.dados[str(ano)] = {"tamanho_bytes": int(val), "linhas": 0}
                else:
                    self.dados[str(ano)] = {"tamanho_bytes": 0, "linhas": 0}
            print(f"[Tracking] Carregado — anos: {list(self.dados.keys())}")
        except self.s3.exceptions.NoSuchKey:
            print("[Tracking] tracking_ssp.json não encontrado — iniciando do zero.")
        except Exception as e:
            print(f"[Tracking] Erro ao carregar — iniciando do zero: {e}")

    def salvar(self):
        payload = json.dumps(self.dados, ensure_ascii=False, indent=2).encode("utf-8")
        self.s3.put_object(Bucket=self.bucket, Key=R2_TRACKING, Body=payload)

    def precisa_processar(self, ano: int, tamanho_atual: int) -> bool:
        entrada = self.dados.get(str(ano), {})
        if not isinstance(entrada, dict):
            return True
        return entrada.get("tamanho_bytes", 0) != tamanho_atual

    def registrar(self, ano: int, tamanho_bytes: int, linhas: int):
        self.dados[str(ano)] = {
            "tamanho_bytes": tamanho_bytes,
            "linhas":        linhas,
            "atualizado_em": hora_brasilia().isoformat(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MEMÓRIA DE DESEMPENHO
# ══════════════════════════════════════════════════════════════════════════════

class MemoriaDesempenho:
    CHAVE = "safedriver/safedriver/datalake/raw/memoria_desempenho.json"

    def __init__(self, s3, bucket: str):
        self.s3     = s3
        self.bucket = bucket
        self._dados = {}
        self._carregar()

    def _carregar(self):
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.CHAVE)
            self._dados = json.loads(obj["Body"].read().decode("utf-8"))
            print(f"[Memória] Histórico carregado — runs: {len(self._dados)}")
        except Exception:
            print("[Memória] Sem histórico — primeira execução.")

    def salvar(self):
        payload = json.dumps(self._dados, ensure_ascii=False, indent=2).encode("utf-8")
        self.s3.put_object(Bucket=self.bucket, Key=self.CHAVE, Body=payload)

    def mae_anterior(self) -> float:
        if not self._dados:
            return float("inf")
        ultimo = sorted(self._dados.keys())[-1]
        return self._dados[ultimo].get("mae", float("inf"))

    def registrar(self, mae: float, r2: float, n_hexagonos: int):
        chave = hora_brasilia().strftime("%Y-%m-%dT%H:%M:%S")
        self._dados[chave] = {"mae": mae, "r2": r2, "hexagonos": n_hexagonos}


# ══════════════════════════════════════════════════════════════════════════════
# DISCORD
# ══════════════════════════════════════════════════════════════════════════════

class DiscordNotifier:
    def __init__(self, webhook_sucesso: str, webhook_erro: str):
        self.ws = webhook_sucesso
        self.we = webhook_erro

    def _post(self, webhook: str, payload: dict):
        if not webhook:
            return
        try:
            requests.post(webhook, json=payload, timeout=15, verify=False)
        except Exception as e:
            print(f"[Discord] Falha ao enviar: {e}")

    def relatorio_executivo(self, run_id, tempo, n_hex, mae, r2, melhoria, top_mun, top_feat):
        emoji = "📈" if melhoria > 0 else "📉"
        msg   = (
            f"**SafeDriver | Run `{run_id}`**\n"
            f"✅ Pipeline concluído em {tempo:.1f}s\n"
            f"🔷 Hexágonos: {n_hex:,} | MAE: {mae:.4f} | R²: {r2:.4f}\n"
            f"{emoji} Melhoria MAE: {melhoria:+.2f}%\n"
            f"📍 Município crítico: **{top_mun}**\n"
            f"🧠 Feature SHAP #1: `{top_feat}`"
        )
        self._post(self.ws, {"content": msg})

    def relatorio_operacional(self, run_id, n_raw, n_prata, n_ouro,
                               mae, r2, mae_ant, anos, status_bq, shap_top3, prefixo_raw):
        shap_str = " | ".join([f"`{f}` {v:.4f}" for f, v in shap_top3]) if shap_top3 else "N/A"
        msg = (
            f"**[OPS] Run `{run_id}`**\n"
            f"📦 Raw: {n_raw:,} | Prata: {n_prata:,} | Ouro: {n_ouro:,}\n"
            f"📅 Anos: {anos}\n"
            f"📊 MAE ant: {mae_ant:.4f} → atual: {mae:.4f} | R²: {r2:.4f}\n"
            f"🗄️ BigQuery: {status_bq}\n"
            f"🧠 SHAP top3: {shap_str}\n"
            f"🗂️ Prefixo R2: `{prefixo_raw}`"
        )
        self._post(self.ws, {"content": msg})

    def sem_novidades(self, run_id, tempo):
        self._post(self.ws, {"content": f"**SafeDriver | Run `{run_id}`** — Sem dados novos ({tempo:.1f}s)"})

    def alerta_erro(self, run_id, titulo, detalhe):
        msg = f"🚨 **ERRO [{titulo}] | Run `{run_id}`**\n```{detalhe[:1800]}```"
        self._post(self.we, {"content": msg})


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SafeDriver:

    def __init__(self):
        self.t_inicio = time.time()
        self.run_id   = run_id_curto()

        # Secrets
        endpoint  = sanitizar_secret(os.environ.get("R2_ENDPOINT_URL", ""))
        key_id    = sanitizar_secret(os.environ.get("R2_ACCESS_KEY_ID", ""))
        secret    = sanitizar_secret(os.environ.get("R2_SECRET_ACCESS_KEY", ""))
        self.bucket = sanitizar_secret(os.environ.get("R2_BUCKET_NAME", ""))
        self.salt   = sanitizar_secret(os.environ.get("LGPD_SALT", ""))

        if not self.salt:
            raise RuntimeError("LGPD_SALT não definido — abortando por segurança LGPD.")

        print(f"[{NOME_SISTEMA}] Iniciando run {self.run_id}")
        print(f"[R2] Bucket   : {self.bucket}")
        print(f"[R2] Endpoint : {endpoint[:40]}...")

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            region_name="auto",
        )

        self.tracking        = TrackingSSP(self.s3, self.bucket)
        self.memoria         = MemoriaDesempenho(self.s3, self.bucket)
        self.anos_processados = []
        self.df_raw          = pd.DataFrame()

        ws = sanitizar_secret(os.environ.get("DISCORD_SUCESSO", ""))
        we = sanitizar_secret(os.environ.get("DISCORD_ERRO", ""))
        self.discord = DiscordNotifier(ws, we)

    def _log(self, evento: str, dados: dict):
        ts  = hora_brasilia().strftime("%H:%M:%S")
        msg = json.dumps(dados, ensure_ascii=False)
        print(f"  [{self.run_id}] {evento}: ***{msg}*** [{ts}]")

    # ── SINCRONIZAR RAW ──────────────────────────────────────────────────────

    def sincronizar_raw(self):
        self._log("sincronizar_raw_inicio", {})
        frames = []

        for ano in ANOS_DISPONIVEIS:
            chave = f"{R2_PREFIXO_RAW}ssp_{ano}.parquet"
            tamanho_r2 = 0

            # 1) Verifica se já existe no R2
            try:
                head = self.s3.head_object(Bucket=self.bucket, Key=chave)
                tamanho_r2 = head["ContentLength"]
                print(f"[R2] ssp_{ano}.parquet encontrado — {tamanho_r2 / (1024*1024):.1f} MB")
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    print(f"[R2] ssp_{ano}.parquet não existe — tentando SSP-SP...")
                else:
                    print(f"[R2] Erro ao checar ssp_{ano}.parquet: {e}")

            # 2) Se não existe no R2, baixa da SSP-SP e salva
            if tamanho_r2 == 0:
                df_ssp = baixar_ssp(ano)
                if df_ssp is None or df_ssp.empty:
                    print(f"[SSP] {ano} — sem dados disponíveis, pulando.")
                    continue
                tamanho_r2 = converter_e_salvar_r2(df_ssp, ano, self.s3, self.bucket)
                self.tracking.registrar(ano, tamanho_r2, len(df_ssp))
                self.tracking.salvar()

            # 3) Verifica se precisa reprocessar
            if not self.tracking.precisa_processar(ano, tamanho_r2):
                print(f"[R2] ssp_{ano}.parquet inalterado — carregando do cache.")

            # 4) Carrega o parquet do R2
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=chave)
                df  = pl.read_parquet(BytesIO(obj["Body"].read())).to_pandas()
                frames.append(df)
                self.anos_processados.append(str(ano))
                self.tracking.registrar(ano, tamanho_r2, len(df))
                print(f"[R2] ssp_{ano}.parquet carregado — {len(df):,} linhas")
            except Exception as e:
                print(f"[R2] Erro ao ler ssp_{ano}.parquet: {e}")

        self.tracking.salvar()

        if frames:
            self.df_raw = pd.concat(frames, ignore_index=True)
            print(f"[Raw] Total consolidado: {len(self.df_raw):,} registros | anos: {self.anos_processados}")
            self._log("sincronizar_raw_fim", {
                "total_registros": len(self.df_raw),
                "anos": self.anos_processados
            })
        else:
            print("[Raw] Nenhum dado disponível — pipeline encerrado.")
            self._log("sincronizar_raw_sem_dados", {})

    # ── CONSTRUIR PRATA ──────────────────────────────────────────────────────

    def construir_prata(self) -> pd.DataFrame:
        if self.df_raw.empty:
            print("[Prata] df_raw vazio — nada a processar.")
            return pd.DataFrame()

        self._log("prata_inicio", {"registros_raw": len(self.df_raw)})

        df = pl.from_pandas(self.df_raw.astype(str))
        df = renomear_sinonimos(df)
        df = df.with_columns([pl.col(c).map_elements(normalizar_texto, return_dtype=pl.Utf8)
                               for c in df.columns if df[c].dtype == pl.Utf8])

        # Filtro SP
        for col_lat in ["LATITUDE", "LATITUDE_BO"]:
            if col_lat in df.columns:
                df = df.rename({col_lat: "LATITUDE"}) if col_lat != "LATITUDE" else df
                break
        for col_lon in ["LONGITUDE", "LONGITUDE_BO"]:
            if col_lon in df.columns:
                df = df.rename({col_lon: "LONGITUDE"}) if col_lon != "LONGITUDE" else df
                break

        df = df.to_pandas()

        # Converter coordenadas
        for col in ["LATITUDE", "LONGITUDE"]:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].str.replace(",", ".").str.strip(), errors="coerce"
                )

        if "LATITUDE" in df.columns and "LONGITUDE" in df.columns:
            df = df[
                df["LATITUDE"].between(SP_LAT_MIN, SP_LAT_MAX) &
                df["LONGITUDE"].between(SP_LON_MIN, SP_LON_MAX)
            ].copy()

        if df.empty or len(df) < MIN_REGISTROS:
            print(f"[Prata] Registros insuficientes após filtro SP: {len(df)}")
            return pd.DataFrame()

        # H3
        df["H3_INDEX"] = df.apply(
            lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], H3_RESOLUCAO)
            if pd.notna(r.get("LATITUDE")) and pd.notna(r.get("LONGITUDE")) else None,
            axis=1
        )
        df = df.dropna(subset=["H3_INDEX"])

        # Período do dia
        hora_col = "HORA_OCORRENCIA_BO" if "HORA_OCORRENCIA_BO" in df.columns else None
        if hora_col:
            df["PERIODO_DIA"]        = df[hora_col].apply(classificar_periodo)
            df["IS_NOITE_MADRUGADA"] = df["PERIODO_DIA"].isin(["NOITE", "MADRUGADA"]).astype(int)
        else:
            df["PERIODO_DIA"]        = "MANHA"
            df["IS_NOITE_MADRUGADA"] = 0

        # Escores por perfil
        if "RUBRICA" in df.columns and hora_col:
            for perfil in PERFIS:
                df[f"ESCORE_{perfil}"] = df.apply(
                    lambda r: calcular_escore(r.get("RUBRICA", ""), r.get(hora_col, "00:00"), perfil),
                    axis=1
                )
            df["ESCORE"] = df[f"ESCORE_{PERFIS[0]}"]  # MOTORISTA como base
        else:
            for perfil in PERFIS:
                df[f"ESCORE_{perfil}"] = 1.0
            df["ESCORE"] = 1.0

        # Flags de categoria
        crimes_patrimonio = {
            "ROUBO DE VEICULO","FURTO DE VEICULO","ROUBO DE MOTOCICLETA",
            "FURTO DE MOTOCICLETA","ROUBO DE CARGA","FURTO","ROUBO"
        }
        crimes_violencia = {
            "HOMICIDIO DOLOSO","LATROCINIO","LESAO CORPORAL DOLOSA",
            "ESTUPRO","ATROPELAMENTO"
        }
        if "RUBRICA" in df.columns:
            df["IS_PATRIMONIO"]       = df["RUBRICA"].isin(crimes_patrimonio).astype(int)
            df["IS_VIOLENCIA_PESSOA"] = df["RUBRICA"].isin(crimes_violencia).astype(int)
        else:
            df["IS_PATRIMONIO"]       = 0
            df["IS_VIOLENCIA_PESSOA"] = 0

        # LGPD — anonimizar campos sensíveis
        campos_sensiveis = [c for c in ["LOGRADOURO", "NUMERO_LOGRADOURO", "BAIRRO"] if c in df.columns]
        for campo in campos_sensiveis:
            df[campo] = df[campo].apply(lambda v: anonimizar_campo(str(v), self.salt))

        # Data
        if "DATA_OCORRENCIA_BO" in df.columns:
            df["DATA_OCORRENCIA_BO"] = pd.to_datetime(df["DATA_OCORRENCIA_BO"], errors="coerce")

        # Salvar prata
        prata_pl = pl.from_pandas(df)
        buf      = BytesIO()
        prata_pl.write_parquet(buf)
        buf.seek(0)
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{R2_PREFIXO_PRATA}prata_atual.parquet",
            Body=buf.read(),
        )

        print(f"[Prata] {len(df):,} registros processados e salvos.")
        self._log("prata_fim", {"registros": len(df)})
        return df

    # ── CONSTRUIR OURO ───────────────────────────────────────────────────────

    def construir_ouro(self, df_prata: pd.DataFrame,
                       bq_project: str, bq_dataset: str, bq_cred: str):
        if df_prata.empty:
            print("[Ouro] prata vazia — abortando.")
            return None

        self._log("ouro_inicio", {"registros_prata": len(df_prata)})

        feriados_sp = holidays.Brazil(state="SP", years=ANOS_DISPONIVEIS)

        # Agregação por hexágono
        agg_base = {
            "QTD_CRIMES":   ("H3_INDEX", "count"),
            "ESCORE_TOTAL": ("ESCORE",   "sum"),
            "ESCORE_MEDIO": ("ESCORE",   "mean"),
        }
        if "LATITUDE" in df_prata.columns:
            agg_base["LATITUDE_MEDIA"]  = ("LATITUDE",  "mean")
            agg_base["LONGITUDE_MEDIA"] = ("LONGITUDE", "mean")

        agg = df_prata.groupby("H3_INDEX").agg(**agg_base).reset_index()

        # Escores por perfil agregados
        for perfil in PERFIS:
            col = f"ESCORE_{perfil}"
            if col in df_prata.columns:
                agg_perfil = df_prata.groupby("H3_INDEX")[col].mean().reset_index()
                agg_perfil.columns = ["H3_INDEX", col]
                agg = agg.merge(agg_perfil, on="H3_INDEX", how="left")

        # Gravidade máxima
        agg["ESCORE_GRAVIDADE_MAX"] = df_prata.groupby("H3_INDEX")["ESCORE"].max().values \
            if len(agg) == len(df_prata.groupby("H3_INDEX")) \
            else agg["ESCORE_MEDIO"] * 1.2

        # Proporções
        for flag, col_prop in [("IS_NOITE_MADRUGADA","PROP_NOITE_MADRUGADA"),
                                ("IS_PATRIMONIO",     "PROP_PATRIMONIO"),
                                ("IS_VIOLENCIA_PESSOA","PROP_VIOLENCIA_PESSOA")]:
            if flag in df_prata.columns:
                prop = df_prata.groupby("H3_INDEX")[flag].mean().reset_index()
                prop.columns = ["H3_INDEX", col_prop]
                agg = agg.merge(prop, on="H3_INDEX", how="left")
            else:
                agg[col_prop] = 0.0

        # Município dominante
        if "NOME_MUNICIPIO" in df_prata.columns:
            mun = (df_prata.groupby("H3_INDEX")["NOME_MUNICIPIO"]
                   .agg(lambda x: x.mode()[0] if not x.empty else "N/A")
                   .reset_index())
            mun.columns = ["H3_INDEX", "MUNICIPIO_DOMINANTE"]
            agg = agg.merge(mun, on="H3_INDEX", how="left")

        # Features temporais
        if "DATA_OCORRENCIA_BO" in df_prata.columns:
            df_prata["IS_FERIADO"] = df_prata["DATA_OCORRENCIA_BO"].apply(
                lambda d: 1 if pd.notna(d) and d.date() in feriados_sp else 0
            )
            fer = df_prata.groupby("H3_INDEX")["IS_FERIADO"].mean().reset_index()
            fer.columns = ["H3_INDEX", "IS_FERIADO"]
            agg = agg.merge(fer, on="H3_INDEX", how="left")
        else:
            agg["IS_FERIADO"] = 0.0

        # Lag e vizinhança
        escore_map = agg.set_index("H3_INDEX")["ESCORE_TOTAL"].to_dict()
        agg["ESCORE_LAG2"]    = agg["ESCORE_TOTAL"] * 0.9
        agg["QTD_LAG2"]       = agg["QTD_CRIMES"]   * 0.9
        agg["ESCORE_VIZ_1"]   = agg["H3_INDEX"].apply(
            lambda h: np.mean([escore_map.get(v, 0) for v in h3.grid_disk(h, 1) if v != h]) or 0
        )
        agg["ESCORE_VIZ_2"]   = agg["H3_INDEX"].apply(
            lambda h: np.mean([escore_map.get(v, 0) for v in h3.grid_disk(h, 2) if v != h]) or 0
        )
        agg["QTD_CRIMES_VIZ"] = agg["ESCORE_VIZ_1"]

        # Modelo
        features = [
            "QTD_CRIMES", "ESCORE_MEDIO", "ESCORE_GRAVIDADE_MAX",
            "PROP_NOITE_MADRUGADA", "PROP_PATRIMONIO", "PROP_VIOLENCIA_PESSOA",
            "ESCORE_LAG2", "QTD_LAG2", "ESCORE_VIZ_1", "ESCORE_VIZ_2",
            "QTD_CRIMES_VIZ", "IS_FERIADO",
        ]
        features = [f for f in features if f in agg.columns]
        X = agg[features].fillna(0)
        y = agg["ESCORE_TOTAL"]

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
        shap_top3, shap_col_names, top_shap_feature = [], [], "N/A"
        try:
            explainer    = shap.TreeExplainer(modelo.estimators_[0])
            shap_values  = explainer.shap_values(X)
            importancias = sorted(
                zip(features, np.abs(shap_values).mean(axis=0)),
                key=lambda x: -x[1]
            )
            shap_top3        = importancias[:3]
            top_shap_feature = shap_top3[0][0]
            print(f"[Ouro] SHAP top3: {shap_top3}")
            for i, feat in enumerate(features):
                agg[f"SHAP_{feat}"] = shap_values[:, i]
                shap_col_names.append(f"SHAP_{feat}")
        except Exception as e:
            print(f"[Ouro] SHAP não crítico: {e}")

        # Município crítico
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
        self.memoria.salvar()

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
