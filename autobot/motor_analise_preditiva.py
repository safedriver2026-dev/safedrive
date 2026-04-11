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

VERSAO_PIPELINE  = "4.3.0"
VERSAO_FEATURES  = "v4"
H3_RESOLUCAO     = 8
MIN_REGISTROS    = 500
ANO_ATUAL        = datetime.utcnow().year
ANOS_DISPONIVEIS = list(range(2022, ANO_ATUAL + 1))

# Prefixo exato conforme estrutura R2: bucket=safedriver / safedriver/datalake/raw/
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


# ══════════════════════════════════════════════════════════════════════════════
# TRACKING — controle incremental por tamanho de arquivo
# ══════════════════════════════════════════════════════════════════════════════

class TrackingSSP:
    """
    Persiste no R2 o tamanho em bytes de cada ssp_XXXX.parquet.
    Lógica central:
      - Ano fechado (< ano atual): processa UMA vez, nunca mais.
      - Ano atual: processa sempre que o tamanho mudar.
    """

    def __init__(self, s3, bucket: str):
        self.s3     = s3
        self.bucket = bucket
        self._estado: dict = self._carregar()

    def _carregar(self) -> dict:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=R2_TRACKING)
            return json.loads(obj["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                print("[Tracking] tracking_ssp.json não encontrado — iniciando do zero.")
                return {}
            raise

    def _salvar(self):
        payload = json.dumps(self._estado, indent=2, ensure_ascii=False).encode("utf-8")
        self.s3.put_object(
            Bucket=self.bucket,
            Key=R2_TRACKING,
            Body=payload,
            ContentType="application/json"
        )
        print("[Tracking] tracking_ssp.json salvo no R2.")

    def tamanho_conhecido(self, ano: int) -> int:
        return self._estado.get(str(ano), {}).get("tamanho_bytes", 0)

    def ultima_data_conhecida(self, ano: int):
        s = self._estado.get(str(ano), {}).get("ultima_data")
        if s:
            try:
                return pd.Timestamp(s)
            except Exception:
                return None
        return None

    def e_ano_fechado(self, ano: int) -> bool:
        return ano < ANO_ATUAL

    def ja_processado(self, ano: int) -> bool:
        return self._estado.get(str(ano), {}).get("processado", False)

    def precisa_processar(self, ano: int, tamanho_atual: int) -> bool:
        if self.e_ano_fechado(ano):
            if self.ja_processado(ano):
                print(f"[Tracking] {ano} fechado e já processado — pular.")
                return False
            print(f"[Tracking] {ano} fechado mas nunca processado — processar.")
            return True
        # Ano atual: compara tamanho
        conhecido = self.tamanho_conhecido(ano)
        mudou = tamanho_atual != conhecido
        print(
            f"[Tracking] {ano} atual — "
            f"tamanho R2={tamanho_atual:,}B, conhecido={conhecido:,}B "
            f"→ {'MUDOU, processar' if mudou else 'sem mudança, pular'}"
        )
        return mudou

    def atualizar(self, ano: int, tamanho_bytes: int, ultima_data, n_registros: int):
        self._estado[str(ano)] = {
            "tamanho_bytes": tamanho_bytes,
            "ultima_data":   str(ultima_data)[:10] if ultima_data else None,
            "registros":     n_registros,
            "processado":    True,
            "atualizado_em": hora_brasilia().isoformat(),
        }
        self._salvar()


# ══════════════════════════════════════════════════════════════════════════════
# MEMÓRIA DO MODELO — residual learning entre execuções
# ══════════════════════════════════════════════════════════════════════════════

class MemoriaModelo:
    CHAVE = "safedriver/datalake/modelos/historico_erros.parquet"

    def __init__(self, s3, bucket: str):
        self.s3     = s3
        self.bucket = bucket
        self.df     = self._carregar()

    def _carregar(self) -> pd.DataFrame:
        try:
            obj  = self.s3.get_object(Bucket=self.bucket, Key=self.CHAVE)
            buf  = BytesIO(obj["Body"].read())
            df   = pd.read_parquet(buf)
            print(f"[Memória] {len(df):,} registros históricos de erro carregados.")
            return df
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                print("[Memória] Sem histórico de erros — primeira execução.")
                return pd.DataFrame()
            raise

    def salvar(self, df_erros: pd.DataFrame):
        if df_erros.empty:
            return
        if not self.df.empty:
            df_erros = pd.concat([self.df, df_erros]).drop_duplicates(
                subset=["H3_INDEX", "MES", "ANO"], keep="last"
            )
        buf = BytesIO()
        df_erros.to_parquet(buf, index=False)
        buf.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=self.CHAVE, Body=buf.read())
        print(f"[Memória] {len(df_erros):,} registros de erro salvos.")

    def enriquecer(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.df.empty:
            df["ERRO_MEDIO_HISTORICO"] = 0.0
            df["TENDENCIA_ERRO"]       = 0.0
            return df
        hist = self.df[["H3_INDEX", "ERRO_MEDIO", "TENDENCIA_ERRO"]].copy()
        df   = df.merge(hist, on="H3_INDEX", how="left", suffixes=("", "_hist"))
        df["ERRO_MEDIO_HISTORICO"] = df.get("ERRO_MEDIO_hist", pd.Series(0.0, index=df.index)).fillna(0.0)
        df["TENDENCIA_ERRO"]       = df.get("TENDENCIA_ERRO_hist", pd.Series(0.0, index=df.index)).fillna(0.0)
        return df


# ══════════════════════════════════════════════════════════════════════════════
# DISCORD — notificações
# ══════════════════════════════════════════════════════════════════════════════

class Discord:
    def __init__(self):
        self.url_sucesso = sanitizar_secret(os.environ.get("DISCORD_SUCESSO", ""))
        self.url_erro    = sanitizar_secret(os.environ.get("DISCORD_ERRO", ""))

    def _enviar(self, url: str, payload: dict):
        if not url:
            return
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"[Discord] Falha ao enviar notificação: {e}")

    def notificar_sucesso(self, run_id, tempo, n_prata, n_ouro, mae, r2, mae_ant, melhoria, status_bq):
        emoji_mae = "📈" if melhoria and melhoria > 0 else "📉"
        self._enviar(self.url_sucesso, {"content": (
            f"✅ **SafeDriver v{VERSAO_PIPELINE}** — `{run_id}`\n"
            f"⏱ Tempo: {tempo:.0f}s\n"
            f"📦 Prata: {n_prata:,} | Ouro: {n_ouro:,}\n"
            f"🎯 MAE: {mae:.2f} | R²: {r2:.4f}\n"
            f"{emoji_mae} MAE anterior: {mae_ant:.2f} | Melhoria: {melhoria:+.2f}\n"
            f"🗄️ BigQuery: {status_bq}"
        )})

    def notificar_erro(self, run_id, titulo, detalhe):
        self._enviar(self.url_erro, {"content": (
            f"🚨 **SafeDriver ERRO** — `{run_id}`\n"
            f"**{titulo}**\n```{str(detalhe)[:1500]}```"
        )})


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SafeDriver:

    def __init__(self):
        self.run_id   = gerar_run_id()
        self.t_inicio = time.time()
        self.discord  = Discord()
        self.df_raw   = pl.DataFrame()

        # Cliente S3/R2 — sanitiza secrets para evitar header corrompido
        self.bucket = sanitizar_secret(os.environ.get("R2_BUCKET_NAME", ""))
        self.s3 = boto3.client(
            "s3",
            endpoint_url          = sanitizar_secret(os.environ.get("R2_ENDPOINT_URL", "")),
            aws_access_key_id     = sanitizar_secret(os.environ.get("R2_ACCESS_KEY_ID", "")),
            aws_secret_access_key = sanitizar_secret(os.environ.get("R2_SECRET_ACCESS_KEY", "")),
            region_name           = "us-east-1",
        )

        self.tracking = TrackingSSP(self.s3, self.bucket)
        self.memoria  = MemoriaModelo(self.s3, self.bucket)

    def _log(self, evento: str, dados: dict):
        print(f"  [{self.run_id}] {evento}: {dados}")

    # ── 1. SINCRONIZAR RAW ──────────────────────────────────────────────────

    def sincronizar_raw(self):
        """
        Para cada ssp_XXXX.parquet no R2:
          1. Consulta o tamanho via head_object (sem listar, sem paginação).
          2. Compara com tracking_ssp.json.
          3. Se mudou (ou nunca processado), baixa e filtra registros novos.
          4. Anos fechados só são processados uma vez.
        """
        self._log("sincronizar_raw_inicio", {})
        frames = []

        for ano in ANOS_DISPONIVEIS:
            chave = f"{R2_PREFIXO_RAW}ssp_{ano}.parquet"

            # Passo 1: checar tamanho sem baixar o arquivo
            try:
                meta = self.s3.head_object(Bucket=self.bucket, Key=chave)
                tamanho_atual = meta["ContentLength"]
                print(f"[R2] {chave} — tamanho: {tamanho_atual:,} bytes")
            except ClientError as e:
                codigo = e.response["Error"]["Code"]
                if codigo in ("404", "NoSuchKey"):
                    print(f"[R2] {chave} não existe no bucket — pular.")
                    continue
                raise

            # Passo 2: decidir se precisa processar
            if not self.tracking.precisa_processar(ano, tamanho_atual):
                continue

            # Passo 3: baixar o arquivo
            print(f"[R2] Baixando {chave}...")
            obj = self.s3.get_object(Bucket=self.bucket, Key=chave)
            buf = BytesIO(obj["Body"].read())
            df  = pl.read_parquet(buf)
            print(f"[R2] {chave} — {len(df):,} registros brutos.")

            # Passo 4: filtro incremental para ano atual
            ultima_data = self.tracking.ultima_data_conhecida(ano)
            if ano == ANO_ATUAL and ultima_data is not None and "DATA_OCORRENCIA_BO" in df.columns:
                n_antes = len(df)
                df = df.filter(
                    pl.col("DATA_OCORRENCIA_BO").cast(pl.Utf8).str.to_datetime(
                        "%Y-%m-%d", strict=False
                    ) > pl.lit(ultima_data)
                )
                print(f"[R2] Incremental {ano}: {n_antes:,} → {len(df):,} registros novos após {ultima_data.date()}")

            if df.is_empty():
                print(f"[R2] {chave} sem registros novos — atualizar tracking sem dados.")
                self.tracking.atualizar(ano, tamanho_atual, ultima_data, 0)
                continue

            frames.append(df)

            # Passo 5: atualizar tracking
            ultima = None
            if "DATA_OCORRENCIA_BO" in df.columns:
                try:
                    ultima = pd.Timestamp(df["DATA_OCORRENCIA_BO"].max())
                except Exception:
                    ultima = None
            self.tracking.atualizar(ano, tamanho_atual, ultima, len(df))

        if not frames:
            self._log("sincronizar_raw_sem_novidades", {})
            print("[R2] Nenhum dado novo detectado em nenhum ano.")
            return

        self.df_raw = pl.concat(frames, how="diagonal_relaxed")
        self._log("sincronizar_raw_fim", {"total_registros": len(self.df_raw)})
        print(f"[R2] Total consolidado: {len(self.df_raw):,} registros")

    # ── 2. CONSTRUIR PRATA ──────────────────────────────────────────────────

    def _resolver_colunas(self, df: pd.DataFrame) -> pd.DataFrame:
        colunas_upper = {c.upper().strip(): c for c in df.columns}
        mapa = {}
        for nome_padrao, sinonimos in SINONIMOS.items():
            if nome_padrao in colunas_upper:
                mapa[colunas_upper[nome_padrao]] = nome_padrao
            else:
                for s in sinonimos:
                    if s in colunas_upper:
                        mapa[colunas_upper[s]] = nome_padrao
                        break
        return df.rename(columns=mapa)

    def construir_prata(self) -> pl.DataFrame:
        if self.df_raw.is_empty():
            print("[Prata] df_raw vazio — nada a processar.")
            return pl.DataFrame()

        self._log("prata_inicio", {"registros": len(self.df_raw)})
        df = self.df_raw.to_pandas()
        df = self._resolver_colunas(df)

        # Tipos numéricos
        for col in ["LATITUDE", "LONGITUDE"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Descartar coordenadas inválidas — conforme documento SSP página 4
        # Sem geocodificação: coordenada inválida = registro descartado
        n_antes = len(df)
        df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
        df = df[(df["LATITUDE"] != 0.0) & (df["LONGITUDE"] != 0.0)]
        df = df[
            df["LATITUDE"].between(SP_LAT_MIN, SP_LAT_MAX) &
            df["LONGITUDE"].between(SP_LON_MIN, SP_LON_MAX)
        ]
        self._log("prata_coords_descartadas", {
            "descartados": n_antes - len(df), "mantidos": len(df)
        })

        if df.empty:
            print("[Prata] Nenhum registro com coordenada válida.")
            return pl.DataFrame()

        # Normalizar texto
        for col in ["NOME_DEPARTAMENTO", "NOME_MUNICIPIO", "LOGRADOURO",
                    "BAIRRO", "RUBRICA", "DESCR_CONDUTA", "NATUREZA_APURADA"]:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: normalizar_texto(x) if pd.notna(x) else x)

        # Datas
        for col in ["DATA_OCORRENCIA_BO", "DATA_REGISTRO"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)

        # Features temporais
        if "DATA_OCORRENCIA_BO" in df.columns:
            feriados_sp = holidays.Brazil(state="SP")
            df["ANO"]        = df["DATA_OCORRENCIA_BO"].dt.year
            df["MES"]        = df["DATA_OCORRENCIA_BO"].dt.month
            df["DIA_SEMANA"] = df["DATA_OCORRENCIA_BO"].dt.dayofweek
            df["FERIADO"]    = df["DATA_OCORRENCIA_BO"].dt.date.apply(
                lambda d: 1 if d in feriados_sp else 0
            )

        # Hora
        if "HORA_OCORRENCIA_BO" in df.columns:
            df["HORA"] = pd.to_datetime(
                df["HORA_OCORRENCIA_BO"], format="%H:%M", errors="coerce"
            ).dt.hour
            df["TURNO"] = pd.cut(
                df["HORA"],
                bins=[-1, 5, 11, 17, 23],
                labels=["MADRUGADA", "MANHA", "TARDE", "NOITE"]
            ).astype(str)

        # H3
        def safe_h3(lat, lon):
            try:
                return h3.latlng_to_cell(lat, lon, H3_RESOLUCAO)
            except Exception:
                return None

        df["H3_INDEX"] = df.apply(lambda r: safe_h3(r["LATITUDE"], r["LONGITUDE"]), axis=1)
        df = df.dropna(subset=["H3_INDEX"])

        result = pl.from_pandas(df)
        self._log("prata_fim", {"registros": len(result)})
        return result

    # ── 3. CONSTRUIR OURO ───────────────────────────────────────────────────

    def _peso_crime(self, rubrica: str, perfil: str) -> float:
        if not isinstance(rubrica, str):
            return 1.0
        base = PESO_PENAL_BASE.get(rubrica, 1.0)
        mult = MULTIPLICADOR_PERFIL.get(perfil, {}).get(rubrica, 1.0)
        return base * mult

    def _agregar_por_perfil(self, df: pd.DataFrame, perfil: str) -> pd.DataFrame:
        if "RUBRICA" in df.columns:
            df = df.copy()
            df["PESO"] = df["RUBRICA"].apply(lambda r: self._peso_crime(r, perfil))
        else:
            df = df.copy()
            df["PESO"] = 1.0

        grp = df.groupby(["H3_INDEX", "ANO", "MES"]).agg(
            QTD_CRIMES    = ("PESO", "count"),
            ESCORE_BRUTO  = ("PESO", "sum"),
            DIA_SEMANA    = ("DIA_SEMANA", "mean"),
            FERIADO       = ("FERIADO",    "mean"),
            TURNO_NOITE   = ("TURNO",      lambda x: (x == "NOITE").mean()),
        ).reset_index()
        grp["PERFIL"] = perfil
        return grp

    def _features_vizinhanca(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona média de crimes dos hexágonos vizinhos — resolve faixa 21-50 crimes."""
        escore_map = df.groupby("H3_INDEX")["ESCORE_BRUTO"].mean().to_dict()

        def media_vizinhos(idx):
            try:
                vizinhos = h3.grid_disk(idx, 1) - {idx}
                valores  = [escore_map.get(v, 0.0) for v in vizinhos]
                return np.mean(valores) if valores else 0.0
            except Exception:
                return 0.0

        df["MEDIA_VIZINHOS"] = df["H3_INDEX"].apply(media_vizinhos)
        return df

    def _lags_temporais(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["H3_INDEX", "PERFIL", "ANO", "MES"])
        df["LAG1"] = df.groupby(["H3_INDEX", "PERFIL"])["ESCORE_BRUTO"].shift(1).fillna(0)
        df["LAG2"] = df.groupby(["H3_INDEX", "PERFIL"])["ESCORE_BRUTO"].shift(2).fillna(0)
        df["MM3"]  = (
            df.groupby(["H3_INDEX", "PERFIL"])["ESCORE_BRUTO"]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
            .fillna(0)
        )
        return df

    def _treinar_modelo(self, df_treino: pd.DataFrame, features: list):
        lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05,
                             num_leaves=63, random_state=42, verbose=-1)
        cat  = CatBoostRegressor(iterations=500, learning_rate=0.05,
                                 depth=6, random_state=42, verbose=0)
        ensemble = VotingRegressor([("lgbm", lgbm), ("cat", cat)])
        tscv  = TimeSeriesSplit(n_splits=5)
        maes  = []
        r2s   = []
        for tr, va in tscv.split(df_treino):
            X_tr = df_treino.iloc[tr][features]
            y_tr = df_treino.iloc[tr]["ESCORE_BRUTO"]
            X_va = df_treino.iloc[va][features]
            y_va = df_treino.iloc[va]["ESCORE_BRUTO"]
            ensemble.fit(X_tr, y_tr)
            pred = ensemble.predict(X_va)
            maes.append(mean_absolute_error(y_va, pred))
            r2s.append(r2_score(y_va, pred))
        ensemble.fit(df_treino[features], df_treino["ESCORE_BRUTO"])
        return ensemble, float(np.mean(maes)), float(np.mean(r2s))

    def _mae_anterior(self) -> float:
        try:
            obj = self.s3.get_object(
                Bucket=self.bucket,
                Key=f"{R2_PREFIXO_MODELO}modelo_atual.json"
            )
            meta = json.loads(obj["Body"].read())
            return meta.get("mae", 999.0)
        except Exception:
            return 999.0

    def construir_ouro(self, df_prata: pl.DataFrame, bq_project, bq_dataset, bq_cred):
        if df_prata.is_empty():
            print("[Ouro] prata vazia — abortando.")
            return None

        self._log("ouro_inicio", {"registros": len(df_prata)})
        df = df_prata.to_pandas()

        # Agregar por perfil
        frames_perfil = []
        for perfil in MULTIPLICADOR_PERFIL:
            frames_perfil.append(self._agregar_por_perfil(df, perfil))
        df_agg = pd.concat(frames_perfil, ignore_index=True)

        # Features de vizinhança e lags
        df_agg = self._features_vizinhanca(df_agg)
        df_agg = self._lags_temporais(df_agg)
        df_agg = self.memoria.enriquecer(df_agg)

        # Normalizar escore 0-100
        mx = df_agg["ESCORE_BRUTO"].max()
        df_agg["ESCORE_NORMALIZADO"] = (df_agg["ESCORE_BRUTO"] / mx * 100).clip(0, 100) if mx > 0 else 0.0

        if len(df_agg) < MIN_REGISTROS:
            print(f"[Ouro] Registros insuficientes: {len(df_agg)} < {MIN_REGISTROS}")
            return None

        features = [
            "QTD_CRIMES", "DIA_SEMANA", "FERIADO", "TURNO_NOITE",
            "MES", "LAG1", "LAG2", "MM3",
            "MEDIA_VIZINHOS", "ERRO_MEDIO_HISTORICO", "TENDENCIA_ERRO"
        ]
        features = [f for f in features if f in df_agg.columns]

        mae_ant  = self._mae_anterior()
        modelo, mae, r2 = self._treinar_modelo(df_agg, features)
        melhoria = mae_ant - mae

        # SHAP
        shap_df = pd.DataFrame()
        try:
            explainer = shap.TreeExplainer(modelo.estimators_[0])
            sv        = explainer.shap_values(df_agg[features].head(2000))
            shap_df   = pd.DataFrame(sv, columns=features)
            shap_df["H3_INDEX"] = df_agg["H3_INDEX"].head(2000).values
            shap_df["RUN_ID"]   = self.run_id
        except Exception as e:
            print(f"[SHAP] Erro: {e}")

        # Salvar modelo no R2
        modelo_key = f"{R2_PREFIXO_MODELO}{self.run_id}_modelo.pkl"
        buf = BytesIO()
        joblib.dump(modelo, buf)
        buf.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=modelo_key, Body=buf.read())

        meta_modelo = {
            "run_id": self.run_id, "mae": mae, "r2": r2,
            "versao": VERSAO_PIPELINE, "features": VERSAO_FEATURES,
            "modelo_key": modelo_key,
            "timestamp": hora_brasilia().isoformat()
        }
        self.s3.put_object(
            Bucket=self.bucket,
            Key=f"{R2_PREFIXO_MODELO}modelo_atual.json",
            Body=json.dumps(meta_modelo).encode()
        )

        # Memória de erros
        pred = modelo.predict(df_agg[features])
        df_agg["ERRO"] = np.abs(pred - df_agg["ESCORE_BRUTO"])
        erros = df_agg.groupby("H3_INDEX").agg(
            ERRO_MEDIO=("ERRO", "mean"),
            TENDENCIA_ERRO=("ERRO", lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0)
        ).reset_index()
        self.memoria.salvar(erros)

        # Salvar ouro no R2
        ouro_key = f"{R2_PREFIXO_OURO}{self.run_id}_ouro.parquet"
        buf = BytesIO()
        df_agg.to_parquet(buf, index=False)
        buf.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=ouro_key, Body=buf.read())

        # BigQuery
        status_bq = "❌ não enviado"
        if bq_project and bq_dataset and bq_cred:
            try:
                info   = json.loads(bq_cred)
                creds  = service_account.Credentials.from_service_account_info(info)
                client = bigquery.Client(project=bq_project, credentials=creds)

                tabela_ouro = f"{bq_project}.{bq_dataset}.fato_risco_hexagono"
                job = client.load_table_from_dataframe(
                    df_agg, tabela_ouro,
                    job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
                )
                job.result()

                if not shap_df.empty:
                    tabela_shap = f"{bq_project}.{bq_dataset}.auditoria_shap"
                    job2 = client.load_table_from_dataframe(
                        shap_df, tabela_shap,
                        job_config=bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
                    )
                    job2.result()

                status_bq = "✅ enviado"
                print(f"[BQ] Dados enviados: {tabela_ouro}")
            except Exception as e:
                status_bq = f"❌ erro: {str(e)[:100]}"
                print(f"[BQ] Erro: {e}")

        self._log("ouro_fim", {
            "registros": len(df_agg), "mae": round(mae, 4),
            "r2": round(r2, 4), "melhoria_mae": round(melhoria, 4)
        })
        return df_agg, mae, r2, mae_ant, melhoria, status_bq

    # ── PROCESSAR ───────────────────────────────────────────────────────────

    def processar(self):
        self._log("pipeline_iniciado", {"versao": VERSAO_PIPELINE, "features": VERSAO_FEATURES})

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
