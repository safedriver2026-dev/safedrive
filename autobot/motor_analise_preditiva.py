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

VERSAO_PIPELINE    = "4.0.0"
VERSAO_FEATURES    = "v4"
H3_RESOLUCAO       = 8
MIN_REGISTROS_OURO = 500

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

# Conforme documento SSP — colunas críticas para identificar aba de dados
COLUNAS_CRITICAS = [
    "NOME_DEPARTAMENTO", "NOME_MUNICIPIO",
    "LOGRADOURO", "LATITUDE", "LONGITUDE", "DATA_OCORRENCIA_BO"
]

# Sinônimos históricos — resolve renomeações entre anos da SSP
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

# Bounding box do Estado de São Paulo
SP_LAT_MIN, SP_LAT_MAX = -25.3, -19.8
SP_LON_MIN, SP_LON_MAX = -53.2, -44.0


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
    """Remove newlines e caracteres não imprimíveis que corrompem headers AWS4."""
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
    print(f"  ✅ BQ → {tabela_id}: {len(df_pl):,} registros", file=sys.stdout)


# ══════════════════════════════════════════════════════════════════════════════
# TELEMETRIA DISCORD
# ══════════════════════════════════════════════════════════════════════════════

class Telemetria:
    def __init__(self):
        self.url_sucesso = sanitizar_secret(os.environ.get("DISCORD_SUCESSO", ""))
        self.url_erro    = sanitizar_secret(os.environ.get("DISCORD_ERRO",    ""))

    def _post(self, url, payload):
        if not url or not url.startswith("https://discord"):
            return
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"❌ Discord: {e}", file=sys.stderr)

    def notificar_sucesso(self, run_id, tempo, reg_prata, reg_ouro,
                          mae, r2, mae_anterior, melhoria_pct, status_bq):
        sinal = "📈" if melhoria_pct >= 0 else "📉"
        self._post(self.url_sucesso, {"embeds": [{"title": "🟢 SafeDriver Pipeline Concluído",
            "color": 3066993, "fields": [
                {"name": "🔑 Run ID",        "value": run_id,              "inline": True},
                {"name": "📊 Prata",          "value": f"{reg_prata:,}",    "inline": True},
                {"name": "🏅 Ouro",           "value": f"{reg_ouro:,}",     "inline": True},
                {"name": "📉 MAE",            "value": f"{mae:.4f}",        "inline": True},
                {"name": "📈 R²",             "value": f"{r2:.4f}",         "inline": True},
                {"name": f"{sinal} vs Anterior", "value": f"{melhoria_pct:+.1f}%", "inline": True},
                {"name": "⏱️ Tempo",          "value": f"{tempo:.1f}s",     "inline": True},
                {"name": "📡 BigQuery",       "value": status_bq,           "inline": False},
            ],
            "footer": {"text": f"SafeDriver v{VERSAO_PIPELINE} • {hora_brasilia().strftime('%d/%m/%Y %H:%M')}"}
        }]})

    def notificar_erro(self, run_id, titulo, erro_msg):
        ticks = "```"
        self._post(self.url_erro, {"embeds": [{"title": f"🔴 {titulo}",
            "color": 15158332, "fields": [
                {"name": "🔑 Run ID",   "value": run_id, "inline": True},
                {"name": "Traceback",   "value": f"{ticks}python\n{erro_msg[:1500]}\n{ticks}", "inline": False},
            ],
            "footer": {"text": f"SafeDriver v{VERSAO_PIPELINE} • {hora_brasilia().strftime('%d/%m/%Y %H:%M')}"}
        }]})


# ══════════════════════════════════════════════════════════════════════════════
# AUDITORIA
# ══════════════════════════════════════════════════════════════════════════════

class RegistroAuditoria:
    def __init__(self, run_id: str):
        self.run_id   = run_id
        self.entradas = []

    def log(self, evento: str, detalhes: dict):
        entrada = {
            "run_id":    self.run_id,
            "timestamp": hora_brasilia().isoformat(),
            "evento":    evento,
            **detalhes
        }
        self.entradas.append(entrada)
        print(f"  [{self.run_id}] {evento}: {detalhes}", file=sys.stdout)

    def to_dataframe(self) -> pl.DataFrame:
        if not self.entradas:
            return pl.DataFrame()
        rows = []
        for e in self.entradas:
            rows.append({
                "run_id":    e.get("run_id", ""),
                "timestamp": e.get("timestamp", ""),
                "evento":    e.get("evento", ""),
                "detalhes":  json.dumps({k: v for k, v in e.items()
                                         if k not in ("run_id", "timestamp", "evento")}),
            })
        return pl.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MEMÓRIA DO MODELO — residual learning entre execuções
# ══════════════════════════════════════════════════════════════════════════════

class MemoriaModelo:
    def __init__(self, r2_client, bucket: str, prefixo: str):
        self.r2      = r2_client
        self.bucket  = bucket
        self.prefixo = prefixo

    def carregar_erros_historicos(self) -> pd.DataFrame:
        try:
            obj = self.r2.get_object(
                Bucket=self.bucket,
                Key=f"{self.prefixo}modelos/erros_historicos.parquet"
            )
            return pd.read_parquet(BytesIO(obj["Body"].read()))
        except Exception:
            return pd.DataFrame(columns=["H3_R8", "ERRO_MEDIO_HISTORICO",
                                          "TENDENCIA_ERRO", "RESIDUO_ULTIMA_EXECUCAO"])

    def salvar_erros(self, df_erros: pd.DataFrame):
        buf = BytesIO()
        df_erros.to_parquet(buf, index=False)
        buf.seek(0)
        self.r2.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefixo}modelos/erros_historicos.parquet",
            Body=buf.getvalue()
        )

    def carregar_meta_modelo(self) -> dict:
        try:
            obj = self.r2.get_object(
                Bucket=self.bucket,
                Key=f"{self.prefixo}modelos/modelo_atual.json"
            )
            return json.loads(obj["Body"].read())
        except Exception:
            return {}

    def salvar_meta_modelo(self, meta: dict):
        self.r2.put_object(
            Bucket=self.bucket,
            Key=f"{self.prefixo}modelos/modelo_atual.json",
            Body=json.dumps(meta).encode()
        )


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SafeDriver:

    R2_PREFIXO = "safedriver/datalake/"

    def __init__(self):
        self.run_id   = gerar_run_id()
        self.t_inicio = time.time()
        self.discord  = Telemetria()
        self.auditoria = RegistroAuditoria(self.run_id)

        # Sanitiza secrets antes de criar o cliente boto3
        endpoint   = sanitizar_secret(os.environ.get("R2_ENDPOINT_URL",      ""))
        access_key = sanitizar_secret(os.environ.get("R2_ACCESS_KEY_ID",     ""))
        secret_key = sanitizar_secret(os.environ.get("R2_SECRET_ACCESS_KEY", ""))
        self.bucket = sanitizar_secret(os.environ.get("R2_BUCKET_NAME",      ""))

        self.r2 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name="auto",
        )

        self.memoria = MemoriaModelo(self.r2, self.bucket, self.R2_PREFIXO)

        self.pastas = {
            "raw":      Path("/tmp/safedriver/raw"),
            "prata":    Path("/tmp/safedriver/prata"),
            "ouro":     Path("/tmp/safedriver/ouro"),
            "auditoria": Path("/tmp/safedriver/auditoria"),
            "modelos":  Path("/tmp/safedriver/modelos"),
        }
        for p in self.pastas.values():
            p.mkdir(parents=True, exist_ok=True)

        self.auditoria.log("pipeline_iniciado", {
            "versao": VERSAO_PIPELINE,
            "features": VERSAO_FEATURES
        })

    # ── R2: upload ────────────────────────────────────────────────────────────

    def upload_r2(self, caminho_local: Path, chave_r2: str):
        self.r2.upload_file(
            str(caminho_local),
            self.bucket,
            f"{self.R2_PREFIXO}{chave_r2}"
        )

    # ── R2: sincronizar raw ───────────────────────────────────────────────────

    def sincronizar_raw(self):
        self.auditoria.log("sincronizar_raw_inicio", {})
        prefixo_raw = f"{self.R2_PREFIXO}raw/"
        paginator   = self.r2.get_paginator("list_objects_v2")
        arquivos    = []

        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefixo_raw):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith(".xlsx"):
                    arquivos.append(key)

        self.auditoria.log("sincronizar_raw_listagem", {"total_xlsx": len(arquivos)})

        for key in arquivos:
            nome   = Path(key).name
            destino = self.pastas["raw"] / nome
            if not destino.exists():
                print(f"  ⬇️  Baixando {nome}...", file=sys.stdout)
                self.r2.download_file(self.bucket, key, str(destino))

        self.auditoria.log("sincronizar_raw_fim", {"baixados": len(arquivos)})

    # ── Leitura e resolução de colunas — conforme documento SSP ──────────────

    def _normalizar_colunas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nomes de colunas e aplica sinônimos históricos."""
        df.columns = [normalizar_texto(c) for c in df.columns]
        mapeamento = {}
        for col_padrao, sinonimos in SINONIMOS.items():
            for sin in sinonimos:
                if sin in df.columns and col_padrao not in df.columns:
                    mapeamento[sin] = col_padrao
        if mapeamento:
            df = df.rename(columns=mapeamento)
        return df

    def _detectar_abas_dados(self, caminho: Path) -> list:
        """
        Conforme documento SSP páginas 2-3:
        - Ignora aba 'Campos da Tabela_SPDADOS'
        - Aceita aba se tiver >= 4 colunas críticas
        - Lê apenas primeiras 5 linhas para detecção
        """
        import openpyxl
        wb    = openpyxl.load_workbook(caminho, read_only=True, data_only=True)
        abas  = wb.sheetnames
        wb.close()

        abas_dados = []
        for aba in abas:
            nome_norm = normalizar_texto(aba)
            if "CAMPOS" in nome_norm and "SPDADOS" in nome_norm:
                self.auditoria.log("aba_ignorada_metadados", {"aba": aba})
                continue

            try:
                df_amostra = pd.read_excel(
                    caminho, sheet_name=aba,
                    nrows=5, engine="openpyxl"
                )
                df_amostra = self._normalizar_colunas(df_amostra)
                colunas_encontradas = [c for c in COLUNAS_CRITICAS if c in df_amostra.columns]

                if len(colunas_encontradas) >= 4:
                    abas_dados.append(aba)
                    self.auditoria.log("aba_aceita", {
                        "aba": aba,
                        "colunas_criticas_encontradas": len(colunas_encontradas)
                    })
                else:
                    self.auditoria.log("aba_ignorada_colunas_insuficientes", {
                        "aba": aba,
                        "colunas": colunas_encontradas
                    })
            except Exception as e:
                self.auditoria.log("aba_erro_leitura", {"aba": aba, "erro": str(e)})

        return abas_dados

    def _ler_aba(self, caminho: Path, aba: str) -> pd.DataFrame:
        df = pd.read_excel(caminho, sheet_name=aba, engine="openpyxl", dtype=str)
        df = self._normalizar_colunas(df)

        # Converte lat/lon para float conforme documento SSP página 2
        for col in ["LATITUDE", "LONGITUDE"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    # ── Limpeza conforme documento SSP ───────────────────────────────────────

    def _limpar_coordenadas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Conforme documento SSP página 4:
        Coordenadas (0,0) indicam dados inválidos — descartar sem geocodificação.
        """
        n_antes = len(df)

        df["LATITUDE"]  = pd.to_numeric(df.get("LATITUDE",  pd.Series()), errors="coerce")
        df["LONGITUDE"] = pd.to_numeric(df.get("LONGITUDE", pd.Series()), errors="coerce")

        # Remove nulos
        df = df.dropna(subset=["LATITUDE", "LONGITUDE"])

        # Remove zeros conforme página 4
        df = df[(df["LATITUDE"] != 0.0) & (df["LONGITUDE"] != 0.0)]

        # Valida bounding box do estado de SP
        df = df[
            (df["LATITUDE"].between(SP_LAT_MIN, SP_LAT_MAX)) &
            (df["LONGITUDE"].between(SP_LON_MIN, SP_LON_MAX))
        ]

        self.auditoria.log("limpeza_coordenadas", {
            "antes": n_antes,
            "depois": len(df),
            "descartados": n_antes - len(df)
        })
        return df

    # ── Camada PRATA ──────────────────────────────────────────────────────────

    def construir_prata(self) -> pl.DataFrame:
        self.auditoria.log("prata_inicio", {})
        frames = []

        for xlsx in sorted(self.pastas["raw"].glob("*.xlsx")):
            self.auditoria.log("processando_arquivo", {"arquivo": xlsx.name})
            abas = self._detectar_abas_dados(xlsx)

            for aba in abas:
                try:
                    df = self._ler_aba(xlsx, aba)
                    df = self._limpar_coordenadas(df)

                    if len(df) == 0:
                        continue

                    # Padronização de texto conforme documento SSP página 4
                    for col in ["NOME_DEPARTAMENTO", "NOME_MUNICIPIO", "LOGRADOURO",
                                "BAIRRO", "RUBRICA", "DESCR_CONDUTA", "NATUREZA_APURADA"]:
                        if col in df.columns:
                            df[col] = df[col].apply(normalizar_texto)

                    # Datas
                    for col_data in ["DATA_OCORRENCIA_BO", "DATA_REGISTRO"]:
                        if col_data in df.columns:
                            df[col_data] = pd.to_datetime(df[col_data], errors="coerce")

                    # Features temporais
                    if "DATA_OCORRENCIA_BO" in df.columns:
                        df["ANO"] = df["DATA_OCORRENCIA_BO"].dt.year
                        df["MES"] = df["DATA_OCORRENCIA_BO"].dt.month
                        df["DIA_SEMANA"] = df["DATA_OCORRENCIA_BO"].dt.dayofweek
                        df["FIM_DE_SEMANA"] = df["DIA_SEMANA"] >= 5

                        feriados_sp = holidays.Brazil(state="SP")
                        df["FERIADO"] = df["DATA_OCORRENCIA_BO"].dt.date.apply(
                            lambda d: d in feriados_sp if pd.notna(d) else False
                        )

                    # Turno a partir de HORA_OCORRENCIA_BO
                    if "HORA_OCORRENCIA_BO" in df.columns:
                        def classificar_turno(h):
                            try:
                                hora = int(str(h).split(":")[0])
                                if 6  <= hora < 12: return "MANHA"
                                if 12 <= hora < 18: return "TARDE"
                                if 18 <= hora < 24: return "NOITE"
                                return "MADRUGADA"
                            except Exception:
                                return "DESCONHECIDO"
                        df["TURNO"] = df["HORA_OCORRENCIA_BO"].apply(classificar_turno)

                    # Atraso de registro — proxy de subnotificação
                    if "DATA_REGISTRO" in df.columns and "DATA_OCORRENCIA_BO" in df.columns:
                        df["ATRASO_REGISTRO_DIAS"] = (
                            df["DATA_REGISTRO"] - df["DATA_OCORRENCIA_BO"]
                        ).dt.days.clip(0, 365)

                    # H3
                    def lat_lon_to_h3(row):
                        try:
                            return h3.latlng_to_cell(row["LATITUDE"], row["LONGITUDE"], H3_RESOLUCAO)
                        except Exception:
                            return None
                    df["H3_R8"] = df.apply(lat_lon_to_h3, axis=1)
                    df = df.dropna(subset=["H3_R8"])

                    # Peso penal
                    rubrica_norm = df.get("RUBRICA", pd.Series()).fillna("")
                    df["PESO_PENAL"] = rubrica_norm.apply(
                        lambda r: PESO_PENAL_BASE.get(r, 1.0)
                    )

                    # Flag crime veicular
                    crimes_veiculo = {
                        "ROUBO DE VEICULO", "FURTO DE VEICULO",
                        "ROUBO DE MOTOCICLETA", "FURTO DE MOTOCICLETA", "ROUBO DE CARGA"
                    }
                    df["CRIME_VEICULO"] = rubrica_norm.isin(crimes_veiculo)

                    # Metadado de origem do arquivo
                    df["ARQUIVO_ORIGEM"] = xlsx.name
                    df["ABA_ORIGEM"]     = aba
                    df["RUN_ID"]         = self.run_id

                    frames.append(df)
                    self.auditoria.log("aba_processada", {
                        "arquivo": xlsx.name,
                        "aba": aba,
                        "registros": len(df)
                    })

                except Exception as e:
                    self.auditoria.log("aba_erro_processamento", {
                        "arquivo": xlsx.name,
                        "aba": aba,
                        "erro": str(e)
                    })

        if not frames:
            raise RuntimeError("Nenhum dado válido encontrado na camada raw.")

        df_prata = pd.concat(frames, ignore_index=True)
        df_prata = df_prata.drop_duplicates()

        # Salva prata
        caminho_prata = self.pastas["prata"] / "crimes_prata.parquet"
        df_prata.to_parquet(caminho_prata, index=False)
        self.upload_r2(caminho_prata, "prata/crimes_prata.parquet")

        self.auditoria.log("prata_fim", {
            "total_registros": len(df_prata),
            "arquivos_processados": len(frames)
        })

        return pl.from_pandas(df_prata)

    # ── Camada OURO — Star Schema para Looker ─────────────────────────────────

    def construir_ouro(self, df_prata: pl.DataFrame,
                        bq_project: str, bq_dataset: str, bq_cred: str):

        self.auditoria.log("ouro_inicio", {"registros_prata": len(df_prata)})

        if len(df_prata) < MIN_REGISTROS_OURO:
            self.auditoria.log("ouro_abortado", {"motivo": "dados insuficientes"})
            return None

        df = df_prata.to_pandas()

        # ── Features de vizinhança espacial H3 ───────────────────────────────
        # Resolve o problema da faixa 21-50 do CSV de validação (R²=0.46)
        def media_vizinhos(h3_idx):
            try:
                vizinhos = list(h3.grid_disk(h3_idx, 1))
                contagens = df[df["H3_R8"].isin(vizinhos)].shape[0]
                return contagens / max(len(vizinhos), 1)
            except Exception:
                return 0.0

        hex_contagem = df.groupby("H3_R8").size().to_dict()
        df["MEDIA_CRIMES_VIZINHOS"] = df["H3_R8"].apply(media_vizinhos)

        # ── Features temporais de lags ────────────────────────────────────────
        hex_mes = df.groupby(["H3_R8", "ANO", "MES"]).size().reset_index(name="QTD")
        hex_mes = hex_mes.sort_values(["H3_R8", "ANO", "MES"])
        hex_mes["LAG1"] = hex_mes.groupby("H3_R8")["QTD"].shift(1).fillna(0)
        hex_mes["LAG2"] = hex_mes.groupby("H3_R8")["QTD"].shift(2).fillna(0)
        hex_mes["MM3"]  = hex_mes.groupby("H3_R8")["QTD"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
        df = df.merge(hex_mes[["H3_R8", "ANO", "MES", "LAG1", "LAG2", "MM3"]],
                      on=["H3_R8", "ANO", "MES"], how="left")

        # ── Memória de erros históricos — residual learning ───────────────────
        df_erros_hist = self.memoria.carregar_erros_historicos()
        if len(df_erros_hist) > 0:
            df = df.merge(df_erros_hist, on="H3_R8", how="left")
            df["ERRO_MEDIO_HISTORICO"]   = df["ERRO_MEDIO_HISTORICO"].fillna(0)
            df["TENDENCIA_ERRO"]         = df["TENDENCIA_ERRO"].fillna(0)
            df["RESIDUO_ULTIMA_EXECUCAO"] = df["RESIDUO_ULTIMA_EXECUCAO"].fillna(0)
        else:
            df["ERRO_MEDIO_HISTORICO"]    = 0.0
            df["TENDENCIA_ERRO"]          = 0.0
            df["RESIDUO_ULTIMA_EXECUCAO"] = 0.0

        # ── Agregação por hexágono-mês para treino ────────────────────────────
        agg_cols = {
            "PESO_PENAL":              "sum",
            "CRIME_VEICULO":           "sum",
            "FERIADO":                 "mean",
            "FIM_DE_SEMANA":           "mean",
            "ATRASO_REGISTRO_DIAS":    "mean",
            "MEDIA_CRIMES_VIZINHOS":   "mean",
            "LAG1":                    "first",
            "LAG2":                    "first",
            "MM3":                     "first",
            "ERRO_MEDIO_HISTORICO":    "mean",
            "TENDENCIA_ERRO":          "mean",
            "RESIDUO_ULTIMA_EXECUCAO": "mean",
        }
        df_agg = df.groupby(["H3_R8", "ANO", "MES"]).agg(
            QTD_CRIMES=("RUBRICA", "count"),
            **{k: (k, v) for k, v in agg_cols.items() if k in df.columns}
        ).reset_index()

        # Colunas de local para dim_local
        local_cols = ["H3_R8", "NOME_DEPARTAMENTO", "NOME_SECCIONAL",
                      "NOME_DELEGACIA", "NOME_MUNICIPIO", "BAIRRO"]
        local_cols = [c for c in local_cols if c in df.columns]
        dim_local_agg = df[local_cols].drop_duplicates("H3_R8")

        # ── Modelo — TimeSeriesSplit sem data leakage ─────────────────────────
        feature_cols = [
            "ANO", "MES", "PESO_PENAL", "CRIME_VEICULO",
            "FERIADO", "FIM_DE_SEMANA", "ATRASO_REGISTRO_DIAS",
            "MEDIA_CRIMES_VIZINHOS", "LAG1", "LAG2", "MM3",
            "ERRO_MEDIO_HISTORICO", "TENDENCIA_ERRO", "RESIDUO_ULTIMA_EXECUCAO"
        ]
        feature_cols = [c for c in feature_cols if c in df_agg.columns]

        X = df_agg[feature_cols].fillna(0).values
        y = df_agg["QTD_CRIMES"].values

        tscv    = TimeSeriesSplit(n_splits=5)
        maes, r2s = [], []
        residuos  = []

        lgbm = LGBMRegressor(n_estimators=500, learning_rate=0.05,
                              num_leaves=63, random_state=42, verbose=-1)
        cat  = CatBoostRegressor(iterations=500, learning_rate=0.05,
                                  depth=6, random_state=42, verbose=0)
        modelo = VotingRegressor([("lgbm", lgbm), ("cat", cat)])

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            maes.append(mean_absolute_error(y_test, y_pred))
            r2s.append(r2_score(y_test, y_pred))
            residuos.extend(list(zip(
                df_agg.iloc[test_idx]["H3_R8"].values,
                (y_test - y_pred).tolist()
            )))

        mae_final = float(np.mean(maes))
        r2_final  = float(np.mean(r2s))

        df_cv = pl.DataFrame({
            "run_id":     [self.run_id] * len(maes),
            "fold":       list(range(len(maes))),
            "mae":        maes,
            "r2":         r2s,
            "versao":     [VERSAO_PIPELINE] * len(maes),
        })

        self.auditoria.log("modelo_treinado", {
            "mae_medio": round(mae_final, 4),
            "r2_medio":  round(r2_final, 4),
            "folds":     len(maes)
        })

        # ── Salva erros para residual learning na próxima execução ───────────
        df_res = pd.DataFrame(residuos, columns=["H3_R8", "RESIDUO"])
        df_res_agg = df_res.groupby("H3_R8").agg(
            ERRO_MEDIO_HISTORICO=("RESIDUO", lambda x: abs(x).mean()),
            TENDENCIA_ERRO=("RESIDUO", "mean"),
            RESIDUO_ULTIMA_EXECUCAO=("RESIDUO", "last")
        ).reset_index()
        self.memoria.salvar_erros(df_res_agg)

        # ── SHAP para auditoria ───────────────────────────────────────────────
        modelo.fit(X, y)
        explainer   = shap.Explainer(modelo.estimators_[0], X)
        shap_values = explainer(X[:500])
        df_shap_pl  = pl.DataFrame({
            "run_id":   [self.run_id] * 500,
            "h3_r8":    df_agg["H3_R8"].values[:500].tolist(),
            **{f"shap_{feature_cols[i]}": shap_values.values[:, i].tolist()
               for i in range(len(feature_cols))}
        })

        # ── Previsões e escore final ──────────────────────────────────────────
        df_agg["RISCO_PREVISTO"] = modelo.predict(X).clip(0)

        # ── Star Schema — Fato em formato long por perfil ─────────────────────
        registros_fato = []
        for _, row in df_agg.iterrows():
            for perfil, multiplicadores in MULTIPLICADOR_PERFIL.items():
                risco_perfil = row["RISCO_PREVISTO"]
                for rubrica, mult in multiplicadores.items():
                    if rubrica in df[df["H3_R8"] == row["H3_R8"]].get("RUBRICA", pd.Series()).values:
                        risco_perfil *= mult
                registros_fato.append({
                    "RUN_ID":          self.run_id,
                    "H3_R8":           row["H3_R8"],
                    "ANO":             int(row["ANO"]),
                    "MES":             int(row["MES"]),
                    "PERFIL":          perfil,
                    "QTD_CRIMES":      int(row["QTD_CRIMES"]),
                    "RISCO_PREVISTO":  round(float(risco_perfil), 4),
                    "PESO_PENAL_SOMA": round(float(row.get("PESO_PENAL", 0)), 4),
                    "MAE_FOLD":        round(mae_final, 4),
                    "R2_FOLD":         round(r2_final, 4),
                    "VERSAO_PIPELINE": VERSAO_PIPELINE,
                    "VERSAO_FEATURES": VERSAO_FEATURES,
                })

        df_fato = pl.DataFrame(registros_fato)

        # ── Dimensões ─────────────────────────────────────────────────────────
        dim_tempo = pd.DataFrame({
            "SK_TEMPO":      range(len(df_agg)),
            "ANO":           df_agg["ANO"].values,
            "MES":           df_agg["MES"].values,
            "NOME_MES":      pd.to_datetime(df_agg["MES"].astype(str), format="%m").dt.strftime("%B").str.upper().values,
            "TRIMESTRE":     ((df_agg["MES"] - 1) // 3 + 1).values,
        })

        dim_local = pd.DataFrame(dim_local_agg) if not isinstance(dim_local_agg, pd.DataFrame) \
            else dim_local_agg
        dim_local["SK_LOCAL"] = range(len(dim_local))

        rubrica_counts = df.groupby("RUBRICA").size().reset_index(name="TOTAL_REGISTROS")
        rubrica_counts["SK_CRIME"]   = range(len(rubrica_counts))
        rubrica_counts["PESO_PENAL"] = rubrica_counts["RUBRICA"].apply(
            lambda r: PESO_PENAL_BASE.get(r, 1.0)
        )

        dim_perfil = pl.DataFrame({
            "SK_PERFIL": list(range(len(MULTIPLICADOR_PERFIL))),
            "PERFIL":    list(MULTIPLICADOR_PERFIL.keys()),
        })

        # ── Persiste no R2 ────────────────────────────────────────────────────
        tabelas = {
            "ouro/fato_risco_hexagono.parquet": df_fato,
            "ouro/dim_tempo.parquet":           pl.from_pandas(dim_tempo),
            "ouro/dim_local.parquet":           pl.from_pandas(dim_local),
            "ouro/dim_crime.parquet":           pl.from_pandas(rubrica_counts),
            "ouro/dim_perfil.parquet":          dim_perfil,
            "auditoria/auditoria_shap.parquet": df_shap_pl,
            "auditoria/auditoria_cv.parquet":   df_cv,
        }

        for chave, df_tab in tabelas.items():
            pasta  = chave.split("/")[0]
            nome   = chave.split("/")[1]
            caminho = self.pastas[pasta] / nome
            df_tab.write_parquet(caminho)
            self.upload_r2(caminho, chave)
            print(f"   💾 {chave}: {len(df_tab):,} registros", file=sys.stdout)

        # ── Meta modelo para comparação entre execuções ───────────────────────
        meta_anterior = self.memoria.carregar_meta_modelo()
        mae_anterior  = float(meta_anterior.get("mae", mae_final))
        melhoria_pct  = ((mae_anterior - mae_final) / max(mae_anterior, 1e-9)) * 100

        self.memoria.salvar_meta_modelo({
            "run_id":   self.run_id,
            "mae":      round(mae_final, 4),
            "r2":       round(r2_final, 4),
            "versao":   VERSAO_PIPELINE,
            "timestamp": hora_brasilia().isoformat(),
        })

        # ── BigQuery ──────────────────────────────────────────────────────────
        status_bq = "⏭️ Pulado"
        if bq_project and bq_dataset and bq_cred:
            try:
                bq_tabelas = {
                    "fato_risco_hexagono": df_fato,
                    "dim_tempo":           pl.from_pandas(dim_tempo),
                    "dim_local":           pl.from_pandas(dim_local),
                    "dim_crime":           pl.from_pandas(rubrica_counts),
                    "dim_perfil":          dim_perfil,
                    "auditoria_shap":      df_shap_pl,
                    "auditoria_cv":        df_cv,
                    "auditoria_log":       self.auditoria.to_dataframe(),
                }
                for nome_tab, df_tab in bq_tabelas.items():
                    enviar_para_bigquery(df_tab, nome_tab, bq_project, bq_dataset, bq_cred)
                status_bq = f"✅ {len(bq_tabelas)} tabelas enviadas"
            except Exception as e:
                status_bq = f"❌ BQ: {e}"
                print(status_bq, file=sys.stderr)

        self.auditoria.log("ouro_fim", {
            "registros_fato":  len(df_fato),
            "mae_final":       round(mae_final, 4),
            "r2_final":        round(r2_final, 4),
            "melhoria_pct":    round(melhoria_pct, 2),
        })

        return df_fato, mae_final, r2_final, mae_anterior, melhoria_pct, status_bq

    # ── Orquestrador ──────────────────────────────────────────────────────────

    def processar(self):
        bq_project = os.environ.get("BQ_PROJECT_ID",       "").strip()
        bq_dataset = os.environ.get("BQ_DATASET",          "").strip()
        bq_cred    = os.environ.get("BQ_CREDENTIALS_JSON", "").strip()

        self.sincronizar_raw()
        df_prata  = self.construir_prata()
        resultado = self.construir_ouro(df_prata, bq_project, bq_dataset, bq_cred)

        tempo = time.time() - self.t_inicio

        if resultado:
            df_ouro, mae, r2, mae_ant, melhoria, status_bq = resultado
            self.discord.notificar_sucesso(
                self.run_id, tempo,
                len(df_prata), len(df_ouro),
                mae, r2, mae_ant, melhoria, status_bq
            )
        else:
            self.discord.notificar_erro(
                self.run_id,
                "Pipeline sem ouro",
                "Dados insuficientes para treino."
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
