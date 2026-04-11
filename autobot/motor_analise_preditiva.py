import sys, os, requests, traceback, hashlib, warnings, time, json, unicodedata, re
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
# CONSTANTES GLOBAIS — versionadas e auditáveis
# ══════════════════════════════════════════════════════════════════════════════

VERSAO_PIPELINE    = "3.0.0"
VERSAO_FEATURES    = "v3"
H3_RESOLUCAO       = 8
MIN_REGISTROS_OURO = 500

# Pesos penais por relevância para motoristas — auditável e versionado
# Qualquer alteração aqui deve incrementar VERSAO_FEATURES
PESO_PENAL_BASE = {
    "HOMICIDIO DOLOSO":               10.0,
    "LATROCINIO":                     10.0,
    "EXTORSAO MEDIANTE SEQUESTRO":    10.0,
    "ROUBO DE VEICULO":               10.0,
    "ROUBO DE MOTOCICLETA":           10.0,
    "ROUBO DE CARGA":                 10.0,
    "ATROPELAMENTO":                   9.0,
    "ESTUPRO":                         9.0,
    "FURTO DE VEICULO":                8.0,
    "FURTO DE MOTOCICLETA":            8.0,
    "ROUBO":                           7.0,
    "ACIDENTE COM MOTOCICLETA":        7.0,
    "LESAO CORPORAL DOLOSA":           6.0,
    "PORTE ILEGAL DE ARMA":            6.0,
    "TRAFICO DE ENTORPECENTES":        5.0,
    "ACIDENTE DE TRANSITO":            5.0,
    "FURTO":                           4.0,
}

MULTIPLICADOR_PERFIL = {
    "MOTORISTA":    {"ROUBO DE VEICULO": 1.5, "FURTO DE VEICULO": 1.5, "ROUBO DE CARGA": 1.4, "LATROCINIO": 1.3},
    "MOTOCICLISTA": {"ROUBO DE MOTOCICLETA": 1.5, "FURTO DE MOTOCICLETA": 1.5, "ACIDENTE COM MOTOCICLETA": 1.5},
    "PEDESTRE":     {"ROUBO": 1.4, "LESAO CORPORAL DOLOSA": 1.4, "ATROPELAMENTO": 1.5, "ESTUPRO": 1.5},
    "CICLISTA":     {"ROUBO": 1.3, "ATROPELAMENTO": 1.8, "ACIDENTE DE TRANSITO": 1.5},
}

# Colunas críticas conforme documento de instruções SSP
COLUNAS_CRITICAS = [
    "NOME_DEPARTAMENTO", "NOME_MUNICIPIO",
    "LOGRADOURO", "LATITUDE", "LONGITUDE", "DATA_OCORRENCIA_BO"
]

# Sinonimos históricos — mapeamento dinâmico de colunas renomeadas pela SSP
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
        self.url_sucesso = os.environ.get("DISCORD_SUCESSO", "").strip(' "\'')
        self.url_erro    = os.environ.get("DISCORD_ERRO",    "").strip(' "\'')

    def _post(self, url, payload):
        if not url or not url.startswith("https://discord"):
            return
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            print(f"❌ Discord: {e}", file=sys.stderr)

    def notificar_sucesso(self, run_id, tempo, reg_prata, reg_ouro,
                           mae, r2, mae_anterior, melhoria_pct, status_bq):
        sinal = "📈" if melhoria_pct > 0 else "📉"
        self._post(self.url_sucesso, {"embeds": [{"title": "🟢 SafeDriver Pipeline Concluído", "color": 3066993, "fields": [
            {"name": "🔑 Run ID",          "value": run_id,                           "inline": True},
            {"name": "📊 Prata",           "value": f"{reg_prata:,}",                 "inline": True},
            {"name": "🏅 Ouro",            "value": f"{reg_ouro:,}",                  "inline": True},
            {"name": "📉 MAE Atual",       "value": f"{mae:.4f}",                     "inline": True},
            {"name": "📈 R²",              "value": f"{r2:.4f}",                      "inline": True},
            {"name": f"{sinal} vs Anterior","value": f"{melhoria_pct:+.1f}% MAE",    "inline": True},
            {"name": "⏱️ Tempo",           "value": f"{tempo:.1f}s",                 "inline": True},
            {"name": "📡 BigQuery",        "value": status_bq,                        "inline": False},
            {"name": "🔢 Versão Pipeline", "value": VERSAO_PIPELINE,                  "inline": True},
        ], "footer": {"text": f"SafeDriver • {hora_brasilia().strftime('%d/%m/%Y %H:%M')}"}}]})

    def notificar_erro(self, run_id, titulo, erro_msg):
        ticks = "```"
        self._post(self.url_erro, {"embeds": [{"title": f"🔴 {titulo}", "color": 15158332, "fields": [
            {"name": "🔑 Run ID",  "value": run_id,                                       "inline": True},
            {"name": "Traceback",  "value": f"{ticks}python\n{erro_msg[:1500]}\n{ticks}", "inline": False},
        ], "footer": {"text": f"SafeDriver • {hora_brasilia().strftime('%d/%m/%Y %H:%M')}"}}]})


# ══════════════════════════════════════════════════════════════════════════════
# REGISTRO DE AUDITORIA — cada execução deixa rastro completo
# ══════════════════════════════════════════════════════════════════════════════

class RegistroAuditoria:
    """
    Cada execução gera um registro imutável salvo no R2 e no BigQuery.
    Permite reconstruir qualquer decisão do modelo em qualquer data.
    """
    def __init__(self, run_id: str):
        self.run_id   = run_id
        self.run_ts   = hora_brasilia().isoformat()
        self.eventos  = []

    def log(self, evento: str, detalhes: dict = None):
        entry = {
            "run_id":   self.run_id,
            "ts":       hora_brasilia().isoformat(),
            "evento":   evento,
            "detalhes": detalhes or {},
        }
        self.eventos.append(entry)
        print(f"  [{self.run_id[:6]}] {evento}", file=sys.stdout)

    def to_dataframe(self) -> pl.DataFrame:
        rows = []
        for e in self.eventos:
            rows.append({
                "RUN_ID":    e["run_id"],
                "RUN_TS":    self.run_ts,
                "TS_EVENTO": e["ts"],
                "EVENTO":    e["evento"],
                "DETALHES":  json.dumps(e["detalhes"], ensure_ascii=False),
                "VERSAO_PIPELINE": VERSAO_PIPELINE,
                "VERSAO_FEATURES": VERSAO_FEATURES,
            })
        return pl.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# MEMÓRIA DO MODELO — o coração do aprendizado contínuo
# ══════════════════════════════════════════════════════════════════════════════

class MemoriaModelo:
    """
    Persiste no R2 o histórico de erros por hexágono de todas as execuções.
    O modelo usa esses erros como features na próxima execução —
    residual learning temporal.
    """
    CHAVE_R2 = "modelos/memoria_erros.parquet"

    def __init__(self, cliente_r2, bucket: str, auditoria: RegistroAuditoria):
        self.r2        = cliente_r2
        self.bucket    = bucket
        self.auditoria = auditoria
        self.df        = self._carregar()

    def _carregar(self) -> pd.DataFrame:
        try:
            obj = self.r2.get_object(Bucket=self.bucket, Key=self.CHAVE_R2)
            df  = pd.read_parquet(BytesIO(obj["Body"].read()))
            self.auditoria.log("memoria_carregada", {
                "execucoes_anteriores": df["RUN_ID"].nunique(),
                "hexagonos_na_memoria": len(df["H3_R8"].unique()),
            })
            return df
        except Exception:
            self.auditoria.log("memoria_nova", {"motivo": "primeiro treino ou arquivo ausente"})
            return pd.DataFrame(columns=[
                "H3_R8", "ANO", "MES", "RUN_ID",
                "ERRO_RESIDUAL", "RISCO_REAL", "RISCO_PREVISTO",
                "MAE_LOCAL", "EXECUCOES_COUNT"
            ])

    def obter_features_residuais(self, df_agg: pd.DataFrame) -> pd.DataFrame:
        """
        Para cada hexágono, calcula features baseadas nos erros históricos:
        - ERRO_MEDIO_HISTORICO: média dos erros absolutos de todas as execuções
        - TENDENCIA_ERRO: se o modelo está melhorando ou piorando nesse hexágono
        - EXECUCOES_COM_DADOS: quantas vezes esse hexágono apareceu no treino
        - RESIDUO_ULTIMA_EXECUCAO: erro da última execução (sinal de correção)
        """
        if self.df.empty:
            df_agg["ERRO_MEDIO_HISTORICO"]  = 0.0
            df_agg["TENDENCIA_ERRO"]         = 0.0
            df_agg["EXECUCOES_COM_DADOS"]    = 0
            df_agg["RESIDUO_ULTIMA_EXECUCAO"]= 0.0
            return df_agg

        resumo = (
            self.df.groupby("H3_R8")
            .agg(
                ERRO_MEDIO_HISTORICO  = ("ERRO_RESIDUAL", "mean"),
                TENDENCIA_ERRO        = ("ERRO_RESIDUAL", lambda x: x.iloc[-1] - x.iloc[0] if len(x) > 1 else 0),
                EXECUCOES_COM_DADOS   = ("RUN_ID", "nunique"),
                RESIDUO_ULTIMA_EXECUCAO = ("ERRO_RESIDUAL", "last"),
            )
            .reset_index()
        )

        df_agg = df_agg.merge(resumo, on="H3_R8", how="left")
        for col in ["ERRO_MEDIO_HISTORICO", "TENDENCIA_ERRO", "RESIDUO_ULTIMA_EXECUCAO"]:
            df_agg[col] = df_agg[col].fillna(0.0)
        df_agg["EXECUCOES_COM_DADOS"] = df_agg["EXECUCOES_COM_DADOS"].fillna(0).astype(int)

        self.auditoria.log("features_residuais_calculadas", {
            "hexagonos_com_historico": int((df_agg["EXECUCOES_COM_DADOS"] > 0).sum()),
            "hexagonos_sem_historico": int((df_agg["EXECUCOES_COM_DADOS"] == 0).sum()),
        })
        return df_agg

    def atualizar(self, df_resultado: pd.DataFrame, run_id: str):
        """
        Após cada treino, salva os erros por hexágono na memória.
        """
        novos = df_resultado[["H3_R8", "ANO", "MES", "RISCO_REAL",
                               "RISCO_PREVISTO", "ERRO_RESIDUAL", "MAE_LOCAL"]].copy()
        novos["RUN_ID"]          = run_id
        novos["EXECUCOES_COUNT"] = 1

        self.df = pd.concat([self.df, novos], ignore_index=True)

        # Mantém apenas as últimas 12 execuções por hexágono para não inflar
        self.df = (
            self.df.sort_values(["H3_R8", "RUN_ID"])
            .groupby("H3_R8")
            .tail(12)
            .reset_index(drop=True)
        )

        self.auditoria.log("memoria_atualizada", {
            "novos_registros": len(novos),
            "total_na_memoria": len(self.df),
        })

    def salvar(self):
        buf = BytesIO()
        self.df.to_parquet(buf, index=False)
        buf.seek(0)
        self.r2.put_object(
            Bucket=self.bucket,
            Key=self.CHAVE_R2,
            Body=buf.getvalue()
        )
        self.auditoria.log("memoria_salva_r2", {"chave": self.CHAVE_R2})


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class SafeDriver:

    R2_PREFIXO = "safedriver/datalake/"

    def __init__(self):
        self.t_inicio  = time.time()
        self.run_id    = gerar_run_id()
        self.discord   = Telemetria()
        self.auditoria = RegistroAuditoria(self.run_id)
        self.pastas    = {
            p: Path(f"datalake/{p}")
            for p in ["raw", "prata", "ouro", "auditoria", "modelos"]
        }
        for p in self.pastas.values():
            p.mkdir(parents=True, exist_ok=True)

        self.r2 = boto3.client(
            "s3",
            endpoint_url          = os.environ.get("R2_ENDPOINT_URL"),
            aws_access_key_id     = os.environ.get("R2_ACCESS_KEY_ID"),
            aws_secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY"),
        )
        self.bucket = os.environ.get("R2_BUCKET_NAME", "safedriver")
        self.auditoria.log("pipeline_iniciado", {
            "run_id":           self.run_id,
            "versao_pipeline":  VERSAO_PIPELINE,
            "versao_features":  VERSAO_FEATURES,
        })

    # ── R2 ───────────────────────────────────────────────────────────────────

    def upload_r2(self, caminho_local: Path, chave_r2: str):
        chave_completa = f"{self.R2_PREFIXO}{chave_r2}"
        self.r2.upload_file(str(caminho_local), self.bucket, chave_completa)
        self.auditoria.log("upload_r2", {"chave": chave_completa, "tamanho": caminho_local.stat().st_size})

    def download_r2(self, chave_r2: str, destino: Path):
        chave_completa = f"{self.R2_PREFIXO}{chave_r2}"
        self.r2.download_file(self.bucket, chave_completa, str(destino))

    # ── Camada RAW → lê parquets anuais que já existem no R2 ─────────────────

    def sincronizar_raw(self):
        self.auditoria.log("sincronizar_raw_inicio")
        prefixo = f"{self.R2_PREFIXO}raw/"
        resp    = self.r2.list_objects_v2(Bucket=self.bucket, Prefix=prefixo)
        objetos = [o["Key"] for o in resp.get("Contents", []) if o["Key"].endswith(".parquet")]

        baixados = 0
        for chave in objetos:
            nome    = Path(chave).name
            destino = self.pastas["raw"] / nome
            if not destino.exists():
                self.r2.download_file(self.bucket, chave, str(destino))
                baixados += 1

        self.auditoria.log("sincronizar_raw_fim", {
            "arquivos_disponiveis": len(objetos),
            "baixados_agora":       baixados,
        })

    # ── Resolução de colunas — lida com renomeações da SSP ───────────────────

    def _resolver_colunas(self, colunas_reais: list) -> dict:
        mapa = {}
        colunas_upper = {c.upper(): c for c in colunas_reais}
        for nome_padrao, sinonimos in SINONIMOS.items():
            if nome_padrao.upper() in colunas_upper:
                mapa[colunas_upper[nome_padrao.upper()]] = nome_padrao
            else:
                for sin in sinonimos:
                    if sin.upper() in colunas_upper:
                        mapa[colunas_upper[sin.upper()]] = nome_padrao
                        self.auditoria.log("coluna_renomeada_detectada", {
                            "original": sin, "padrao": nome_padrao
                        })
                        break
        return mapa

    # ── Camada PRATA — limpeza rigorosa sem recuperação de coordenadas ────────

    def construir_prata(self) -> pl.DataFrame:
        self.auditoria.log("prata_inicio")
        arquivos = sorted(self.pastas["raw"].glob("ssp_*.parquet"))

        if not arquivos:
            raise RuntimeError("Nenhum arquivo raw encontrado.")

        dfs = []
        for arq in arquivos:
            df = pl.read_parquet(arq)

            # Resolver colunas renomeadas
            mapa = self._resolver_colunas(df.columns)
            if mapa:
                df = df.rename(mapa)

            # Manter apenas colunas conhecidas que existem
            colunas_alvo = [
                "NOME_DEPARTAMENTO", "NOME_SECCIONAL", "NOME_DELEGACIA",
                "NOME_MUNICIPIO", "LOGRADOURO", "NUMERO_LOGRADOURO",
                "BAIRRO", "LATITUDE", "LONGITUDE",
                "DATA_OCORRENCIA_BO", "HORA_OCORRENCIA_BO",
                "RUBRICA", "DESCR_CONDUTA", "NATUREZA_APURADA", "DATA_REGISTRO"
            ]
            colunas_presentes = [c for c in colunas_alvo if c in df.columns]
            df = df.select(colunas_presentes)

            # Garantir tipos
            for col in ["LATITUDE", "LONGITUDE"]:
                if col in df.columns:
                    df = df.with_columns(pl.col(col).cast(pl.Float64, strict=False))

            dfs.append(df)
            self.auditoria.log("raw_carregado", {"arquivo": arq.name, "registros": len(df)})

        df = pl.concat(dfs, how="diagonal")
        total_bruto = len(df)

        # ── Limpeza conforme documento de instruções ──────────────────────────

        # 1. Padronizar texto
        for col in ["NOME_MUNICIPIO", "NOME_DEPARTAMENTO", "LOGRADOURO", "BAIRRO", "RUBRICA"]:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).map_elements(normalizar_texto, return_dtype=pl.Utf8)
                )

        # 2. Filtrar apenas São Paulo
        df = df.filter(
            pl.col("NOME_MUNICIPIO").str.contains("SAO PAULO")
        )
        apos_sp = len(df)

        # 3. Descartar coordenadas nulas ou zero — sem recuperação conforme instrução
        df = df.filter(
            pl.col("LATITUDE").is_not_null()  &
            pl.col("LONGITUDE").is_not_null() &
            (pl.col("LATITUDE")  != 0.0)      &
            (pl.col("LONGITUDE") != 0.0)
        )
        apos_coords = len(df)

        # 4. Descartar campos críticos nulos
        for col in ["RUBRICA", "DATA_OCORRENCIA_BO"]:
            if col in df.columns:
                df = df.filter(pl.col(col).is_not_null())

        # 5. Parse de data
        if "DATA_OCORRENCIA_BO" in df.columns:
            df = df.with_columns(
                pl.col("DATA_OCORRENCIA_BO").cast(pl.Utf8)
                  .str.strptime(pl.Date, "%Y-%m-%d", strict=False)
                  .alias("DATA_OCORRENCIA_BO")
            )
            df = df.filter(pl.col("DATA_OCORRENCIA_BO").is_not_null())

        # 6. Metadados de qualidade LGPD-safe
        df = df.with_columns([
            pl.lit(self.run_id).alias("PRATA_RUN_ID"),
            pl.lit(hora_brasilia().isoformat()).alias("PRATA_TS"),
            pl.lit(VERSAO_PIPELINE).alias("VERSAO_PIPELINE"),
        ])

        total_final = len(df)
        self.auditoria.log("prata_fim", {
            "total_bruto":        total_bruto,
            "apos_filtro_sp":     apos_sp,
            "apos_filtro_coords": apos_coords,
            "total_final":        total_final,
            "descartados_pct":    round((1 - total_final / total_bruto) * 100, 1),
        })

        caminho = self.pastas["prata"] / "prata_ocorrencias.parquet"
        df.write_parquet(caminho)
        self.upload_r2(caminho, "prata/prata_ocorrencias.parquet")

        return df

    # ── Camada OURO — features + modelo + star schema ─────────────────────────

    def construir_ouro(self, df_prata: pl.DataFrame,
                        bq_project: str, bq_dataset: str, bq_cred: str):

        self.auditoria.log("ouro_inicio")

        if len(df_prata) < MIN_REGISTROS_OURO:
            self.auditoria.log("ouro_abortado", {"motivo": "dados insuficientes"})
            return None

        # Carregar memória de erros anteriores
        memoria = MemoriaModelo(self.r2, self.bucket, self.auditoria)

        df = df_prata.to_pandas()

        # ── Features temporais ────────────────────────────────────────────────
        df["DATA_OCORRENCIA_BO"] = pd.to_datetime(df["DATA_OCORRENCIA_BO"], errors="coerce")
        df = df.dropna(subset=["DATA_OCORRENCIA_BO"])
        df["ANO"]  = df["DATA_OCORRENCIA_BO"].dt.year
        df["MES"]  = df["DATA_OCORRENCIA_BO"].dt.month
        df["DIA_SEMANA"]   = df["DATA_OCORRENCIA_BO"].dt.dayofweek
        df["FIM_SEMANA"]   = (df["DIA_SEMANA"] >= 5).astype(int)
        df["HORA"]         = pd.to_numeric(
            df["HORA_OCORRENCIA_BO"].str[:2] if "HORA_OCORRENCIA_BO" in df.columns else 0,
            errors="coerce"
        ).fillna(12).astype(int)
        df["TURNO"] = pd.cut(df["HORA"], bins=[-1,6,12,18,23],
                              labels=["MADRUGADA","MANHA","TARDE","NOITE"])

        # ── Feriados SP ───────────────────────────────────────────────────────
        feriados_sp = set()
        for ano in df["ANO"].unique():
            feriados_sp.update(holidays.Brazil(state="SP", years=int(ano)).keys())
        df["FERIADO"] = df["DATA_OCORRENCIA_BO"].dt.date.isin(feriados_sp).astype(int)

        # ── Peso penal por rubrica e perfil ───────────────────────────────────
        def calcular_peso(rubrica, perfil):
            rubrica_norm = normalizar_texto(str(rubrica)) if rubrica else ""
            peso_base    = 1.0
            for crime, peso in PESO_PENAL_BASE.items():
                if crime in rubrica_norm:
                    peso_base = peso
                    break
            mult = MULTIPLICADOR_PERFIL.get(perfil, {})
            for crime, m in mult.items():
                if crime in rubrica_norm:
                    return peso_base * m
            return peso_base

        for perfil in MULTIPLICADOR_PERFIL.keys():
            df[f"PESO_{perfil}"] = df["RUBRICA"].apply(
                lambda r: calcular_peso(r, perfil)
            )

        # ── Flag crime veicular ───────────────────────────────────────────────
        crimes_veiculo = ["ROUBO DE VEICULO", "FURTO DE VEICULO",
                          "ROUBO DE CARGA", "ROUBO DE MOTOCICLETA", "FURTO DE MOTOCICLETA"]
        df["CRIME_VEICULO"] = df["RUBRICA"].apply(
            lambda r: int(any(c in normalizar_texto(str(r)) for c in crimes_veiculo))
        )

        # ── Atraso de registro (proxy de subnotificação) ──────────────────────
        if "DATA_REGISTRO" in df.columns:
            df["DATA_REGISTRO"] = pd.to_datetime(df["DATA_REGISTRO"], errors="coerce")
            df["ATRASO_REGISTRO_DIAS"] = (
                df["DATA_REGISTRO"] - df["DATA_OCORRENCIA_BO"]
            ).dt.days.clip(0, 365).fillna(0)
        else:
            df["ATRASO_REGISTRO_DIAS"] = 0

        # ── H3 resolução 8 ────────────────────────────────────────────────────
        df["H3_R8"] = df.apply(
            lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], H3_RESOLUCAO)
            if pd.notna(r["LATITUDE"]) and pd.notna(r["LONGITUDE"]) else None,
            axis=1
        )
        df = df.dropna(subset=["H3_R8"])

        # ── Agregação por hexágono × mês ──────────────────────────────────────
        agg = df.groupby(["H3_R8", "ANO", "MES"]).agg(
            QTD_CRIMES          = ("RUBRICA", "count"),
            PESO_PENAL_TOTAL    = ("PESO_MOTORISTA", "sum"),
            PCT_FIM_SEMANA      = ("FIM_SEMANA", "mean"),
            PCT_FERIADO         = ("FERIADO", "mean"),
            QTD_CRIMES_VEICULO  = ("CRIME_VEICULO", "sum"),
            MEDIA_ATRASO        = ("ATRASO_REGISTRO_DIAS", "mean"),
            **{f"QTD_{p}": (f"PESO_{p}", "sum") for p in MULTIPLICADOR_PERFIL.keys()},
        ).reset_index()

        agg["PCT_CRIMES_VEICULO"] = agg["QTD_CRIMES_VEICULO"] / agg["QTD_CRIMES"].clip(lower=1)

        # ── Features de vizinhança espacial H3 ───────────────────────────────
        # Resolve o problema da faixa 21-50 identificado no CSV de validação
        h3_crimes = agg.groupby("H3_R8")["QTD_CRIMES"].mean().to_dict()
        def media_vizinhos(hex_id):
            vizinhos = h3.grid_disk(hex_id, 1) - {hex_id}
            vals = [h3_crimes.get(v, 0) for v in vizinhos]
            return np.mean(vals) if vals else 0.0
        agg["MEDIA_CRIMES_VIZINHOS"] = agg["H3_R8"].apply(media_vizinhos)

        # ── Lags temporais ────────────────────────────────────────────────────
        agg = agg.sort_values(["H3_R8", "ANO", "MES"])
        agg["QTD_CRIMES_LAG1"] = agg.groupby("H3_R8")["QTD_CRIMES"].shift(1).fillna(0)
        agg["QTD_CRIMES_LAG2"] = agg.groupby("H3_R8")["QTD_CRIMES"].shift(2).fillna(0)
        agg["QTD_CRIMES_MM3"]  = agg.groupby("H3_R8")["QTD_CRIMES"].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean()
        ).fillna(0)

        # ── Injetar features residuais da memória ────────────────────────────
        agg = memoria.obter_features_residuais(agg)

        # ── Variável alvo ─────────────────────────────────────────────────────
        agg["RISCO_REAL"] = (
            agg["PESO_PENAL_TOTAL"] *
            (1 + agg["PCT_FIM_SEMANA"]) *
            (1 + agg["PCT_FERIADO"] * 0.3)
        ).round(4)

        # ── Features do modelo ────────────────────────────────────────────────
        FEATURES = [
            "QTD_CRIMES", "PESO_PENAL_TOTAL",
            "PCT_FIM_SEMANA", "PCT_FERIADO",
            "QTD_CRIMES_VEICULO", "PCT_CRIMES_VEICULO",
            "MEDIA_ATRASO", "MEDIA_CRIMES_VIZINHOS",
            "QTD_CRIMES_LAG1", "QTD_CRIMES_LAG2", "QTD_CRIMES_MM3",
            # Features de memória — aprendizado contínuo
            "ERRO_MEDIO_HISTORICO", "TENDENCIA_ERRO",
            "EXECUCOES_COM_DADOS", "RESIDUO_ULTIMA_EXECUCAO",
        ]

        agg_valido = agg.dropna(subset=FEATURES + ["RISCO_REAL"])
        X = agg_valido[FEATURES].values
        y = agg_valido["RISCO_REAL"].values

        if len(agg_valido) < 100:
            self.auditoria.log("ouro_abortado", {"motivo": "dados insuficientes após agregação"})
            return None

        # ── Treino com TimeSeriesSplit — zero data leakage ────────────────────
        tscv   = TimeSeriesSplit(n_splits=5)
        maes   = []
        r2s    = []

        lgbm = LGBMRegressor(
            n_estimators=800, learning_rate=0.03,
            num_leaves=63, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )
        cat = CatBoostRegressor(
            iterations=600, learning_rate=0.03,
            depth=7, random_seed=42, verbose=0
        )
        modelo = VotingRegressor([("lgbm", lgbm), ("cat", cat)])

        resultados_cv = []
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            mae_fold = mean_absolute_error(y_test, y_pred)
            r2_fold  = r2_score(y_test, y_pred)
            maes.append(mae_fold)
            r2s.append(r2_fold)
            resultados_cv.append({
                "RUN_ID": self.run_id, "FOLD": fold,
                "MAE": mae_fold, "R2": r2_fold,
                "N_TRAIN": len(train_idx), "N_TEST": len(test_idx),
                "VERSAO_FEATURES": VERSAO_FEATURES,
            })

        mae_final = float(np.mean(maes))
        r2_final  = float(np.mean(r2s))

        # Treino final em todos os dados
        modelo.fit(X, y)
        y_pred_final = modelo.predict(X)

        self.auditoria.log("modelo_treinado", {
            "mae_cv":  round(mae_final, 4),
            "r2_cv":   round(r2_final, 4),
            "n_amostras": len(agg_valido),
            "features": FEATURES,
        })

        # ── Calcular erros residuais para memória ─────────────────────────────
        agg_valido = agg_valido.copy()
        agg_valido["RISCO_PREVISTO"] = y_pred_final
        agg_valido["ERRO_RESIDUAL"]  = agg_valido["RISCO_REAL"] - agg_valido["RISCO_PREVISTO"]
        agg_valido["MAE_LOCAL"]      = agg_valido["ERRO_RESIDUAL"].abs()

        # ── Atualizar e salvar memória ────────────────────────────────────────
        memoria.atualizar(agg_valido, self.run_id)
        memoria.salvar()

        # ── Salvar modelo serializado ─────────────────────────────────────────
        modelo_path = self.pastas["modelos"] / f"modelo_{self.run_id}.joblib"
        joblib.dump({"modelo": modelo, "features": FEATURES, "run_id": self.run_id,
                     "mae": mae_final, "r2": r2_final, "ts": hora_brasilia().isoformat()},
                    modelo_path)
        self.upload_r2(modelo_path, f"modelos/modelo_{self.run_id}.joblib")

        # ── Salvar ponteiro para modelo atual (sempre sobrescreve) ────────────
        ponteiro_path = self.pastas["modelos"] / "modelo_atual.json"
        with open(ponteiro_path, "w") as f:
            json.dump({"run_id": self.run_id, "mae": mae_final, "r2": r2_final,
                       "ts": hora_brasilia().isoformat(), "features": FEATURES}, f)
        self.upload_r2(ponteiro_path, "modelos/modelo_atual.json")

        # ── SHAP — auditoria do modelo ────────────────────────────────────────
        explainer = shap.TreeExplainer(modelo.estimators_[0])
        shap_vals = explainer.shap_values(X)
        df_shap   = pd.DataFrame(shap_vals, columns=FEATURES)
        df_shap["H3_R8"]  = agg_valido["H3_R8"].values
        df_shap["ANO"]    = agg_valido["ANO"].values
        df_shap["MES"]    = agg_valido["MES"].values
        df_shap["RUN_ID"] = self.run_id

        # ── Percentil de risco ────────────────────────────────────────────────
        agg_valido["SCORE_PERCENTIL"] = pd.qcut(
            agg_valido["RISCO_PREVISTO"], q=10, labels=False, duplicates="drop"
        ).fillna(0).astype(int)

        # ══════════════════════════════════════════════════════════════════════
        # STAR SCHEMA — tabelas do Looker
        # ══════════════════════════════════════════════════════════════════════

        # dim_tempo
        datas = pd.date_range("2022-01-01", periods=60, freq="MS")
        dim_tempo = pd.DataFrame({
            "SK_TEMPO":      [int(d.strftime("%Y%m%d")) for d in datas],
            "ANO":           datas.year,
            "MES":           datas.month,
            "TRIMESTRE":     datas.quarter,
            "NOME_MES":      datas.strftime("%B"),
            "ANO_MES":       datas.strftime("%Y-%m"),
            "IS_PRIMEIRO_SEM": (datas.month <= 6).astype(int),
        })

        # dim_local
        dim_local_agg = agg_valido.groupby("H3_R8").agg(
            LAT_CENTRO  = ("H3_R8", lambda x: h3.cell_to_latlng(x.iloc[0])[0]),
            LON_CENTRO  = ("H3_R8", lambda x: h3.cell_to_latlng(x.iloc[0])[1]),
        ).reset_index()
        dim_local_agg["SK_LOCAL"] = dim_local_agg["H3_R8"].apply(
            lambda x: int(hashlib.md5(x.encode()).hexdigest()[:8], 16)
        )
        dim_local_agg["RESOLUCAO_H3"] = H3_RESOLUCAO

        # dim_crime
        rubrica_counts = df.groupby("RUBRICA").agg(
            QTD_TOTAL       = ("RUBRICA", "count"),
            MEDIA_PESO_PENAL= ("PESO_MOTORISTA", "mean"),
        ).reset_index()
        rubrica_counts["SK_CRIME"] = range(1, len(rubrica_counts) + 1)
        rubrica_counts["CRIME_VEICULO"] = rubrica_counts["RUBRICA"].apply(
            lambda r: int(any(c in r for c in crimes_veiculo))
        )

        # dim_perfil
        dim_perfil = pl.DataFrame({
            "SK_PERFIL": [1, 2, 3, 4],
            "PERFIL":    ["MOTORISTA", "MOTOCICLISTA", "PEDESTRE", "CICLISTA"],
            "DESCRICAO": [
                "Condutor de automóvel",
                "Condutor de motocicleta",
                "Usuário a pé",
                "Ciclista"
            ],
        })

        # fato_risco_hexagono — granularidade hexágono × mês × perfil
        fatos = []
        for _, row in agg_valido.iterrows():
            for sk_perfil, perfil in enumerate(MULTIPLICADOR_PERFIL.keys(), 1):
                risco_perfil = float(row.get(f"QTD_{perfil}", 0))
                fatos.append({
                    "SK_LOCAL":   int(hashlib.md5(str(row["H3_R8"]).encode()).hexdigest()[:8], 16),
                    "SK_TEMPO":   int(f"{int(row['ANO'])}{int(row['MES']):02d}01"),
                    "SK_PERFIL":  sk_perfil,
                    "H3_R8":      row["H3_R8"],
                    "ANO":        int(row["ANO"]),
                    "MES":        int(row["MES"]),

                    # Métricas observadas
                    "QTD_CRIMES_TOTAL":   int(row["QTD_CRIMES"]),
                    "PESO_PENAL_TOTAL":   round(float(row["PESO_PENAL_TOTAL"]), 2),
                    "PCT_FIM_SEMANA":     round(float(row["PCT_FIM_SEMANA"]), 4),
                    "PCT_FERIADO":        round(float(row["PCT_FERIADO"]), 4),
                    "MEDIA_CRIMES_VIZINHOS": round(float(row["MEDIA_CRIMES_VIZINHOS"]), 2),

                    # Métricas do modelo
                    "RISCO_REAL":         round(float(row["RISCO_REAL"]), 4),
                    "RISCO_PREVISTO":     round(float(row["RISCO_PREVISTO"]), 4),
                    "RISCO_PERFIL":       round(risco_perfil, 4),
                    "ERRO_RESIDUAL":      round(float(row["ERRO_RESIDUAL"]), 4),
                    "SCORE_PERCENTIL":    int(row["SCORE_PERCENTIL"]),

                    # Memória do modelo — rastreabilidade do aprendizado
                    "ERRO_MEDIO_HISTORICO":   round(float(row["ERRO_MEDIO_HISTORICO"]), 4),
                    "TENDENCIA_ERRO":         round(float(row["TENDENCIA_ERRO"]), 4),
                    "EXECUCOES_COM_DADOS":    int(row["EXECUCOES_COM_DADOS"]),
                    "RESIDUO_ULTIMA_EXECUCAO":round(float(row["RESIDUO_ULTIMA_EXECUCAO"]), 4),

                    # Rastreabilidade
                    "RUN_ID":           self.run_id,
                    "RUN_TS":           hora_brasilia().isoformat(),
                    "MAE_MODELO":       round(mae_final, 4),
                    "R2_MODELO":        round(r2_final, 4),
                    "VERSAO_PIPELINE":  VERSAO_PIPELINE,
                    "VERSAO_FEATURES":  VERSAO_FEATURES,
                })

        df_fato    = pl.DataFrame(fatos)
        df_shap_pl = pl.from_pandas(df_shap)
        df_cv      = pl.DataFrame(resultados_cv)

        # ── Salvar todas as tabelas ────────────────────────────────────────────
        tabelas = {
            "ouro/fato_risco_hexagono.parquet":  df_fato,
            "ouro/dim_tempo.parquet":            pl.from_pandas(dim_tempo),
            "ouro/dim_local.parquet":            pl.from_pandas(dim_local_agg),
            "ouro/dim_crime.parquet":            pl.from_pandas(rubrica_counts),
            "ouro/dim_perfil.parquet":           dim_perfil,
            "auditoria/auditoria_shap.parquet":  df_shap_pl,
            "auditoria/auditoria_cv.parquet":    df_cv,
            "auditoria/auditoria_log.parquet":   self.auditoria.to_dataframe(),
        }

        for chave, df_tabela in tabelas.items():
            pasta  = chave.split("/")[0]
            nome   = chave.split("/")[1]
            caminho = self.pastas[pasta] / nome
            df_tabela.write_parquet(caminho)
            self.upload_r2(caminho, chave)

        # ── Carregar MAE anterior para comparação ─────────────────────────────
        mae_anterior  = mae_final
        melhoria_pct  = 0.0
        try:
            obj = self.r2.get_object(
                Bucket=self.bucket,
                Key=f"{self.R2_PREFIXO}modelos/modelo_atual.json"
            )
            dados_anterior = json.loads(obj["Body"].read())
            if dados_anterior.get("run_id") != self.run_id:
                mae_anterior = float(dados_anterior.get("mae", mae_final))
                melhoria_pct = ((mae_anterior - mae_final) / mae_anterior) * 100
        except Exception:
            pass

        # ── BigQuery ───────────────────────────────────────────────────────────
        status_bq = "⏭️ Pulado"
        if bq_project and bq_dataset and bq_cred:
            try:
                bq_tabelas = {
                    "fato_risco_hexagono": df_fato,
                    "dim_tempo":           pl.from_pandas(dim_tempo),
                    "dim_local":           pl.from_pandas(dim_local_agg),
                    "dim_crime":           pl.from_pandas(rubrica_counts),
                    "dim_perfil":          dim_perfil,
                    "auditoria_shap":      df_shap_pl,
                    "auditoria_cv":        df_cv,
                    "auditoria_log":       self.auditoria.to_dataframe(),
                }
                for nome, df_tab in bq_tabelas.items():
                    enviar_para_bigquery(df_tab, nome, bq_project, bq_dataset, bq_cred)
                status_bq = f"✅ {len(bq_tabelas)} tabelas enviadas"
            except Exception as e:
                status_bq = f"❌ BQ: {e}"
                print(status_bq, file=sys.stderr)

        self.auditoria.log("ouro_fim", {
            "registros_fato": len(df_fato),
            "mae_final":      round(mae_final, 4),
            "r2_final":       round(r2_final, 4),
            "melhoria_pct":   round(melhoria_pct, 2),
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
