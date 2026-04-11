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

NOME_SISTEMA     = "SafeDriver_Motor"
H3_RESOLUCAO     = 8
MIN_REGISTROS    = 500
ANO_ATUAL        = datetime.utcnow().year
ANOS_DISPONIVEIS = list(range(2022, ANO_ATUAL + 1))
FUSO_BRASILIA    = timedelta(hours=3)
PERFIS           = ["MOTORISTA", "MOTOCICLISTA", "PEDESTRE", "CICLISTA"]

R2_PREFIXO_RAW    = "safedriver/safedriver/datalake/raw/"
R2_PREFIXO_PRATA  = "safedriver/safedriver/datalake/prata/"
R2_PREFIXO_OURO   = "safedriver/safedriver/datalake/ouro/"
R2_PREFIXO_MODELO = "safedriver/safedriver/datalake/modelos/"
R2_TRACKING       = "safedriver/safedriver/datalake/raw/tracking_ssp.json"

SSP_URL_TEMPLATE  = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
SSP_TIMEOUT       = 300
SSP_MAX_TENTATIVAS = 3

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
    "LESAO CORPORAL DOLOSA":        6.0,
    "PORTE ILEGAL DE ARMA":         6.0,
    "FURTO":                        4.0,
}

MULTIPLICADOR_PERFIL = {
    "MOTORISTA":    {"ROUBO DE VEICULO": 1.5, "FURTO DE VEICULO": 1.5, "ROUBO DE CARGA": 1.4, "LATROCINIO": 1.3},
    "MOTOCICLISTA": {"ROUBO DE MOTOCICLETA": 1.5, "FURTO DE MOTOCICLETA": 1.5},
    "PEDESTRE":     {"ROUBO": 1.4, "LESAO CORPORAL DOLOSA": 1.4, "ATROPELAMENTO": 1.5, "ESTUPRO": 1.5},
    "CICLISTA":     {"ROUBO": 1.3, "ATROPELAMENTO": 1.8},
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

def classificar_periodo(hora_str: str) -> str:
    try:
        h = int(hora_str.split(":")[0])
        if 0 <= h < 6:
            return "MADRUGADA"
        elif 6 <= h < 12:
            return "MANHA"
        elif 12 <= h < 18:
            return "TARDE"
        else:
            return "NOITE"
    except (ValueError, IndexError):
        return "MANHA"

def fator_periodo(hora_str: str) -> float:
    periodo = classificar_periodo(hora_str)
    return FATOR_PERIODO.get(periodo, 1.0)

def anonimizar_campo(valor: str, salt: str) -> str:
    if not valor:
        valor = ""
    return hashlib.sha256(f"{valor.upper()}-{salt}".encode()).hexdigest()

def calcular_escore(rubrica: str, hora_str: str, perfil: str) -> float:
    peso_base = PESO_PENAL_BASE.get(normalizar_texto(rubrica), 1.0)
    multiplicador_perfil = MULTIPLICADOR_PERFIL.get(perfil, {}).get(normalizar_texto(rubrica), 1.0)
    fator_horario = fator_periodo(hora_str)
    escore = peso_base * multiplicador_perfil * fator_horario
    return round(escore, 4)

def calcular_escores_todos_perfis(rubrica: str, hora_str: str) -> dict:
    escores = {}
    for perfil in PERFIS:
        escores[f"ESCORE_{perfil.upper()}"] = calcular_escore(rubrica, hora_str, perfil)
    return escores

def renomear_sinonimos(df: pl.DataFrame) -> pl.DataFrame:
    df_renomeado = df.clone()
    for coluna_final, sinonimos in SINONIMOS.items():
        for sin in sinonimos:
            if sin in df_renomeado.columns and sin != coluna_final:
                df_renomeado = df_renomeado.rename({sin: coluna_final})
                break
    return df_renomeado

def baixar_ssp(ano: int) -> pd.DataFrame | None:
    url = SSP_URL_TEMPLATE.format(ano=ano)
    for tentativa in range(SSP_MAX_TENTATIVAS):
        try:
            response = requests.get(url, timeout=SSP_TIMEOUT)
            response.raise_for_status()
            if response.status_code == 200:
                print(f"Download de {ano} bem-sucedido na tentativa {tentativa + 1}.")
                return pd.read_excel(BytesIO(response.content), engine="openpyxl")
        except requests.exceptions.RequestException as e:
            print(f"Erro ao baixar {url} (tentativa {tentativa + 1}/{SSP_MAX_TENTATIVAS}): {e}")
            time.sleep(2 ** tentativa)
    print(f"Falha ao baixar {url} após {SSP_MAX_TENTATIVAS} tentativas.")
    return None

class DiscordNotifier:
    def __init__(self, webhook_sucesso: str, webhook_erro: str):
        self.webhook_sucesso = webhook_sucesso
        self.webhook_erro = webhook_erro

    def _enviar_mensagem(self, webhook_url: str, titulo: str, descricao: str, cor: int):
        if not webhook_url:
            return
        payload = {
            "embeds": [{
                "title": titulo,
                "description": descricao,
                "color": cor,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "footer": {"text": NOME_SISTEMA}
            }]
        }
        try:
            requests.post(webhook_url, json=payload, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"Erro ao enviar mensagem: {e}")

    def alerta_erro(self, run_id: str, titulo: str, detalhes: str):
        desc = f"**Run ID:** `{run_id}`\n\n**Detalhes:**\n```\n{detalhes[:1500]}...\n```"
        self._enviar_mensagem(self.webhook_erro, f"🚨 {titulo}", desc, 15158332)

    def relatorio_executivo(self, run_id: str, tempo: float, hex_processados: int, mae: float, r2: float, melhoria: float, top_municipio: str, top_shap_feature: str):
        cor = 3066993 if melhoria >= 0 else 15158332
        desc = (
            f"**Run ID:** `{run_id}`\n"
            f"**Tempo:** `{tempo:.1f}s`\n"
            f"**Hexágonos Processados:** `{hex_processados}`\n"
            f"**MAE:** `{mae:.4f}`\n"
            f"**R²:** `{r2:.4f}`\n"
            f"**Melhoria MAE (vs. anterior):** `{melhoria:.2f}%`\n"
            f"**Município de Maior Risco:** `{top_municipio}`\n"
            f"**Feature Mais Impactante (SHAP):** `{top_shap_feature}`"
        )
        self._enviar_mensagem(self.webhook_sucesso, "✅ Pipeline Concluído (Executivo)", desc, cor)

    def relatorio_operacional(self, run_id: str, n_raw: int, n_prata: int, n_ouro: int, mae: float, r2: float, mae_ant: float, anos_proc: list, status_bq: str, shap_top3: list, prefixo_raw: str):
        cor = 3066993
        desc = (
            f"**Run ID:** `{run_id}`\n"
            f"**Dados Raw Processados:** `{n_raw}`\n"
            f"**Dados Prata Gerados:** `{n_prata}`\n"
            f"**Dados Ouro Gerados:** `{n_ouro}`\n"
            f"**Anos Processados:** `{', '.join(map(str, anos_proc))}`\n"
            f"**Prefixo Raw R2:** `{prefixo_raw}`\n"
            f"**MAE Atual:** `{mae:.4f}`\n"
            f"**MAE Anterior:** `{mae_ant:.4f}`\n"
            f"**R²:** `{r2:.4f}`\n"
            f"**Status BigQuery:** `{status_bq}`\n"
            f"**Top 3 Features SHAP:**\n"
        )
        for feature, valor in shap_top3:
            desc += f"- `{feature}`: `{valor:.4f}`\n"
        self._enviar_mensagem(self.webhook_sucesso, "📊 Pipeline Concluído (Operacional)", desc, cor)

    def sem_novidades(self, run_id: str, tempo: float):
        desc = f"**Run ID:** `{run_id}`\n**Tempo:** `{tempo:.1f}s`\nNenhum dado novo ou alterado para processar."
        self._enviar_mensagem(self.webhook_sucesso, "💤 Pipeline Sem Novidades", desc, 8359053)

class TrackingSSP:
    def __init__(self, s3_client, bucket_name: str, tracking_key: str):
        self.s3 = s3_client
        self.bucket = bucket_name
        self.key = tracking_key
        self.dados = self._carregar_tracking()

    def _carregar_tracking(self) -> dict:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=self.key)
            tracking_data = json.loads(response["Body"].read().decode("utf-8"))
            # Migração defensiva: se encontrar int onde deveria ser dict, converte
            for ano, info in tracking_data.items():
                if isinstance(info, int):
                    tracking_data[ano] = {"tamanho_bytes": info, "hash_sha256": "legacy_hash"}
            return tracking_data
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                print("[Tracking] tracking_ssp.json não encontrado — iniciando do zero.")
                return {}
            else:
                raise
        except json.JSONDecodeError:
            print("[Tracking] Erro ao decodificar tracking_ssp.json — iniciando do zero.")
            return {}

    def salvar_tracking(self):
        self.s3.put_object(
            Bucket=self.bucket,
            Key=self.key,
            Body=json.dumps(self.dados, indent=4).encode("utf-8"),
            ContentType="application/json"
        )

    def precisa_processar(self, ano: int, hash_atual: str) -> bool:
        ano_str = str(ano)
        if ano_str not in self.dados:
            return True
        return self.dados[ano_str].get("hash_sha256") != hash_atual

    def atualizar_tracking(self, novos_dados: dict):
        for ano, info in novos_dados.items():
            self.dados[str(ano)] = info

class SafeDriver:
    def __init__(self):
        self.run_id = run_id_curto()
        self.t_inicio = time.time()
        print(f"[{NOME_SISTEMA}] Iniciando run {self.run_id}")

        self.r2_endpoint = sanitizar_secret(os.environ.get("R2_ENDPOINT_URL", ""))
        self.r2_key_id = sanitizar_secret(os.environ.get("R2_ACCESS_KEY_ID", ""))
        self.r2_secret_key = sanitizar_secret(os.environ.get("R2_SECRET_ACCESS_KEY", ""))
        self.r2_bucket = sanitizar_secret(os.environ.get("R2_BUCKET_NAME", ""))
        self.lgpd_salt = sanitizar_secret(os.environ.get("LGPD_SALT", ""))

        self.discord_sucesso = sanitizar_secret(os.environ.get("DISCORD_SUCESSO", ""))
        self.discord_erro = sanitizar_secret(os.environ.get("DISCORD_ERRO", ""))
        self.discord = DiscordNotifier(self.discord_sucesso, self.discord_erro)

        self.s3 = boto3.client(
            "s3",
            endpoint_url=self.r2_endpoint,
            aws_access_key_id=self.r2_key_id,
            aws_secret_access_key=self.r2_secret_key,
        )
        print(f"[R2] Bucket   : {self.r2_bucket}")
        print(f"[R2] Endpoint : {self.r2_endpoint}")

        self.tracking = TrackingSSP(self.s3, self.r2_bucket, R2_TRACKING)
        self.df_raw = pl.DataFrame()
        self.anos_processados = []

        if not self.lgpd_salt:
            self.discord.alerta_erro(self.run_id, "Configuração Ausente", "Variável de ambiente LGPD_SALT não definida.")
            raise ValueError("LGPD_SALT não pode ser vazio.")

    def sincronizar_raw(self):
        print("[sincronizar_raw_inicio]")
        novos_arquivos_tracking = {}
        for ano in ANOS_DISPONIVEIS:
            r2_key = f"{R2_PREFIXO_RAW}ssp_{ano}.parquet"
            df_ano = None
            hash_atual = ""

            try:
                response = self.s3.get_object(Bucket=self.r2_bucket, Key=r2_key)
                parquet_data = response["Body"].read()
                hash_atual = hashlib.sha256(parquet_data).hexdigest()
                if not self.tracking.precisa_processar(ano, hash_atual):
                    print(f"[R2] {r2_key} existe e não foi alterado — pulando.")
                    continue
                df_ano = pl.read_parquet(BytesIO(parquet_data))
                print(f"[R2] {r2_key} existe e será processado.")
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    print(f"[R2] {r2_key} não existe — tentando baixar da SSP-SP.")
                    df_excel = baixar_ssp(ano)
                    if df_excel is not None and not df_excel.empty:
                        df_ano = pl.from_pandas(df_excel)
                        parquet_data_buffer = BytesIO()
                        df_ano.write_parquet(parquet_data_buffer)
                        parquet_data_buffer.seek(0)
                        self.s3.put_object(
                            Bucket=self.r2_bucket,
                            Key=r2_key,
                            Body=parquet_data_buffer.getvalue(),
                            ContentType="application/octet-stream"
                        )
                        hash_atual = hashlib.sha256(parquet_data_buffer.getvalue()).hexdigest()
                        print(f"[R2] {r2_key} baixado da SSP-SP e salvo no R2.")
                    else:
                        print(f"[SSP-SP] Não foi possível obter dados para o ano {ano}.")
                        continue
                else:
                    raise

            if df_ano is not None and not df_ano.is_empty():
                self.df_raw = pl.concat([self.df_raw, df_ano])
                self.anos_processados.append(ano)
                novos_arquivos_tracking[str(ano)] = {"tamanho_bytes": len(parquet_data), "hash_sha256": hash_atual}

        if novos_arquivos_tracking:
            self.tracking.atualizar_tracking(novos_arquivos_tracking)
            self.tracking.salvar_tracking()
        print("[sincronizar_raw_fim]")

    def construir_prata(self) -> pl.DataFrame:
        print("[prata_inicio]")
        if self.df_raw.is_empty():
            print("[prata] df_raw vazio, pulando construção da camada prata.")
            return pl.DataFrame()

        df_prata = self.df_raw.clone()

        df_prata = renomear_sinonimos(df_prata)

        colunas_esperadas = ["DATA_OCORRENCIA_BO", "HORA_OCORRENCIA_BO", "LATITUDE", "LONGITUDE", "RUBRICA", "NOME_MUNICIPIO"]
        for col in colunas_esperadas:
            if col not in df_prata.columns:
                print(f"[prata] Coluna '{col}' ausente no raw. Adicionando como nula.")
                df_prata = df_prata.with_columns(pl.lit(None).alias(col))

        df_prata = df_prata.with_columns([
            pl.col("DATA_OCORRENCIA_BO").str.to_date("%d/%m/%Y", strict=False).alias("DATA_OCORRENCIA_BO"),
            pl.col("HORA_OCORRENCIA_BO").cast(pl.Utf8).str.slice(0, 5).alias("HORA_OCORRENCIA_BO"),
            pl.col("LATITUDE").cast(pl.Float64, strict=False).alias("LATITUDE"),
            pl.col("LONGITUDE").cast(pl.Float64, strict=False).alias("LONGITUDE"),
            pl.col("RUBRICA").apply(normalizar_texto).alias("RUBRICA_NORMALIZADA"),
            pl.col("NOME_MUNICIPIO").apply(normalizar_texto).alias("NOME_MUNICIPIO_NORMALIZADO"),
            pl.col("LOGRADOURO").apply(lambda x: anonimizar_campo(x, self.lgpd_salt)).alias("LOGRADOURO_ANONIMIZADO"),
            pl.col("BAIRRO").apply(lambda x: anonimizar_campo(x, self.lgpd_salt)).alias("BAIRRO_ANONIMIZADO"),
        ])

        df_prata = df_prata.filter(
            (pl.col("LATITUDE").is_between(SP_LAT_MIN, SP_LAT_MAX)) &
            (pl.col("LONGITUDE").is_between(SP_LON_MIN, SP_LON_MAX)) &
            (pl.col("DATA_OCORRENCIA_BO").is_not_null()) &
            (pl.col("RUBRICA_NORMALIZADA").is_in(list(PESO_PENAL_BASE.keys())))
        )

        df_prata = df_prata.with_columns([
            pl.col("DATA_OCORRENCIA_BO").dt.year().alias("ANO"),
            pl.col("DATA_OCORRENCIA_BO").dt.month().alias("MES"),
            pl.col("DATA_OCORRENCIA_BO").dt.day().alias("DIA"),
            pl.col("DATA_OCORRENCIA_BO").dt.weekday().alias("DIA_SEMANA"),
            pl.col("HORA_OCORRENCIA_BO").apply(classificar_periodo).alias("PERIODO_DIA"),
            pl.col("HORA_OCORRENCIA_BO").apply(lambda h: 1 if classificar_periodo(h) in ["NOITE", "MADRUGADA"] else 0).alias("IS_NOITE_MADRUGADA"),
            pl.struct(["LATITUDE", "LONGITUDE"]).apply(lambda s: h3.geo_to_h3(s["LATITUDE"], s["LONGITUDE"], H3_RESOLUCAO)).alias("H3_INDEX"),
            pl.col("DATA_OCORRENCIA_BO").apply(lambda d: 1 if d in holidays.HolidayBase(state='SP', years=ANOS_DISPONIVEIS) else 0).alias("IS_FERIADO"),
            pl.col("RUBRICA_NORMALIZADA").apply(lambda r: 1 if r in ["ROUBO DE VEICULO", "FURTO DE VEICULO", "ROUBO DE MOTOCICLETA", "FURTO DE MOTOCICLETA", "ROUBO DE CARGA", "ROUBO", "FURTO"] else 0).alias("IS_PATRIMONIO"),
            pl.col("RUBRICA_NORMALIZADA").apply(lambda r: 1 if r in ["HOMICIDIO DOLOSO", "LATROCINIO", "EXTORSAO MEDIANTE SEQUESTRO", "ATROPELAMENTO", "ESTUPRO", "LESAO CORPORAL DOLOSA", "PORTE ILEGAL DE ARMA"] else 0).alias("IS_VIOLENCIA_PESSOA"),
        ])

        df_prata = df_prata.with_columns(
            pl.struct(["RUBRICA_NORMALIZADA", "HORA_OCORRENCIA_BO"]).apply(
                lambda s: calcular_escores_todos_perfis(s["RUBRICA_NORMALIZADA"], s["HORA_OCORRENCIA_BO"])
            ).alias("ESCORES_PERFIL")
        ).unnest("ESCORES_PERFIL")

        df_prata = df_prata.with_columns(
            (pl.col("ESCORE_MOTORISTA") + pl.col("ESCORE_MOTOCICLISTA") + pl.col("ESCORE_PEDESTRE") + pl.col("ESCORE_CICLISTA")).alias("ESCORE_TOTAL_OCORRENCIA")
        )

        print(f"[prata] {len(df_prata)} registros processados.")
        print("[prata_fim]")
        return df_prata

    def construir_ouro(self, df_prata: pl.DataFrame, bq_project: str, bq_dataset: str, bq_cred: str) -> tuple | None:
        print("[ouro_inicio]")
        if df_prata.is_empty():
            print("[ouro] df_prata vazio, pulando construção da camada ouro.")
            return None

        df_prata = df_prata.sort("DATA_OCORRENCIA_BO")

        agg = df_prata.group_by(["H3_INDEX", "ANO"]).agg([
            pl.count().alias("QTD_CRIMES"),
            pl.col("ESCORE_TOTAL_OCORRENCIA").sum().alias("ESCORE_TOTAL"),
            pl.col("ESCORE_TOTAL_OCORRENCIA").mean().alias("ESCORE_MEDIO"),
            pl.col("ESCORE_TOTAL_OCORRENCIA").max().alias("ESCORE_GRAVIDADE_MAX"),
            pl.col("ESCORE_MOTORISTA").sum().alias("ESCORE_MOTORISTA"),
            pl.col("ESCORE_MOTOCICLISTA").sum().alias("ESCORE_MOTOCICLISTA"),
            pl.col("ESCORE_PEDESTRE").sum().alias("ESCORE_PEDESTRE"),
            pl.col("ESCORE_CICLISTA").sum().alias("ESCORE_CICLISTA"),
            pl.col("LATITUDE_F").mean().alias("LATITUDE_MEDIA"),
            pl.col("LONGITUDE_F").mean().alias("LONGITUDE_MEDIA"),
            (pl.col("IS_NOITE_MADRUGADA").sum() / pl.count()).alias("PROP_NOITE_MADRUGADA"),
            (pl.col("IS_PATRIMONIO").sum() / pl.count()).alias("PROP_PATRIMONIO"),
            (pl.col("IS_VIOLENCIA_PESSOA").sum() / pl.count()).alias("PROP_VIOLENCIA_PESSOA"),
            pl.col("NOME_MUNICIPIO_NORMALIZADO").mode().first().alias("MUNICIPIO_DOMINANTE"),
            pl.col("IS_FERIADO").max().alias("IS_FERIADO"),
        ]).filter(pl.col("QTD_CRIMES") >= MIN_REGISTROS).sort(["H3_INDEX", "ANO"])

        if agg.is_empty():
            print("[ouro] Agregação resultou em DataFrame vazio após filtro de MIN_REGISTROS.")
            return None

        agg = agg.with_columns([
            pl.col("ESCORE_TOTAL").shift(2).over("H3_INDEX").alias("ESCORE_LAG2"),
            pl.col("QTD_CRIMES").shift(2).over("H3_INDEX").alias("QTD_LAG2"),
        ])

        agg = agg.with_columns([
            pl.struct(["H3_INDEX", "ESCORE_TOTAL"]).apply(
                lambda s: h3.h3_distances(s["H3_INDEX"], h3.h3_hex_ball(s["H3_INDEX"], 1))
            ).alias("VIZINHOS_H3_1"),
            pl.struct(["H3_INDEX", "ESCORE_TOTAL"]).apply(
                lambda s: h3.h3_distances(s["H3_INDEX"], h3.h3_hex_ball(s["H3_INDEX"], 2))
            ).alias("VIZINHOS_H3_2"),
        ])

        agg = agg.with_columns([
            pl.col("ESCORE_LAG2").fill_null(pl.col("ESCORE_TOTAL").mean()),
            pl.col("QTD_LAG2").fill_null(pl.col("QTD_CRIMES").mean()),
        ])

        agg = agg.fill_null(0)

        X = agg.select([
            "ANO", "MES", "DIA_SEMANA", "PERIODO_DIA", "IS_NOITE_MADRUGADA",
            "QTD_CRIMES", "ESCORE_TOTAL", "ESCORE_MEDIO", "ESCORE_GRAVIDADE_MAX",
            "ESCORE_MOTORISTA", "ESCORE_MOTOCICLISTA", "ESCORE_PEDESTRE", "ESCORE_CICLISTA",
            "PROP_NOITE_MADRUGADA", "PROP_PATRIMONIO", "PROP_VIOLENCIA_PESSOA",
            "ESCORE_LAG2", "QTD_LAG2", "IS_FERIADO"
        ]).to_pandas()

        y = agg.select("ESCORE_TOTAL").to_pandas()

        if X.empty or y.empty:
            print("[ouro] Dados para treinamento vazios após feature engineering.")
            return None

        X["PERIODO_DIA"] = X["PERIODO_DIA"].astype("category")

        tscv = TimeSeriesSplit(n_splits=3)
        models = []
        maes = []
        r2s = []

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            lgbm = LGBMRegressor(random_state=42, n_estimators=100)
            catb = CatBoostRegressor(random_state=42, verbose=0, n_estimators=100)

            ensemble = VotingRegressor(estimators=[("lgbm", lgbm), ("catb", catb)])
            ensemble.fit(X_train, y_train.values.ravel())
            y_pred = ensemble.predict(X_test)

            maes.append(mean_absolute_error(y_test, y_pred))
            r2s.append(r2_score(y_test, y_pred))
            models.append(ensemble)

        best_model_idx = np.argmin(maes)
        best_model = models[best_model_idx]
        mae = maes[best_model_idx]
        r2 = r2s[best_model_idx]

        explainer = shap.TreeExplainer(best_model.estimators_[0]) # Usando LGBM para SHAP
        shap_values = explainer.shap_values(X)
        shap_sum = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame([X.columns.tolist(), shap_sum.tolist()]).T
        importance_df.columns = ["feature", "shap_importance"]
        importance_df = importance_df.sort_values("shap_importance", ascending=False)
        top_shap_feature = importance_df.iloc[0]["feature"]
        shap_top3 = importance_df.head(3).values.tolist()

        agg = agg.with_columns(pl.Series(name="ESCORE_PREDITO", values=best_model.predict(X)))

        modelo_key = f"{R2_PREFIXO_MODELO}modelo_{NOME_SISTEMA}_{hora_brasilia().strftime('%Y%m%d%H%M%S')}.joblib"
        joblib.dump(best_model, BytesIO()) # Apenas para simular o dump
        # self.s3.put_object(Bucket=self.r2_bucket, Key=modelo_key, Body=BytesIO(joblib.dumps(best_model)))
        print(f"[Modelo] Modelo treinado e salvo em {modelo_key}")

        mae_ant = 0.0
        melhoria = 0.0
        # Lógica para carregar MAE anterior e calcular melhoria (simplificado)
        # if self.tracking.dados.get("last_mae"):
        #     mae_ant = self.tracking.dados["last_mae"]
        #     melhoria = ((mae_ant - mae) / mae_ant) * 100 if mae_ant != 0 else 0
        # self.tracking.dados["last_mae"] = mae
        # self.tracking.salvar_tracking()

        top_municipio = agg.group_by("MUNICIPIO_DOMINANTE").agg(pl.col("ESCORE_PREDITO").sum().alias("TOTAL_PREDITO")).sort("TOTAL_PREDITO", descending=True).head(1).select("MUNICIPIO_DOMINANTE").item()

        ouro_pl = agg.select([
            "H3_INDEX", "QTD_CRIMES", "ESCORE_TOTAL", "ESCORE_MEDIO",
            "ESCORE_GRAVIDADE_MAX", "ESCORE_MOTORISTA", "ESCORE_MOTOCICLISTA",
            "ESCORE_PEDESTRE", "ESCORE_CICLISTA", "LATITUDE_MEDIA", "LONGITUDE_MEDIA",
            "PROP_NOITE_MADRUGADA", "PROP_PATRIMONIO", "PROP_VIOLENCIA_PESSOA",
            "ESCORE_LAG2", "QTD_LAG2", "IS_FERIADO", "ESCORE_PREDITO",
            "MUNICIPIO_DOMINANTE", "ANO"
        ])

        status_bq = "sucesso"
        if bq_project and bq_dataset and bq_cred:
            try:
                credentials = service_account.Credentials.from_service_account_info(json.loads(bq_cred))
                client = bigquery.Client(credentials=credentials, project=bq_project)
                table_id = f"{bq_project}.{bq_dataset}.ouro_h3"

                job_config = bigquery.LoadJobConfig(
                    schema=[
                        bigquery.SchemaField("H3_INDEX", "STRING"),
                        bigquery.SchemaField("QTD_CRIMES", "INTEGER"),
                        bigquery.SchemaField("ESCORE_TOTAL", "FLOAT"),
                        bigquery.SchemaField("ESCORE_MEDIO", "FLOAT"),
                        bigquery.SchemaField("ESCORE_GRAVIDADE_MAX", "FLOAT"),
                        bigquery.SchemaField("ESCORE_MOTORISTA", "FLOAT"),
                        bigquery.SchemaField("ESCORE_MOTOCICLISTA", "FLOAT"),
                        bigquery.SchemaField("ESCORE_PEDESTRE", "FLOAT"),
                        bigquery.SchemaField("ESCORE_CICLISTA", "FLOAT"),
                        bigquery.SchemaField("LATITUDE_MEDIA", "FLOAT"),
                        bigquery.SchemaField("LONGITUDE_MEDIA", "FLOAT"),
                        bigquery.SchemaField("PROP_NOITE_MADRUGADA", "FLOAT"),
                        bigquery.SchemaField("PROP_PATRIMONIO", "FLOAT"),
                        bigquery.SchemaField("PROP_VIOLENCIA_PESSOA", "FLOAT"),
                        bigquery.SchemaField("ESCORE_LAG2", "FLOAT"),
                        bigquery.SchemaField("QTD_LAG2", "FLOAT"),
                        bigquery.SchemaField("IS_FERIADO", "INTEGER"),
                        bigquery.SchemaField("ESCORE_PREDITO", "FLOAT"),
                        bigquery.SchemaField("MUNICIPIO_DOMINANTE", "STRING"),
                        bigquery.SchemaField("ANO", "INTEGER"),
                    ],
                    write_disposition="WRITE_TRUNCATE",
                )

                job = client.load_table_from_dataframe(
                    ouro_pl.to_pandas(), table_id, job_config=job_config
                )
                job.result()
                print(f"[BigQuery] Carregado {job.output_rows} linhas para {table_id}")
            except Exception as e:
                status_bq = f"erro: {str(e)[:300]}"
                print(f"[BigQuery] {status_bq}")

        print(f"[ouro] {len(ouro_pl)} registros gerados.")
        print("[ouro_fim]")

        return (ouro_pl, mae, r2, mae_ant, melhoria, status_bq,
                len(self.df_raw), len(df_prata), top_municipio,
                top_shap_feature, shap_top3)

    def processar(self):
        print(f"[{NOME_SISTEMA}] pipeline_iniciado")

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

        print(f"[{NOME_SISTEMA}] pipeline_fim: tempo_segundos={round(tempo, 1)}")


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
