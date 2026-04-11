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
    "PEDESTRE":     {"ROUBO": 1.4, "LESAO CORPORAL DOLOSA": 1.4, "ESTUPRO": 1.5},
    "CICLISTA":     {"ROUBO": 1.3},
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

COLUNAS_CRITICAS_SSP = [
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
    print(f"[SSP] Tentando baixar {url}")
    for tentativa in range(1, SSP_MAX_TENTATIVAS + 1):
        try:
            response = requests.get(url, timeout=SSP_TIMEOUT, verify=False)
            response.raise_for_status()
            print(f"[SSP] Download de {ano} bem-sucedido na tentativa {tentativa}.")

            excel_file = pd.ExcelFile(BytesIO(response.content))
            df_consolidado = pd.DataFrame()

            for sheet_name in excel_file.sheet_names:
                if "Campos da Tabela_SPDADOS" in sheet_name:
                    continue

                try:
                    df_sheet = excel_file.parse(sheet_name, dtype=str) # Ler tudo como string

                    colunas_presentes = [col for col in COLUNAS_CRITICAS_SSP if col in df_sheet.columns]
                    if len(colunas_presentes) >= 5: # Critério de 5 colunas críticas
                        print(f"[SSP] Aba '{sheet_name}' identificada como dados para o ano {ano}.")
                        df_consolidado = pd.concat([df_consolidado, df_sheet], ignore_index=True)
                    else:
                        print(f"[SSP] Aba '{sheet_name}' ignorada (poucas colunas críticas).")
                except Exception as e:
                    print(f"[SSP] Erro ao processar aba '{sheet_name}' para {ano}: {e}")

            if not df_consolidado.empty:
                return df_consolidado
            else:
                print(f"[SSP] Nenhuma aba de dados encontrada para o ano {ano}.")
                return None

        except requests.exceptions.RequestException as e:
            print(f"[SSP] Erro ao baixar {url} (tentativa {tentativa}/{SSP_MAX_TENTATIVAS}): {e}")
            if tentativa < SSP_MAX_TENTATIVAS:
                time.sleep(5 * tentativa)
    print(f"[SSP] Falha ao baixar {url} após {SSP_MAX_TENTATIVAS} tentativas.")
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
            print(f"[Discord] Erro ao enviar mensagem: {e}")

    def alerta_erro(self, run_id: str, titulo: str, detalhes: str):
        desc = f"**Run ID:** `{run_id}`\n**Detalhes:** ```{detalhes}```"
        self._enviar_mensagem(self.webhook_erro, f"🚨 {titulo}", desc, 15158332)

    def relatorio_executivo(self, run_id: str, tempo: float, n_registros: int, mae: float, r2: float, melhoria: float, top_municipio: str, top_shap_feature: str):
        desc = (
            f"**Run ID:** `{run_id}`\n"
            f"**Tempo de Execução:** `{tempo:.1f}s`\n"
            f"**Registros Ouro Gerados:** `{n_registros}`\n"
            f"**MAE do Modelo:** `{mae:.4f}`\n"
            f"**R² do Modelo:** `{r2:.4f}`\n"
            f"**Melhoria em MAE (vs. anterior):** `{melhoria:.2f}%`\n"
            f"**Município de Maior Risco (predito):** `{top_municipio}`\n"
            f"**Feature Mais Impactante (SHAP):** `{top_shap_feature}`"
        )
        self._enviar_mensagem(self.webhook_sucesso, f"✅ Execução Concluída - {NOME_SISTEMA}", desc, 3066993)

    def relatorio_operacional(self, run_id: str, n_raw: int, n_prata: int, n_ouro: int, mae: float, r2: float, mae_ant: float, anos_processados: list, status_bq: str, shap_top3: list, prefixo_raw: str):
        shap_str = "\n".join([f"- `{f}`: `{v:.2f}`" for f, v in shap_top3])
        desc = (
            f"**Run ID:** `{run_id}`\n"
            f"**Dados Raw Processados:** `{n_raw}`\n"
            f"**Dados Prata Gerados:** `{n_prata}`\n"
            f"**Dados Ouro Gerados:** `{n_ouro}`\n"
            f"**Anos Processados:** `{', '.join(map(str, anos_processados))}`\n"
            f"**Prefixo R2 Raw:** `{prefixo_raw}`\n"
            f"**MAE Atual:** `{mae:.4f}` (Anterior: `{mae_ant:.4f}`)\n"
            f"**R² Atual:** `{r2:.4f}`\n"
            f"**Status BigQuery:** `{status_bq}`\n"
            f"**Top 3 Features SHAP:**\n{shap_str}"
        )
        self._enviar_mensagem(self.webhook_sucesso, f"📊 Detalhes Operacionais - {NOME_SISTEMA}", desc, 16776960)

    def sem_novidades(self, run_id: str, tempo: float):
        desc = f"**Run ID:** `{run_id}`\n**Tempo de Execução:** `{tempo:.1f}s`\nNenhum dado novo ou alterado para processar."
        self._enviar_mensagem(self.webhook_sucesso, f"💤 Sem Novidades - {NOME_SISTEMA}", desc, 10070709)

class TrackingSSP:
    def __init__(self, s3_client, bucket_name: str, tracking_key: str):
        self.s3 = s3_client
        self.bucket = bucket_name
        self.key = tracking_key
        self.dados = self._carregar_tracking()

    def _carregar_tracking(self) -> dict:
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=self.key)
            content = response["Body"].read().decode("utf-8")
            tracking_raw = json.loads(content)

            # Migração defensiva: se encontrar int onde deveria ser dict, converte
            migrated_tracking = {}
            for ano, info in tracking_raw.items():
                if isinstance(info, int):
                    migrated_tracking[ano] = {"tamanho_bytes": info, "hash_sha256": "legacy_hash"}
                else:
                    migrated_tracking[ano] = info
            return migrated_tracking
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                print("[Tracking] tracking_ssp.json não encontrado — iniciando do zero.")
                return {}
            else:
                raise
        except Exception as e:
            print(f"[Tracking] Erro ao carregar tracking_ssp.json: {e} — iniciando do zero.")
            return {}

    def salvar_tracking(self):
        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=self.key,
                Body=json.dumps(self.dados, indent=2).encode("utf-8"),
                ContentType="application/json"
            )
        except Exception as e:
            print(f"[Tracking] Erro ao salvar tracking_ssp.json: {e}")

    def precisa_processar(self, ano: int, tamanho_bytes: int, hash_sha256: str) -> bool:
        ano_str = str(ano)
        if ano_str not in self.dados:
            return True

        info_existente = self.dados[ano_str]
        if info_existente.get("tamanho_bytes") != tamanho_bytes or info_existente.get("hash_sha256") != hash_sha256:
            return True
        return False

    def atualizar_tracking(self, ano: int, tamanho_bytes: int, hash_sha256: str):
        self.dados[str(ano)] = {"tamanho_bytes": tamanho_bytes, "hash_sha256": hash_sha256}

class SafeDriver:
    def __init__(self):
        self.run_id = run_id_curto()
        self.t_inicio = time.time()
        self.anos_processados = []
        self.df_raw = pl.DataFrame()

        self.r2_bucket = sanitizar_secret(os.environ.get("R2_BUCKET_NAME", ""))
        self.s3 = boto3.client(
            "s3",
            endpoint_url=sanitizar_secret(os.environ.get("R2_ENDPOINT_URL", "")),
            aws_access_key_id=sanitizar_secret(os.environ.get("R2_ACCESS_KEY_ID", "")),
            aws_secret_access_key=sanitizar_secret(os.environ.get("R2_SECRET_ACCESS_KEY", "")),
        )
        self.tracking = TrackingSSP(self.s3, self.r2_bucket, R2_TRACKING)

        webhook_sucesso = sanitizar_secret(os.environ.get("DISCORD_SUCESSO", ""))
        webhook_erro = sanitizar_secret(os.environ.get("DISCORD_ERRO", ""))
        self.discord = DiscordNotifier(webhook_sucesso, webhook_erro)

        self.lgpd_salt = sanitizar_secret(os.environ.get("LGPD_SALT", ""))
        if not self.lgpd_salt:
            raise ValueError("LGPD_SALT não configurado. Anonimização não pode ser garantida.")

    def sincronizar_raw(self):
        print("[sincronizar_raw_inicio]")
        df_raw_acumulado = pl.DataFrame()

        for ano in ANOS_DISPONIVEIS:
            r2_key = f"{R2_PREFIXO_RAW}ssp_{ano}.parquet"

            try:
                response = self.s3.get_object(Bucket=self.r2_bucket, Key=r2_key)
                parquet_data = response["Body"].read()
                current_hash = hashlib.sha256(parquet_data).hexdigest()
                current_size = len(parquet_data)

                if self.tracking.precisa_processar(ano, current_size, current_hash):
                    print(f"[R2] {r2_key} alterado ou novo — processando.")
                    df_ano = pl.read_parquet(BytesIO(parquet_data))
                    df_raw_acumulado = pl.concat([df_raw_acumulado, df_ano])
                    self.anos_processados.append(ano)
                    self.tracking.atualizar_tracking(ano, current_size, current_hash)
                else:
                    print(f"[R2] {r2_key} inalterado — pulando.")
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    print(f"[R2] {r2_key} não existe — tentando baixar da SSP-SP.")
                    df_ssp = baixar_ssp(ano)
                    if df_ssp is not None and not df_ssp.empty:
                        # Forçar todas as colunas para string antes de converter para Polars
                        for col in df_ssp.columns:
                            df_ssp[col] = df_ssp[col].astype(str)

                        df_ano = pl.from_pandas(df_ssp)

                        parquet_buffer = BytesIO()
                        df_ano.write_parquet(parquet_buffer)
                        parquet_buffer.seek(0)

                        current_hash = hashlib.sha256(parquet_buffer.getvalue()).hexdigest()
                        current_size = len(parquet_buffer.getvalue())

                        self.s3.put_object(
                            Bucket=self.r2_bucket,
                            Key=r2_key,
                            Body=parquet_buffer.getvalue(),
                            ContentType="application/octet-stream"
                        )
                        print(f"[R2] {r2_key} salvo no R2 após download da SSP-SP.")
                        df_raw_acumulado = pl.concat([df_raw_acumulado, df_ano])
                        self.anos_processados.append(ano)
                        self.tracking.atualizar_tracking(ano, current_size, current_hash)
                    else:
                        print(f"[SSP] Falha ao obter dados da SSP-SP para o ano {ano}.")
                else:
                    print(f"[R2] Erro ao acessar {r2_key}: {e}")

        self.tracking.salvar_tracking()
        self.df_raw = df_raw_acumulado
        print(f"[raw] {len(self.df_raw)} registros acumulados.")
        print("[sincronizar_raw_fim]")

    def construir_prata(self) -> pl.DataFrame:
        print("[construir_prata_inicio]")
        if self.df_raw.is_empty():
            print("[prata] df_raw vazio, pulando construção da camada prata.")
            return pl.DataFrame()

        df_prata = self.df_raw.clone()

        df_prata = renomear_sinonimos(df_prata)

        df_prata = df_prata.with_columns([
            pl.col("LATITUDE").cast(pl.Float64, strict=False).alias("LATITUDE_F"),
            pl.col("LONGITUDE").cast(pl.Float64, strict=False).alias("LONGITUDE_F"),
            pl.col("DATA_OCORRENCIA_BO").str.to_datetime("%d/%m/%Y", strict=False).alias("DATA_OCORRENCIA_BO_DT"),
            pl.col("HORA_OCORRENCIA_BO").str.slice(0, 5).alias("HORA_OCORRENCIA_BO_STR"),
            pl.col("RUBRICA").map_elements(normalizar_texto, return_type=pl.String).alias("RUBRICA_NORMALIZADA"),
            pl.col("NOME_MUNICIPIO").map_elements(normalizar_texto, return_type=pl.String).alias("NOME_MUNICIPIO_NORMALIZADO"),
        ])

        df_prata = df_prata.filter(
            (pl.col("LATITUDE_F").is_between(SP_LAT_MIN, SP_LAT_MAX)) &
            (pl.col("LONGITUDE_F").is_between(SP_LON_MIN, SP_LON_MAX)) &
            (pl.col("DATA_OCORRENCIA_BO_DT").is_not_null()) &
            (pl.col("RUBRICA_NORMALIZADA").is_in(list(PESO_PENAL_BASE.keys()))) # Filtrar apenas crimes de interesse
        )

        df_prata = df_prata.with_columns([
            pl.col("DATA_OCORRENCIA_BO_DT").dt.year().alias("ANO"),
            pl.col("DATA_OCORRENCIA_BO_DT").dt.month().alias("MES"),
            pl.col("DATA_OCORRENCIA_BO_DT").dt.weekday().alias("DIA_SEMANA"),
            pl.col("HORA_OCORRENCIA_BO_STR").map_elements(classificar_periodo, return_type=pl.String).alias("PERIODO_DIA"),
            pl.col("HORA_OCORRENCIA_BO_STR").map_elements(lambda h: 1 if classificar_periodo(h) in ["NOITE", "MADRUGADA"] else 0, return_type=pl.Int8).alias("IS_NOITE_MADRUGADA"),
            pl.struct(["LATITUDE_F", "LONGITUDE_F"]).map_elements(
                lambda coords: h3.geo_to_h3(coords["LATITUDE_F"], coords["LONGITUDE_F"], H3_RESOLUCAO),
                return_type=pl.String
            ).alias("H3_INDEX"),
            pl.col("LOGRADOURO").map_elements(lambda x: anonimizar_campo(x, self.lgpd_salt), return_type=pl.String).alias("LOGRADOURO_ANONIMIZADO"),
            pl.col("BAIRRO").map_elements(lambda x: anonimizar_campo(x, self.lgpd_salt), return_type=pl.String).alias("BAIRRO_ANONIMIZADO"),
            pl.col("NOME_MUNICIPIO").map_elements(lambda x: anonimizar_campo(x, self.lgpd_salt), return_type=pl.String).alias("NOME_MUNICIPIO_ANONIMIZADO"),
            pl.col("DATA_OCORRENCIA_BO_DT").map_elements(lambda d: 1 if d in holidays.HolidayBase(country='BR', state='SP', years=d.year) else 0, return_type=pl.Int8).alias("IS_FERIADO"),
            pl.col("RUBRICA_NORMALIZADA").map_elements(lambda r: 1 if r in ["ROUBO", "FURTO", "ROUBO DE VEICULO", "FURTO DE VEICULO", "ROUBO DE MOTOCICLETA", "FURTO DE MOTOCICLETA", "ROUBO DE CARGA"] else 0, return_type=pl.Int8).alias("IS_PATRIMONIO"),
            pl.col("RUBRICA_NORMALIZADA").map_elements(lambda r: 1 if r in ["HOMICIDIO DOLOSO", "LATROCINIO", "EXTORSAO MEDIANTE SEQUESTRO", "ESTUPRO", "LESAO CORPORAL DOLOSA", "PORTE ILEGAL DE ARMA"] else 0, return_type=pl.Int8).alias("IS_VIOLENCIA_PESSOA"),
        ])

        df_prata = df_prata.with_columns(
            pl.struct(["RUBRICA_NORMALIZADA", "HORA_OCORRENCIA_BO_STR"]).map_elements(
                lambda s: calcular_escore(s["RUBRICA_NORMALIZADA"], s["HORA_OCORRENCIA_BO_STR"], "PEDESTRE")
            ).alias("ESCORE_TOTAL_OCORRENCIA")
        )

        df_prata = df_prata.with_columns([
            pl.struct(["RUBRICA_NORMALIZADA", "HORA_OCORRENCIA_BO_STR"]).map_elements(
                lambda s: calcular_escores_todos_perfis(s["RUBRICA_NORMALIZADA"], s["HORA_OCORRENCIA_BO_STR"])["ESCORE_MOTORISTA"]
            ).alias("ESCORE_MOTORISTA"),
            pl.struct(["RUBRICA_NORMALIZADA", "HORA_OCORRENCIA_BO_STR"]).map_elements(
                lambda s: calcular_escores_todos_perfis(s["RUBRICA_NORMALIZADA"], s["HORA_OCORRENCIA_BO_STR"])["ESCORE_MOTOCICLISTA"]
            ).alias("ESCORE_MOTOCICLISTA"),
            pl.struct(["RUBRICA_NORMALIZADA", "HORA_OCORRENCIA_BO_STR"]).map_elements(
                lambda s: calcular_escores_todos_perfis(s["RUBRICA_NORMALIZADA"], s["HORA_OCORRENCIA_BO_STR"])["ESCORE_PEDESTRE"]
            ).alias("ESCORE_PEDESTRE"),
            pl.struct(["RUBRICA_NORMALIZADA", "HORA_OCORRENCIA_BO_STR"]).map_elements(
                lambda s: calcular_escores_todos_perfis(s["RUBRICA_NORMALIZADA"], s["HORA_OCORRENCIA_BO_STR"])["ESCORE_CICLISTA"]
            ).alias("ESCORE_CICLISTA"),
        ])

        print(f"[prata] {len(df_prata)} registros gerados.")
        print("[construir_prata_fim]")
        return df_prata

    def construir_ouro(self, df_prata: pl.DataFrame, bq_project: str, bq_dataset: str, bq_cred: str) -> tuple | None:
        print("[construir_ouro_inicio]")
        if df_prata.is_empty():
            print("[ouro] df_prata vazio, pulando construção da camada ouro.")
            return None

        df_agregado = df_prata.group_by(["H3_INDEX", "ANO"]).agg([
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
            pl.col("IS_FERIADO").max().alias("IS_FERIADO"),
            pl.col("NOME_MUNICIPIO_ANONIMIZADO").mode().first().alias("MUNICIPIO_DOMINANTE"),
        ])

        df_agregado = df_agregado.sort(["H3_INDEX", "ANO"])

        df_agregado = df_agregado.with_columns([
            pl.col("ESCORE_TOTAL").shift(2).over("H3_INDEX").alias("ESCORE_LAG2"),
            pl.col("QTD_CRIMES").shift(2).over("H3_INDEX").alias("QTD_LAG2"),
        ])

        df_agregado = df_agregado.fill_null(0).fill_nan(0)

        if len(df_agregado) < MIN_REGISTROS:
            print(f"[ouro] Poucos registros ({len(df_agregado)}) para treinar o modelo. Mínimo: {MIN_REGISTROS}.")
            return None

        features = [
            "QTD_CRIMES", "ESCORE_TOTAL", "ESCORE_MEDIO", "ESCORE_GRAVIDADE_MAX",
            "ESCORE_MOTORISTA", "ESCORE_MOTOCICLISTA", "ESCORE_PEDESTRE", "ESCORE_CICLISTA",
            "PROP_NOITE_MADRUGADA", "PROP_PATRIMONIO", "PROP_VIOLENCIA_PESSOA",
            "ESCORE_LAG2", "QTD_LAG2", "IS_FERIADO"
        ]
        target = "ESCORE_TOTAL"

        X = df_agregado.select(features).to_pandas()
        y = df_agregado.select(target).to_pandas()
        anos = df_agregado.select("ANO").to_pandas().squeeze()

        if X.empty or y.empty:
            print("[ouro] Dados para treinamento vazios após agregação.")
            return None

        tscv = TimeSeriesSplit(n_splits=3)

        lgbm = LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.1, num_leaves=20)
        catb = CatBoostRegressor(random_state=42, verbose=0, n_estimators=100, learning_rate=0.1)

        model = VotingRegressor(estimators=[('lgbm', lgbm), ('catb', catb)])

        oof_preds = np.zeros(len(X))
        oof_targets = np.zeros(len(X))

        for fold, (train_index, test_index) in enumerate(tscv.split(X, y, anos)):
            print(f"[modelo] Treinando Fold {fold+1}...")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train.values.ravel())
            oof_preds[test_index] = model.predict(X_test)
            oof_targets[test_index] = y_test.values.ravel()

        mae = mean_absolute_error(oof_targets, oof_preds)
        r2 = r2_score(oof_targets, oof_preds)
        print(f"[modelo] MAE: {mae:.4f}, R2: {r2:.4f}")

        mae_ant = float(os.environ.get("MAE_ANTERIOR", "0.0"))
        melhoria = ((mae_ant - mae) / mae_ant) * 100 if mae_ant > 0 else 0.0
        if mae_ant == 0.0:
            print("[modelo] Primeira execução ou MAE anterior não disponível.")
        elif melhoria > 0:
            print(f"[modelo] Melhoria no MAE: {melhoria:.2f}%")
        else:
            print(f"[modelo] Piora no MAE: {melhoria:.2f}%")

        model_key = f"{R2_PREFIXO_MODELO}modelo_{NOME_SISTEMA}_{hora_brasilia().strftime('%Y%m%d%H%M%S')}.joblib"
        model_buffer = BytesIO()
        joblib.dump(model, model_buffer)
        model_buffer.seek(0)
        self.s3.put_object(
            Bucket=self.r2_bucket,
            Key=model_key,
            Body=model_buffer.getvalue(),
            ContentType="application/octet-stream"
        )
        print(f"[R2] Modelo salvo como {model_key}.")

        explainer = shap.TreeExplainer(model.estimators_[0]) # Usar o primeiro estimador para SHAP
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_values = np.array(shap_values).mean(axis=0)

        feature_importance = pd.DataFrame({
            'feature': features,
            'shap_abs_mean': np.abs(shap_values).mean(axis=0)
        }).sort_values(by='shap_abs_mean', ascending=False)

        top_shap_feature = feature_importance.iloc[0]['feature']
        shap_top3 = feature_importance.head(3).values.tolist()

        df_agregado = df_agregado.with_columns(pl.Series(name="ESCORE_PREDITO", values=model.predict(X)))

        top_municipio = df_agregado.group_by("MUNICIPIO_DOMINANTE").agg(pl.col("ESCORE_PREDITO").sum().alias("SOMA_ESCORE_PREDITO")).sort("SOMA_ESCORE_PREDITO", descending=True).head(1).select("MUNICIPIO_DOMINANTE").item()

        ouro_key = f"{R2_PREFIXO_OURO}ouro_h3_{hora_brasilia().strftime('%Y%m%d%H%M%S')}.parquet"
        ouro_buffer = BytesIO()
        df_agregado.write_parquet(ouro_buffer)
        ouro_buffer.seek(0)
        self.s3.put_object(
            Bucket=self.r2_bucket,
            Key=ouro_key,
            Body=ouro_buffer.getvalue(),
            ContentType="application/octet-stream"
        )
        print(f"[R2] Camada ouro salva como {ouro_key}.")

        ouro_pl = df_agregado.select([
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
        print("[construir_ouro_fim]")

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
