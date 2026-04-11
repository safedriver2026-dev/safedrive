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
    "NOME_MUNICIPIO":     ["MUNICIPIO", "MUN", "CIDADE"], # Adicionado "CIDADE" aqui
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

# Definir um conjunto padrão de colunas esperadas após a renomeação para garantir consistência
COLUNAS_PADRAO_RAW = list(set(SINONIMOS.keys())) # Todas as colunas alvo dos sinonimos

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
            df_consolidado_abas = []

            for sheet_name in excel_file.sheet_names:
                if "Campos da Tabela_SPDADOS" in sheet_name:
                    continue

                try:
                    df_sheet = excel_file.parse(sheet_name, dtype=str) # Ler tudo como string

                    # Verificar colunas críticas para identificar abas de dados
                    colunas_presentes = [col for col in COLUNAS_CRITICAS_SSP if col in df_sheet.columns]
                    if len(colunas_presentes) >= 5:
                        print(f"[SSP] Aba '{sheet_name}' identificada como dados para o ano {ano}.")
                        df_consolidado_abas.append(df_sheet)
                    else:
                        print(f"[SSP] Aba '{sheet_name}' ignorada (poucas colunas críticas).")
                except Exception as e:
                    print(f"[SSP] Erro ao processar aba '{sheet_name}' do ano {ano}: {e}")
                    continue

            if not df_consolidado_abas:
                print(f"[SSP] Nenhuma aba de dados encontrada para o ano {ano}.")
                return None

            return pd.concat(df_consolidado_abas, ignore_index=True)

        except requests.exceptions.RequestException as e:
            print(f"[SSP] Erro ao baixar {url} (tentativa {tentativa}/{SSP_MAX_TENTATIVAS}): {e}")
            time.sleep(5)
    print(f"[SSP] Falha ao baixar {url} após {SSP_MAX_TENTATIVAS} tentativas.")
    return None

class TrackingSSP:
    def __init__(self, s3_client, r2_bucket, r2_tracking_key):
        self.s3 = s3_client
        self.r2_bucket = r2_bucket
        self.r2_tracking_key = r2_tracking_key
        self.dados = self._carregar_tracking()

    def _carregar_tracking(self):
        try:
            response = self.s3.get_object(Bucket=self.r2_bucket, Key=self.r2_tracking_key)
            tracking_content = response["Body"].read().decode("utf-8")
            dados = json.loads(tracking_content)

            # Migração defensiva: se encontrar int onde deveria ser dict, corrige
            for ano, info in dados.items():
                if isinstance(info, int):
                    dados[ano] = {"tamanho_bytes": info, "hash_sha256": ""}
            return dados
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                print("[Tracking] tracking_ssp.json não encontrado — iniciando do zero.")
                return {}
            else:
                raise
        except json.JSONDecodeError:
            print("[Tracking] tracking_ssp.json corrompido — iniciando do zero.")
            return {}

    def salvar_tracking(self):
        self.s3.put_object(
            Bucket=self.r2_bucket,
            Key=self.r2_tracking_key,
            Body=json.dumps(self.dados, indent=2).encode("utf-8"),
            ContentType="application/json"
        )

    def atualizar_tracking(self, ano: int, tamanho_bytes: int, hash_sha256: str):
        self.dados[str(ano)] = {"tamanho_bytes": tamanho_bytes, "hash_sha256": hash_sha256}

    def precisa_processar(self, ano: int, tamanho_bytes_atual: int, hash_sha256_atual: str) -> bool:
        ano_str = str(ano)
        if ano_str not in self.dados:
            return True

        info_anterior = self.dados[ano_str]
        return (info_anterior.get("tamanho_bytes") != tamanho_bytes_atual or
                info_anterior.get("hash_sha256") != hash_sha256_atual)

class SafeDriver:
    def __init__(self):
        self.run_id = run_id_curto()
        self.t_inicio = time.time()
        self.discord = DiscordNotifier()
        self.df_raw = pl.DataFrame()
        self.anos_processados = []

        self.r2_bucket = sanitizar_secret(os.environ.get("R2_BUCKET_NAME", ""))
        r2_endpoint = sanitizar_secret(os.environ.get("R2_ENDPOINT_URL", ""))
        r2_key_id = sanitizar_secret(os.environ.get("R2_ACCESS_KEY_ID", ""))
        r2_access_key = sanitizar_secret(os.environ.get("R2_SECRET_ACCESS_KEY", ""))
        lgpd_salt = sanitizar_secret(os.environ.get("LGPD_SALT", ""))

        if not all([self.r2_bucket, r2_endpoint, r2_key_id, r2_access_key, lgpd_salt]):
            raise ValueError("Variáveis de ambiente R2_BUCKET_NAME, R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY e LGPD_SALT devem ser configuradas.")

        self.lgpd_salt = lgpd_salt

        self.s3 = boto3.client(
            "s3",
            endpoint_url=r2_endpoint,
            aws_access_key_id=r2_key_id,
            aws_secret_access_key=r2_access_key
        )
        self.tracking = TrackingSSP(self.s3, self.r2_bucket, R2_TRACKING)

        print(f"[{NOME_SISTEMA}] pipeline_iniciado")

    def sincronizar_raw(self):
        print("[sincronizar_raw_inicio]")
        df_raw_acumulado = pl.DataFrame()

        for ano in ANOS_DISPONIVEIS:
            r2_key_ano = f"{R2_PREFIXO_RAW}ssp_{ano}.parquet"
            df_ano = pl.DataFrame()
            baixado_da_ssp = False

            try:
                response = self.s3.get_object(Bucket=self.r2_bucket, Key=r2_key_ano)
                parquet_data = response["Body"].read()
                df_ano = pl.read_parquet(BytesIO(parquet_data))
                print(f"[R2] {r2_key_ano} encontrado no R2.")

                hash_atual = hashlib.sha256(parquet_data).hexdigest()
                tamanho_atual = len(parquet_data)

                if self.tracking.precisa_processar(ano, tamanho_atual, hash_atual):
                    print(f"[R2] {r2_key_ano} encontrado, mas alterado ou não rastreado — processando.")
                    self.anos_processados.append(ano)
                else:
                    print(f"[R2] {r2_key_ano} inalterado — ignorando processamento.")
                    continue

            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    print(f"[R2] {r2_key_ano} não existe — tentando baixar da SSP-SP.")
                    df_pandas_ssp = baixar_ssp(ano)
                    if df_pandas_ssp is not None and not df_pandas_ssp.empty:
                        # Converter para Polars e aplicar renomeação e padronização de colunas
                        df_ano = pl.from_pandas(df_pandas_ssp)
                        df_ano = renomear_sinonimos(df_ano)

                        # Garantir que todas as colunas padrão RAW existam, preenchendo com null
                        for col in COLUNAS_PADRAO_RAW:
                            if col not in df_ano.columns:
                                df_ano = df_ano.with_columns(pl.lit(None).alias(col).cast(pl.String))

                        # Selecionar e reordenar as colunas para garantir consistência
                        df_ano = df_ano.select(COLUNAS_PADRAO_RAW)

                        # Salvar no R2
                        buffer = BytesIO()
                        df_ano.write_parquet(buffer)
                        parquet_data = buffer.getvalue()
                        self.s3.put_object(
                            Bucket=self.r2_bucket,
                            Key=r2_key_ano,
                            Body=parquet_data,
                            ContentType="application/octet-stream"
                        )
                        print(f"[R2] {r2_key_ano} salvo no R2 após download da SSP-SP.")
                        hash_atual = hashlib.sha256(parquet_data).hexdigest()
                        tamanho_atual = len(parquet_data)
                        baixado_da_ssp = True
                        self.anos_processados.append(ano)
                    else:
                        print(f"[R2] Falha ao obter dados para o ano {ano} da SSP-SP. Ignorando.")
                        continue
                else:
                    raise

            if not df_ano.is_empty():
                if df_raw_acumulado.is_empty():
                    df_raw_acumulado = df_ano
                else:
                    # Antes de concatenar, garantir que as colunas são as mesmas
                    # Isso já deveria ter sido feito acima, mas é uma checagem final
                    if set(df_raw_acumulado.columns) != set(df_ano.columns):
                        print(f"[sincronizar_raw] Alerta: Colunas inconsistentes entre anos. Padronizando antes de concatenar.")
                        # Re-aplicar padronização caso haja alguma falha anterior
                        for col in COLUNAS_PADRAO_RAW:
                            if col not in df_raw_acumulado.columns:
                                df_raw_acumulado = df_raw_acumulado.with_columns(pl.lit(None).alias(col).cast(pl.String))
                            if col not in df_ano.columns:
                                df_ano = df_ano.with_columns(pl.lit(None).alias(col).cast(pl.String))
                        df_raw_acumulado = df_raw_acumulado.select(COLUNAS_PADRAO_RAW)
                        df_ano = df_ano.select(COLUNAS_PADRAO_RAW)

                    df_raw_acumulado = pl.concat([df_raw_acumulado, df_ano])

                if baixado_da_ssp:
                    self.tracking.atualizar_tracking(ano, tamanho_atual, hash_atual)
                    self.tracking.salvar_tracking()

        self.df_raw = df_raw_acumulado
        print(f"[sincronizar_raw] {len(self.df_raw)} registros RAW acumulados.")
        print("[sincronizar_raw_fim]")

    def construir_prata(self) -> pl.DataFrame:
        print("[construir_prata_inicio]")
        if self.df_raw.is_empty():
            print("[construir_prata] DataFrame RAW vazio. Pulando construção da camada prata.")
            return pl.DataFrame()

        df_prata = self.df_raw.clone()

        # Tipagem explícita e tratamento de nulos/inválidos
        df_prata = df_prata.with_columns([
            pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).alias("LATITUDE_F"),
            pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).alias("LONGITUDE_F"),
            pl.col("DATA_OCORRENCIA_BO").str.to_datetime("%d/%m/%Y", strict=False).alias("DATA_OCORRENCIA_DT"),
            pl.col("HORA_OCORRENCIA_BO").fill_null("00:00").fill_nan("00:00").alias("HORA_OCORRENCIA_STR"),
            pl.col("NOME_MUNICIPIO").fill_null("NAO_INFORMADO").fill_nan("NAO_INFORMADO").alias("NOME_MUNICIPIO_STR"),
            pl.col("LOGRADOURO").fill_null("NAO_INFORMADO").fill_nan("NAO_INFORMADO").alias("LOGRADOURO_STR"),
            pl.col("RUBRICA").fill_null("NAO_INFORMADO").fill_nan("NAO_INFORMADO").alias("RUBRICA_STR"),
        ])

        # Filtrar coordenadas inválidas (fora de SP ou 0,0)
        df_prata = df_prata.filter(
            (pl.col("LATITUDE_F").is_between(SP_LAT_MIN, SP_LAT_MAX)) &
            (pl.col("LONGITUDE_F").is_between(SP_LON_MIN, SP_LON_MAX)) &
            (~((pl.col("LATITUDE_F") == 0.0) & (pl.col("LONGITUDE_F") == 0.0)))
        )

        # Anonimização
        df_prata = df_prata.with_columns([
            pl.col("NOME_MUNICIPIO_STR").apply(lambda x: anonimizar_campo(x, self.lgpd_salt)).alias("NOME_MUNICIPIO_ANONIMIZADO"),
            pl.col("LOGRADOURO_STR").apply(lambda x: anonimizar_campo(x, self.lgpd_salt)).alias("LOGRADOURO_ANONIMIZADO"),
        ])

        # Geração de Features
        df_prata = df_prata.with_columns([
            pl.struct(["LATITUDE_F", "LONGITUDE_F"]).apply(lambda x: h3.geo_to_h3(x["LATITUDE_F"], x["LONGITUDE_F"], H3_RESOLUCAO)).alias("H3_INDEX"),
            pl.col("DATA_OCORRENCIA_DT").dt.year().alias("ANO"),
            pl.col("DATA_OCORRENCIA_DT").dt.month().alias("MES"),
            pl.col("DATA_OCORRENCIA_DT").dt.weekday().alias("DIA_SEMANA"),
            pl.col("HORA_OCORRENCIA_STR").apply(classificar_periodo).alias("PERIODO_DIA"),
            pl.col("PERIODO_DIA").is_in(["NOITE", "MADRUGADA"]).cast(pl.Int8).alias("IS_NOITE_MADRUGADA"),
            pl.col("RUBRICA_STR").apply(normalizar_texto).alias("RUBRICA_NORMALIZADA"),
        ])

        # Feriados
        feriados_sp = holidays.Brazil(state='SP', years=ANOS_DISPONIVEIS)
        df_prata = df_prata.with_columns(
            pl.col("DATA_OCORRENCIA_DT").apply(lambda x: x in feriados_sp).cast(pl.Int8).alias("IS_FERIADO")
        )

        # Classificação de tipo de crime (violência/patrimônio)
        crimes_violencia_pessoa = ["HOMICIDIO DOLOSO", "LATROCINIO", "EXTORSAO MEDIANTE SEQUESTRO", "ESTUPRO", "LESAO CORPORAL DOLOSA"]
        crimes_patrimonio = ["ROUBO DE VEICULO", "ROUBO DE MOTOCICLETA", "ROUBO DE CARGA", "FURTO DE VEICULO", "FURTO DE MOTOCICLETA", "ROUBO", "FURTO"]

        df_prata = df_prata.with_columns([
            pl.col("RUBRICA_NORMALIZADA").is_in(crimes_violencia_pessoa).cast(pl.Int8).alias("IS_VIOLENCIA_PESSOA"),
            pl.col("RUBRICA_NORMALIZADA").is_in(crimes_patrimonio).cast(pl.Int8).alias("IS_PATRIMONIO"),
        ])

        # Cálculo de escores por perfil
        df_prata = df_prata.with_columns([
            pl.struct(["RUBRICA_NORMALIZADA", "HORA_OCORRENCIA_STR"]).apply(
                lambda x: calcular_escores_todos_perfis(x["RUBRICA_NORMALIZADA"], x["HORA_OCORRENCIA_STR"])
            ).alias("ESCORES_PERFIL")
        ])
        df_prata = df_prata.unnest("ESCORES_PERFIL")

        # Selecionar colunas finais para a camada prata
        colunas_prata_finais = [
            "H3_INDEX", "ANO", "MES", "DIA_SEMANA", "PERIODO_DIA", "IS_NOITE_MADRUGADA",
            "LATITUDE_F", "LONGITUDE_F", "RUBRICA_NORMALIZADA", "NOME_MUNICIPIO_ANONIMIZADO",
            "LOGRADOURO_ANONIMIZADO", "IS_FERIADO", "IS_VIOLENCIA_PESSOA", "IS_PATRIMONIO",
            "ESCORE_MOTORISTA", "ESCORE_MOTOCICLISTA", "ESCORE_PEDESTRE", "ESCORE_CICLISTA"
        ]
        df_prata = df_prata.select(colunas_prata_finais)

        print(f"[construir_prata] {len(df_prata)} registros PRATA gerados.")
        print("[construir_prata_fim]")
        return df_prata

    def construir_ouro(self, df_prata: pl.DataFrame, bq_project: str, bq_dataset: str, bq_cred: str) -> tuple | None:
        print("[construir_ouro_inicio]")
        if df_prata.is_empty():
            print("[construir_ouro] DataFrame PRATA vazio. Pulando construção da camada ouro.")
            return None

        df_prata = df_prata.with_columns([
            (pl.col("ESCORE_MOTORISTA") + pl.col("ESCORE_MOTOCICLISTA") + pl.col("ESCORE_PEDESTRE") + pl.col("ESCORE_CICLISTA")).alias("ESCORE_TOTAL_OCORRENCIA")
        ])

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

        # Treinar o modelo final com todos os dados
        model.fit(X, y.values.ravel())

        # Predições para o DataFrame completo
        df_agregado = df_agregado.with_columns(pl.Series("ESCORE_PREDITO", model.predict(X)).cast(pl.Float64))

        # SHAP para explicabilidade
        explainer = shap.TreeExplainer(model.estimators_[0]) # Usar o primeiro estimador (LGBM) para SHAP
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list): # Para modelos multi-output, shap_values pode ser uma lista
            shap_values = np.array(shap_values).sum(axis=0) # Somar para obter um único array de importância

        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values(by='importance', ascending=False)

        top_shap_feature = feature_importances.iloc[0]['feature'] if not feature_importances.empty else "N/A"
        shap_top3 = feature_importances.head(3).to_dict('records')

        # Comparar com o último R2 salvo (se existir)
        mae_ant = 0.0
        melhoria = 0.0
        # Lógica para carregar R2 anterior e calcular melhoria (omitida para brevidade)

        # Salvar o modelo
        modelo_bytes = BytesIO()
        joblib.dump(model, modelo_bytes)
        modelo_bytes.seek(0)
        self.s3.put_object(
            Bucket=self.r2_bucket,
            Key=f"{R2_PREFIXO_MODELO}modelo_{hora_brasilia().strftime('%Y%m%d%H%M%S')}.pkl",
            Body=modelo_bytes.getvalue(),
            ContentType="application/octet-stream"
        )
        print("[modelo] Modelo salvo no R2.")

        # Selecionar e reordenar colunas para o BigQuery
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
