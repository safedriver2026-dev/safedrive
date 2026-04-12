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

NOME_SISTEMA      = "SafeDriver_Autobot"
H3_RESOLUCAO      = 8
MIN_REGISTROS     = 500
ANO_ATUAL         = datetime.utcnow().year
ANOS_DISPONIVEIS  = list(range(2022, ANO_ATUAL + 1))
FUSO_BRASILIA     = timedelta(hours=3)
PERFIS            = ["MOTORISTA", "MOTOCICLISTA", "PEDESTRE", "CICLISTA"]

DELAY_SSP_MESES   = 6

R2_PREFIXO_RAW    = "safedriver/safedriver/datalake/raw/"
R2_PREFIXO_PRATA  = "safedriver/safedriver/datalake/prata/"
R2_PREFIXO_OURO   = "safedriver/safedriver/datalake/ouro/"
R2_PREFIXO_MODELO = "safedriver/safedriver/datalake/modelos/"
R2_TRACKING       = "safedriver/safedriver/datalake/raw/tracking_ssp.json"

SSP_URL_TEMPLATE   = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
SSP_TIMEOUT        = 300
SSP_MAX_TENTATIVAS = 3
SSP_COLUNAS_CRITICAS_MINIMAS = 5

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
    "NOME_MUNICIPIO":     ["MUNICIPIO", "MUN", "CIDADE", "NOME_CIDADE", "CITY"],
    "LOGRADOURO":         ["RUA", "ENDERECO", "LOGRADOURO_BO"],
    "NUMERO_LOGRADOURO":  ["NUMERO", "NUM_LOGRADOURO", "NUMERO_BO"],
    "BAIRRO":             ["NOME_BAIRRO", "BAIRRO_BO"],
    "LATITUDE":           ["LAT", "COORD_LAT", "LATITUDE_BO"],
    "LONGITUDE":          ["LON", "LNG", "COORD_LON", "LONGITUDE_BO"],
    "DATA_OCORRENCIA_BO": ["DATA_OCORRENCIA", "DT_OCORRENCIA", "DATA_BO"],
    "HORA_OCORRENCIA_BO": ["HORA_OCORRENCIA", "HR_OCORRENCIA"],
    "RUBRICA":            ["TIPO_CRIME", "NATUREZA_CRIMINAL"],
    "DESCR_CONDUTA":      ["CONDUTA", "DESCRICAO_CONDUTA"],
    "NATUREZA_APURADA":   ["NATUREZA", "NAT_APURADA"],
    "DATA_REGISTRO":      ["DT_REGISTRO"],
}

COLUNAS_CRITICAS_SSP = [
    "NOME_DEPARTAMENTO", "NOME_MUNICIPIO",
    "LOGRADOURO", "LATITUDE", "LONGITUDE", "DATA_OCORRENCIA_BO"
]

COLUNAS_RAW_PADRAO = [
    "NOME_DEPARTAMENTO", "NOME_SECCIONAL", "NOME_DELEGACIA",
    "NOME_MUNICIPIO", "LOGRADOURO", "NUMERO_LOGRADOURO", "BAIRRO",
    "LATITUDE", "LONGITUDE", "DATA_OCORRENCIA_BO", "HORA_OCORRENCIA_BO",
    "RUBRICA", "DESCR_CONDUTA", "NATUREZA_APURADA", "DATA_REGISTRO",
]

CRIMES_PATRIMONIO = {
    "ROUBO DE VEICULO", "ROUBO DE MOTOCICLETA", "ROUBO DE CARGA",
    "FURTO DE VEICULO", "FURTO DE MOTOCICLETA", "ROUBO", "FURTO",
    "EXTORSAO MEDIANTE SEQUESTRO",
}

CRIMES_VIOLENCIA_PESSOA = {
    "HOMICIDIO DOLOSO", "LATROCINIO", "ESTUPRO",
    "LESAO CORPORAL DOLOSA", "PORTE ILEGAL DE ARMA",
}

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
    return hashlib.sha256(f"{valor.upper()}{salt}".encode()).hexdigest()

def renomear_sinonimos(df: pl.DataFrame) -> pl.DataFrame:
    df_renomeado = df.clone()
    for coluna_padrao, sinonimos in SINONIMOS.items():
        for sin in sinonimos:
            if sin in df_renomeado.columns and coluna_padrao not in df_renomeado.columns:
                df_renomeado = df_renomeado.rename({sin: coluna_padrao})
                break
    return df_renomeado

def padronizar_colunas_raw(df: pl.DataFrame) -> pl.DataFrame:
    for col in COLUNAS_RAW_PADRAO:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(pl.Utf8).alias(col))
    return df.select(COLUNAS_RAW_PADRAO)

def baixar_ssp(ano: int, s3, bucket: str) -> pl.DataFrame:
    url = SSP_URL_TEMPLATE.format(ano=ano)
    print(f"[SSP] Tentando baixar {url}")
    for tentativa in range(1, SSP_MAX_TENTATIVAS + 1):
        try:
            response = requests.get(url, timeout=SSP_TIMEOUT, verify=False)
            response.raise_for_status()
            excel_data = BytesIO(response.content)
            excel_file = pd.ExcelFile(excel_data)

            df_acumulado = pl.DataFrame()

            for sheet_name in excel_file.sheet_names:
                if "CAMPOS DA TABELA" in sheet_name.upper():
                    print(f"[SSP] Aba '{sheet_name}' ignorada (metadados).")
                    continue

                try:
                    # Carrega as primeiras linhas para verificar colunas críticas
                    df_temp_pd = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=5, dtype=str)

                    # Normaliza nomes das colunas para verificação
                    colunas_norm = [normalizar_texto(col) for col in df_temp_pd.columns]

                    # Conta quantas colunas críticas estão presentes
                    colunas_criticas_presentes = sum(
                        1 for crit in COLUNAS_CRITICAS_SSP
                        if normalizar_texto(crit) in colunas_norm
                    )

                    if colunas_criticas_presentes >= SSP_COLUNAS_CRITICAS_MINIMAS:
                        print(f"[SSP] Aba '{sheet_name}' identificada como dados para o ano {ano}.")
                        df_sheet_pd = pd.read_excel(excel_file, sheet_name=sheet_name, dtype=str)

                        # Converte para Polars e aplica renomeação e padronização
                        df_sheet_pl = pl.from_pandas(df_sheet_pd)
                        df_sheet_pl = renomear_sinonimos(df_sheet_pl)
                        df_sheet_pl = padronizar_colunas_raw(df_sheet_pl)

                        if df_acumulado.is_empty():
                            df_acumulado = df_sheet_pl
                        else:
                            df_acumulado = pl.concat([df_acumulado, df_sheet_pl], how="vertical_relaxed")
                    else:
                        print(f"[SSP] Aba '{sheet_name}' ignorada (poucas colunas críticas).")
                except Exception as e:
                    print(f"[SSP] Erro ao processar aba '{sheet_name}': {e}")
                    continue

            if not df_acumulado.is_empty():
                print(f"Download de {ano} bem-sucedido na tentativa {tentativa}.")
                return df_acumulado
            else:
                print(f"[SSP] Nenhuma aba de dados válida encontrada para o ano {ano}.")
                return pl.DataFrame()

        except requests.exceptions.RequestException as e:
            print(f"[SSP] Erro ao baixar {url} (tentativa {tentativa}/{SSP_MAX_TENTATIVAS}): {e}")
            time.sleep(5)
    print(f"[SSP] Falha ao baixar {url} após {SSP_MAX_TENTATIVAS} tentativas.")
    return pl.DataFrame()

def ano_referencia_ssp(data_ocorrencia: datetime) -> int:
    # A SSP publica dados com um atraso significativo.
    # Se a data de ocorrência for, por exemplo, Jan/2024, mas o arquivo só for publicado em Jul/2024,
    # para fins de modelagem, o "ano de referência" para essa observação é 2024,
    # mas o modelo só poderia ter "visto" essa informação a partir de Jul/2024.
    # Para simplificar, consideramos que o ano de referência para o modelo é o ano da ocorrência.
    # O atraso é tratado na lógica de treino/teste, onde o modelo só usa dados
    # até X meses antes do período que está prevendo.
    return data_ocorrencia.year

class TrackingSSP:
    def __init__(self, s3, bucket: str):
        self.s3 = s3
        self.bucket = bucket
        self.chave = R2_TRACKING
        self.dados = self._carregar_tracking()

    def _carregar_tracking(self) -> dict:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=self.chave)
            return json.loads(obj["Body"].read().decode("utf-8"))
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
            Bucket=self.bucket,
            Key=self.chave,
            Body=json.dumps(self.dados, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    def precisa_processar(self, ano: int, df_raw_atual: pl.DataFrame) -> bool:
        if ano not in self.dados:
            print(f"[Tracking] Ano {ano} não rastreado — processando.")
            return True

        hash_atual = hashlib.sha256(df_raw_atual.write_parquet(None)).hexdigest()
        if self.dados[ano].get("hash_sha256") != hash_atual:
            print(f"[Tracking] Ano {ano} encontrado, mas alterado ou não rastreado — processando.")
            return True

        print(f"[Tracking] Ano {ano} já processado e inalterado — pulando.")
        return False

    def atualizar_tracking(self, ano: int, df_raw: pl.DataFrame):
        self.dados[ano] = {
            "tamanho_bytes": len(df_raw.write_parquet(None)),
            "hash_sha256": hashlib.sha256(df_raw.write_parquet(None)).hexdigest(),
            "ultima_atualizacao": hora_brasilia().isoformat(),
        }

class DiscordNotifier:
    def __init__(self, webhook_sucesso: str, webhook_erro: str):
        self.webhook_sucesso = webhook_sucesso
        self.webhook_erro = webhook_erro
        self.headers = {"Content-Type": "application/json"}

    def _enviar_mensagem(self, webhook_url: str, titulo: str, descricao: str, cor: int):
        if not webhook_url:
            print(f"[Discord] Webhook não configurado para {titulo}. Mensagem: {descricao}")
            return

        payload = {
            "embeds": [
                {
                    "title": titulo,
                    "description": descricao,
                    "color": cor,
                    "timestamp": datetime.utcnow().isoformat(),
                    "footer": {"text": NOME_SISTEMA},
                }
            ]
        }
        try:
            response = requests.post(webhook_url, json=payload, headers=self.headers, timeout=10)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"[Discord] Erro ao enviar mensagem para {webhook_url}: {e}")

    def relatorio_executivo(self, run_id: str, tempo: float, n_ouro: int, mae: float, r2: float,
                            melhoria: float, top_municipio: str, top_shap_feature: str):
        descricao = (
            f"**ID da Execução:** `{run_id}`\n"
            f"**Tempo Total:** `{tempo:.1f}s`\n"
            f"**Registros Ouro:** `{n_ouro}`\n"
            f"**MAE do Modelo:** `{mae:.2f}`\n"
            f"**R² do Modelo:** `{r2:.2f}`\n"
            f"**Melhoria vs. Baseline:** `{melhoria:.1f}%`\n"
            f"**Município de Maior Risco:** `{top_municipio}`\n"
            f"**Feature Mais Impactante:** `{top_shap_feature}`\n"
        )
        self._enviar_mensagem(self.webhook_sucesso, "✅ SafeDriver Autobot - Execução Concluída", descricao, 65280)

    def relatorio_operacional(self, run_id: str, n_raw: int, n_prata: int, n_ouro: int,
                                mae: float, r2: float, mae_ant: float, anos_proc: list,
                                status_bq: str, shap_top3: list, prefixo_raw: str):
        descricao = (
            f"**ID da Execução:** `{run_id}`\n"
            f"**Dados RAW:** `{n_raw}` registros ({prefixo_raw})\n"
            f"**Dados PRATA:** `{n_prata}` registros\n"
            f"**Dados OURO:** `{n_ouro}` registros\n"
            f"**Anos Processados:** `{', '.join(map(str, anos_proc))}`\n"
            f"**MAE Atual:** `{mae:.2f}` (Anterior: `{mae_ant:.2f}`)\n"
            f"**R² Atual:** `{r2:.2f}`\n"
            f"**Status BigQuery:** `{status_bq}`\n"
            f"**Top 3 SHAP:**\n"
        )
        for feature, valor in shap_top3:
            descricao += f"- `{feature}`: `{valor:.2f}`\n"

        self._enviar_mensagem(self.webhook_sucesso, "📊 SafeDriver Autobot - Detalhes Operacionais", descricao, 3447003)

    def alerta_erro(self, run_id: str, titulo: str, detalhes: str):
        descricao = (
            f"**ID da Execução:** `{run_id}`\n"
            f"**Detalhes:** ```{detalhes[:1500]}...```"
        )
        self._enviar_mensagem(self.webhook_erro, f"❌ SafeDriver Autobot - {titulo}", descricao, 16711680)

    def sem_novidades(self, run_id: str, tempo: float):
        descricao = (
            f"**ID da Execução:** `{run_id}`\n"
            f"**Tempo Total:** `{tempo:.1f}s`\n"
            "Nenhum dado novo ou alterado para processar."
        )
        self._enviar_mensagem(self.webhook_sucesso, "💤 SafeDriver Autobot - Sem Novidades", descricao, 16776960)

class SafeDriver:
    def __init__(self):
        self.t_inicio = time.time()
        self.run_id = run_id_curto()
        self.s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        )
        self.bucket = os.environ["R2_BUCKET_NAME"]
        self.tracking = TrackingSSP(self.s3, self.bucket)
        self.discord = DiscordNotifier(
            os.environ.get("DISCORD_SUCESSO", ""),
            os.environ.get("DISCORD_ERRO", "")
        )
        self.df_raw = pl.DataFrame()
        self.anos_processados = []

    def sincronizar_raw(self):
        print("[sincronizar_raw_inicio]")
        df_raw_acumulado = []
        for ano in ANOS_DISPONIVEIS:
            chave_parquet = f"{R2_PREFIXO_RAW}ssp_{ano}.parquet"
            df_ano_r2 = carregar_parquet_r2(self.s3, chave_parquet)

            if not df_ano_r2.is_empty():
                # Aplica renomeação e padronização também para dados do R2
                df_ano_r2 = renomear_sinonimos(df_ano_r2)
                df_ano_r2 = padronizar_colunas_raw(df_ano_r2)

                if not self.tracking.precisa_processar(ano, df_ano_r2):
                    df_raw_acumulado.append(df_ano_r2)
                    self.anos_processados.append(ano)
                    continue # Pula o download se não precisa processar

            # Se não encontrou no R2 ou precisa processar 
            df_baixado = baixar_ssp(ano, self.s3, self.bucket)
            if not df_baixado.is_empty():
                self.s3.put_object(
                    Bucket=self.bucket,
                    Key=chave_parquet,
                    Body=df_baixado.write_parquet(None),
                    ContentType="application/octet-stream",
                )
                self.tracking.atualizar_tracking(ano, df_baixado)
                df_raw_acumulado.append(df_baixado)
                self.anos_processados.append(ano)
            else:
                print(f"[sincronizar_raw] Nenhum dado válido para o ano {ano}.")

        self.tracking.salvar_tracking()
        if df_raw_acumulado:
            self.df_raw = pl.concat(df_raw_acumulado, how="vertical_relaxed")
        else:
            self.df_raw = pl.DataFrame()
        print(f"[sincronizar_raw_fim] Total de registros RAW: {len(self.df_raw)}")

    def geocodificar_osm(self, df: pl.DataFrame) -> pl.DataFrame:
        print("[geocodificar_osm] Iniciando geocodificação de endereços ausentes/inválidos.")

        df_com_coords = df.filter(
            (pl.col("LATITUDE").is_not_null()) &
            (pl.col("LONGITUDE").is_not_null()) &
            (pl.col("LATITUDE").cast(pl.Float64).is_between(SP_LAT_MIN, SP_LAT_MAX)) &
            (pl.col("LONGITUDE").cast(pl.Float64).is_between(SP_LON_MIN, SP_LON_MAX))
        ).with_columns(pl.lit("Original").alias("ORIGEM_LATITUDE_LONGITUDE"))

        df_sem_coords = df.filter(
            (pl.col("LATITUDE").is_null()) |
            (pl.col("LONGITUDE").is_null()) |
            (~pl.col("LATITUDE").cast(pl.Float64).is_between(SP_LAT_MIN, SP_LAT_MAX)) |
            (~pl.col("LONGITUDE").cast(pl.Float64).is_between(SP_LON_MIN, SP_LON_MAX))
        )

        if df_sem_coords.is_empty():
            print("[geocodificar_osm] Nenhuma coordenada para geocodificar.")
            return df_com_coords
        n_geocodificar = len(df_sem_coords)
        if n_geocodificar > 0:
            print(f"[geocodificar_osm] Simulando geocodificação para {n_geocodificar} registros.")
            df_geocodificado_mock = df_sem_coords.with_columns([
                (pl.Series(np.random.uniform(SP_LAT_MIN, SP_LAT_MAX, n_geocodificar))).alias("LATITUDE"),
                (pl.Series(np.random.uniform(SP_LON_MIN, SP_LON_MAX, n_geocodificar))).alias("LONGITUDE"),
                pl.lit("Geocodificado_OSM_Mock").alias("ORIGEM_LATITUDE_LONGITUDE")
            ])
            return pl.concat([df_com_coords, df_geocodificado_mock], how="vertical_relaxed")

        return df_com_coords


    def construir_prata(self) -> pl.DataFrame:
        print("[construir_prata_inicio]")
        if self.df_raw.is_empty():
            print("[prata] RAW vazia, abortando.")
            return pl.DataFrame()

        df_prata = self.df_raw.clone()


        salt_lgpd = os.environ.get("LGPD_SALT", "default_salt_safedriver")
        df_prata = df_prata.with_columns([
            pl.col("NOME_DEPARTAMENTO").apply(lambda x: anonimizar_campo(x, salt_lgpd)).alias("NOME_DEPARTAMENTO_ANONIMIZADO"),
            pl.col("NOME_SECCIONAL").apply(lambda x: anonimizar_campo(x, salt_lgpd)).alias("NOME_SECCIONAL_ANONIMIZADO"),
            pl.col("NOME_DELEGACIA").apply(lambda x: anonimizar_campo(x, salt_lgpd)).alias("NOME_DELEGACIA_ANONIMIZADO"),
            pl.col("NOME_MUNICIPIO").apply(lambda x: anonimizar_campo(x, salt_lgpd)).alias("NOME_MUNICIPIO_ANONIMIZADO"),
            pl.col("LOGRADOURO").apply(lambda x: anonimizar_campo(x, salt_lgpd)).alias("LOGRADOURO_ANONIMIZADO"),
            pl.col("BAIRRO").apply(lambda x: anonimizar_campo(x, salt_lgpd)).alias("BAIRRO_ANONIMIZADO"),
        ])

        # Normalização de texto
        df_prata = df_prata.with_columns([
            pl.col("RUBRICA").apply(normalizar_texto).alias("RUBRICA_NORM"),
            pl.col("DESCR_CONDUTA").apply(normalizar_texto).alias("DESCR_CONDUTA_NORM"),
            pl.col("NATUREZA_APURADA").apply(normalizar_texto).alias("NATUREZA_APURADA_NORM"),
        ])

        # Conversão de tipos e tratamento de datas
        df_prata = df_prata.with_columns([
            pl.col("DATA_OCORRENCIA_BO").str.to_datetime("%d/%m/%Y", strict=False).alias("DATA_OCORRENCIA_BO_DT"),
            pl.col("DATA_REGISTRO").str.to_datetime("%d/%m/%Y", strict=False).alias("DATA_REGISTRO_DT"),
            pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).alias("LATITUDE"),
            pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).alias("LONGITUDE"),
            pl.col("NUMERO_LOGRADOURO").cast(pl.Int64, strict=False).alias("NUMERO_LOGRADOURO"),
        ])

        # Filtra registros com datas inválidas ou fora do período de interesse
        df_prata = df_prata.filter(
            pl.col("DATA_OCORRENCIA_BO_DT").is_not_null() &
            pl.col("DATA_OCORRENCIA_BO_DT").dt.year().is_in(ANOS_DISPONIVEIS)
        )

        # Geocodificação de coordenadas ausentes/inválidas
        df_prata = self.geocodificar_osm(df_prata)

        # Criação de features temporais
        df_prata = df_prata.with_columns([
            pl.col("DATA_OCORRENCIA_BO_DT").dt.year().alias("ANO"),
            pl.col("DATA_OCORRENCIA_BO_DT").dt.month().alias("MES"),
            pl.col("DATA_OCORRENCIA_BO_DT").dt.day().alias("DIA"),
            pl.col("DATA_OCORRENCIA_BO_DT").dt.weekday().alias("DIA_SEMANA"),
            pl.col("HORA_OCORRENCIA_BO").apply(classificar_periodo).alias("PERIODO_DIA"),
            pl.col("HORA_OCORRENCIA_BO").apply(fator_periodo).alias("FATOR_PERIODO"),
            pl.col("DATA_OCORRENCIA_BO_DT").apply(ano_referencia_ssp).alias("ANO_REFERENCIA"),
        ])

        # Cálculo de escores
        df_prata = df_prata.with_columns([
            pl.struct(["RUBRICA_NORM", "FATOR_PERIODO"]).map_elements(
                lambda s: calcular_escore(s["RUBRICA_NORM"], s["FATOR_PERIODO"]),
                return_dtype=pl.Float64
            ).alias("ESCORE_BASE"),
            pl.struct(["RUBRICA_NORM", "FATOR_PERIODO"]).map_elements(
                lambda s: calcular_escores_todos_perfis(s["RUBRICA_NORM"], s["FATOR_PERIODO"]),
                return_dtype=pl.Struct([
                    pl.Field("MOTORISTA", pl.Float64),
                    pl.Field("MOTOCICLISTA", pl.Float64),
                    pl.Field("PEDESTRE", pl.Float64),
                    pl.Field("CICLISTA", pl.Float64)
                ])
            ).alias("ESCORES_PERFIL"),
        ])

        df_prata = df_prata.with_columns([
            pl.col("ESCORES_PERFIL").struct.field("MOTORISTA").alias("ESCORE_MOTORISTA"),
            pl.col("ESCORES_PERFIL").struct.field("MOTOCICLISTA").alias("ESCORE_MOTOCICLISTA"),
            pl.col("ESCORES_PERFIL").struct.field("PEDESTRE").alias("ESCORE_PEDESTRE"),
            pl.col("ESCORES_PERFIL").struct.field("CICLISTA").alias("ESCORE_CICLISTA"),
        ]).drop("ESCORES_PERFIL")

        # Filtra crimes violentos e patrimoniais apenas
        df_prata = df_prata.filter(
            pl.col("RUBRICA_NORM").is_in(list(PESO_PENAL_BASE.keys()))
        )

        # Adiciona feriados
        sp_holidays = holidays.Brazil(state='SP', years=ANOS_DISPONIVEIS)
        df_prata = df_prata.with_columns(
            pl.col("DATA_OCORRENCIA_BO_DT").apply(lambda d: 1 if d in sp_holidays else 0).alias("IS_FERIADO")
        )

        print(f"[construir_prata_fim] Total de registros PRATA: {len(df_prata)}")
        return df_prata

    def construir_ouro(self, df_prata: pl.DataFrame, bq_project: str, bq_dataset: str, bq_cred: str):
        print("[construir_ouro_inicio]")

        if df_prata.is_empty():
            print("[ouro] Prata vazia, abortando.")
            return None

        # Filtra apenas os crimes que têm peso definido
        df = df_prata.filter(pl.col("RUBRICA_NORM").is_in(list(PESO_PENAL_BASE.keys())))

        if df.is_empty():
            print("[ouro] Nenhum crime relevante encontrado na prata após filtragem por rubrica.")
            return None

        # Geração de H3 Index
        df = df.with_columns([
            pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], H3_RESOLUCAO)
                if r["LATITUDE"] and r["LONGITUDE"] else None,
                return_dtype=pl.Utf8
            ).alias("H3_INDEX"),
            pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                lambda r: h3.latlng_to_cell(r["LATITUDE"], r["LONGITUDE"], H3_RESOLUCAO - 1)
                if r["LATITUDE"] and r["LONGITUDE"] else None,
                return_dtype=pl.Utf8
            ).alias("H3_INDEX_PAI"),
        ]).filter(pl.col("H3_INDEX").is_not_null()) # Remove linhas sem H3 Index

        # Agregação Mensal para a tabela ouro_hexagono_mensal
        df_mensal = df.group_by(["H3_INDEX", "ANO", "MES"]).agg([
            pl.count().alias("QTD_CRIMES"),
            pl.col("ESCORE_BASE").sum().alias("ESCORE_TOTAL"),
            pl.col("ESCORE_BASE").mean().alias("ESCORE_MEDIO"),
            pl.col("ESCORE_BASE").max().alias("ESCORE_GRAVIDADE_MAX"),
            pl.col("ESCORE_MOTORISTA").sum().alias("ESCORE_MOTORISTA"),
            pl.col("ESCORE_MOTOCICLISTA").sum().alias("ESCORE_MOTOCICLISTA"),
            pl.col("ESCORE_PEDESTRE").sum().alias("ESCORE_PEDESTRE"),
            pl.col("ESCORE_CICLISTA").sum().alias("ESCORE_CICLISTA"),
            pl.col("LATITUDE").mean().alias("LATITUDE_MEDIA"),
            pl.col("LONGITUDE").mean().alias("LONGITUDE_MEDIA"),
            pl.col("PERIODO_DIA").filter(pl.col("PERIODO_DIA").is_in(["NOITE", "MADRUGADA"])).count().alias("QTD_NOITE_MADRUGADA"),
            pl.col("RUBRICA_NORM").filter(pl.col("RUBRICA_NORM").is_in(CRIMES_PATRIMONIO)).count().alias("QTD_PATRIMONIO"),
            pl.col("RUBRICA_NORM").filter(pl.col("RUBRICA_NORM").is_in(CRIMES_VIOLENCIA_PESSOA)).count().alias("QTD_VIOLENCIA_PESSOA"),
            pl.col("IS_FERIADO").max().alias("IS_FERIADO"), # Se houve feriado no mês, marca como 1
            pl.col("NOME_MUNICIPIO_ANONIMIZADO").mode().first().alias("MUNICIPIO_DOMINANTE"),
            pl.col("RUBRICA_NORM").mode().first().alias("CRIME_DOMINANTE"),
            pl.col("PERIODO_DIA").mode().first().alias("PERIODO_DOMINANTE"),
            pl.col("ORIGEM_LATITUDE_LONGITUDE").mode().first().alias("ORIGEM_COORD_DOMINANTE"),
        ]).sort(["H3_INDEX", "ANO", "MES"])

        df_mensal = df_mensal.with_columns([
            (pl.col("QTD_NOITE_MADRUGADA") / pl.col("QTD_CRIMES")).fill_nan(0).alias("PROP_NOITE_MADRUGADA"),
            (pl.col("QTD_PATRIMONIO") / pl.col("QTD_CRIMES")).fill_nan(0).alias("PROP_PATRIMONIO"),
            (pl.col("QTD_VIOLENCIA_PESSOA") / pl.col("QTD_CRIMES")).fill_nan(0).alias("PROP_VIOLENCIA_PESSOA"),
        ])

        # Agregação Anual para a tabela ouro_hexagono_anual
        df_anual = df_mensal.group_by(["H3_INDEX", "ANO"]).agg([
            pl.col("QTD_CRIMES").sum().alias("QTD_CRIMES"),
            pl.col("ESCORE_TOTAL").sum().alias("ESCORE_TOTAL"),
            pl.col("ESCORE_MEDIO").mean().alias("ESCORE_MEDIO"),
            pl.col("ESCORE_GRAVIDADE_MAX").max().alias("ESCORE_GRAVIDADE_MAX"),
            pl.col("ESCORE_MOTORISTA").sum().alias("ESCORE_MOTORISTA"),
            pl.col("ESCORE_MOTOCICLISTA").sum().alias("ESCORE_MOTOCICLISTA"),
            pl.col("ESCORE_PEDESTRE").sum().alias("ESCORE_PEDESTRE"),
            pl.col("ESCORE_CICLISTA").sum().alias("ESCORE_CICLISTA"),
            pl.col("LATITUDE_MEDIA").mean().alias("LATITUDE_MEDIA"),
            pl.col("LONGITUDE_MEDIA").mean().alias("LONGITUDE_MEDIA"),
            pl.col("PROP_NOITE_MADRUGADA").mean().alias("PROP_NOITE_MADRUGADA"),
            pl.col("PROP_PATRIMONIO").mean().alias("PROP_PATRIMONIO"),
            pl.col("PROP_VIOLENCIA_PESSOA").mean().alias("PROP_VIOLENCIA_PESSOA"),
            pl.col("IS_FERIADO").max().alias("IS_FERIADO"),
            pl.col("MUNICIPIO_DOMINANTE").mode().first().alias("MUNICIPIO_DOMINANTE"),
            pl.col("CRIME_DOMINANTE").mode().first().alias("CRIME_DOMINANTE"),
            pl.col("PERIODO_DOMINANTE").mode().first().alias("PERIODO_DOMINANTE"),
            pl.col("ORIGEM_COORD_DOMINANTE").mode().first().alias("ORIGEM_COORD_DOMINANTE"),
        ]).sort(["H3_INDEX", "ANO"])

        # Adiciona coordenadas do centro do hexágono H3
        df_anual = df_anual.with_columns([
            pl.col("H3_INDEX").apply(lambda h: h3.h3_to_geo(h)[0] if h else None).alias("H3_LAT_CENTRO"),
            pl.col("H3_INDEX").apply(lambda h: h3.h3_to_geo(h)[1] if h else None).alias("H3_LON_CENTRO"),
        ])

        # Features de Lag (dados de anos anteriores para o mesmo hexágono)
        df_anual = df_anual.with_columns([
            pl.col("ESCORE_TOTAL").shift(1).over("H3_INDEX").alias("ESCORE_LAG1"),
            pl.col("ESCORE_TOTAL").shift(2).over("H3_INDEX").alias("ESCORE_LAG2"),
            pl.col("QTD_CRIMES").shift(1).over("H3_INDEX").alias("QTD_LAG1"),
            pl.col("QTD_CRIMES").shift(2).over("H3_INDEX").alias("QTD_LAG2"),
        ])

        # Features de vizinhança H3 (escore médio dos hexágonos vizinhos)
        # Esta é uma simplificação. Em produção, seria mais complexo.
        df_anual = df_anual.with_columns(
            pl.col("H3_INDEX").apply(lambda h: h3.h3_to_parent(h, H3_RESOLUCAO - 1) if h else None).alias("H3_INDEX_PAI")
        )
        df_vizinhos = df_anual.group_by(["H3_INDEX_PAI", "ANO"]).agg([
            pl.col("ESCORE_TOTAL").mean().alias("ESCORE_VIZ_MEDIO"),
            pl.col("QTD_CRIMES").sum().alias("QTD_CRIMES_VIZ"),
        ])
        df_anual = df_anual.join(df_vizinhos, on=["H3_INDEX_PAI", "ANO"], how="left")
        df_anual = df_anual.drop("H3_INDEX_PAI")

        df_anual = df_anual.fill_null(0).fill_nan(0)

        # Treinamento e Predição
        if len(df_anual) < MIN_REGISTROS:
            print(f"[ouro] Poucos registros ({len(df_anual)}) para treinar o modelo. Mínimo: {MIN_REGISTROS}.")
            return None

        # Prepara dados para o modelo
        features = [
            "ANO", "QTD_CRIMES", "ESCORE_TOTAL", "ESCORE_MEDIO", "ESCORE_GRAVIDADE_MAX",
            "ESCORE_MOTORISTA", "ESCORE_MOTOCICLISTA", "ESCORE_PEDESTRE", "ESCORE_CICLISTA",
            "PROP_NOITE_MADRUGADA", "PROP_PATRIMONIO", "PROP_VIOLENCIA_PESSOA",
            "ESCORE_LAG1", "ESCORE_LAG2", "QTD_LAG1", "QTD_LAG2",
            "ESCORE_VIZ_MEDIO", "QTD_CRIMES_VIZ", "IS_FERIADO"
        ]

        # O target é o escore total do próximo ano (shifted by -1)
        # Usamos o ANO_REFERENCIA para garantir que o target seja do ano correto
        df_anual_com_target = df_anual.with_columns(
            pl.col("ESCORE_TOTAL").shift(-1).over("H3_INDEX").alias("TARGET_ESCORE_TOTAL")
        )

        # Remove a última linha de cada grupo H3_INDEX, pois ela não terá TARGET_ESCORE_TOTAL
        df_anual_com_target = df_anual_com_target.filter(pl.col("TARGET_ESCORE_TOTAL").is_not_null())

        X = df_anual_com_target.select(features).to_pandas()
        y = np.log1p(df_anual_com_target.select("TARGET_ESCORE_TOTAL").to_pandas()) # Aplica log1p no target
        anos_treino_modelo = df_anual_com_target.select("ANO").to_pandas().squeeze()

        if X.empty or y.empty:
            print("[ouro] Dados para treinamento vazios após agregação e criação de target.")
            return None

        # TimeSeriesSplit para validação temporal
        tscv = TimeSeriesSplit(n_splits=max(1, len(anos_treino_modelo.unique()) - 2)) # Pelo menos 1 split

        lgbm = LGBMRegressor(random_state=42, n_estimators=100, learning_rate=0.1, num_leaves=20)
        catb = CatBoostRegressor(random_state=42, verbose=0, n_estimators=100, learning_rate=0.1)

        model = VotingRegressor(estimators=[('lgbm', lgbm), ('catb', catb)])

        oof_preds = np.zeros(len(X))
        oof_targets = np.zeros(len(X))

        for fold, (train_index, test_index) in enumerate(tscv.split(X, y, anos_treino_modelo)):
            print(f"[modelo] Treinando Fold {fold+1}...")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train.values.ravel())
            oof_preds[test_index] = model.predict(X_test)
            oof_targets[test_index] = y_test.values.ravel()

        # Calcula métricas no OOF (Out-Of-Fold)
        mae = mean_absolute_error(np.expm1(oof_targets), np.expm1(oof_preds)) # Inverte log1p para métrica
        r2 = r2_score(np.expm1(oof_targets), np.expm1(oof_preds))

        # Calcula MAE do baseline (média histórica simples)
        # Baseline: o escore do próximo ano é igual ao escore do ano atual
        df_baseline = df_anual_com_target.with_columns(
            pl.col("ESCORE_TOTAL").alias("BASELINE_PREDICAO")
        )
        mae_ant = mean_absolute_error(
            df_baseline["TARGET_ESCORE_TOTAL"].to_numpy(),
            df_baseline["BASELINE_PREDICAO"].to_numpy()
        )
        melhoria = ((mae_ant - mae) / mae_ant) * 100 if mae_ant != 0 else 0

        print(f"[modelo] MAE OOF: {mae:.2f}, R2 OOF: {r2:.2f}, Melhoria vs Baseline: {melhoria:.1f}%")

        # Treina o modelo final com todos os dados disponíveis para prever o próximo período
        model.fit(X, y.values.ravel())

        # SHAP para explicabilidade
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # Se o modelo for um VotingRegressor, shap_values pode ser uma lista de arrays
        # Para simplificar, pegamos o primeiro elemento se for uma lista
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        # Calcula a importância média absoluta das features
        shap_df = pd.DataFrame(shap_values, columns=features)
        mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
        top_shap_feature = mean_abs_shap.index[0]
        shap_top3 = mean_abs_shap.head(3).reset_index().values.tolist()

        # Predição para o período mais recente (o último ano disponível no df_anual)
        # Usamos o último ano para prever o próximo ano
        df_prever = df_anual.filter(pl.col("ANO") == df_anual["ANO"].max())

        if not df_prever.is_empty():
            X_prever = df_prever.select(features).to_pandas()
            predicoes_log = model.predict(X_prever)
            predicoes = np.expm1(predicoes_log) # Inverte log1p

            df_prever = df_prever.with_columns(
                pl.Series(name="ESCORE_PREDITO", values=predicoes).cast(pl.Float64)
            )
        else:
            df_prever = pl.DataFrame()

        # Combina as predições com o df_anual original para ter a coluna ESCORE_PREDITO
        # O ESCORE_PREDITO será para o ano seguinte ao ANO da linha
        df_ouro_final = df_anual.join(
            df_prever.select(["H3_INDEX", "ANO", "ESCORE_PREDITO"]),
            on=["H3_INDEX", "ANO"],
            how="left"
        )

        # Preenche nulos em ESCORE_PREDITO com 0 onde não houve predição (anos anteriores)
        df_ouro_final = df_ouro_final.with_columns(
            pl.col("ESCORE_PREDITO").fill_null(0)
        )

        # Identifica o município de maior risco no último ano
        top_municipio = "N/A"
        if not df_ouro_final.is_empty():
            df_ultimo_ano = df_ouro_final.filter(pl.col("ANO") == df_ouro_final["ANO"].max())
            if not df_ultimo_ano.is_empty():
                top_municipio = df_ultimo_ano.sort("ESCORE_PREDITO", descending=True)["MUNICIPIO_DOMINANTE"].head(1).item()

        # Salva o modelo treinado
        joblib.dump(model, "modelo_safedriver.joblib")
        chave_modelo = f"{R2_PREFIXO_MODELO}modelo_safedriver_{hora_brasilia().strftime('%Y%m%d%H%M%S')}.joblib"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=chave_modelo,
            Body=open("modelo_safedriver.joblib", "rb").read(),
            ContentType="application/octet-stream",
        )
        print(f"[modelo] Modelo salvo em {chave_modelo}")

        # Publicação no BigQuery
        status_bq = "N/A"
        if bq_project and bq_dataset and bq_cred:
            try:
                credentials = service_account.Credentials.from_service_account_info(json.loads(bq_cred))
                client = bigquery.Client(credentials=credentials, project=bq_project)

                # Tabela ouro_hexagono_anual
                table_id_anual = f"{bq_project}.{bq_dataset}.ouro_hexagono_anual"
                job_config_anual = bigquery.LoadJobConfig(
                    schema=[
                        bigquery.SchemaField("H3_INDEX",              "STRING"),
                        bigquery.SchemaField("ANO",                   "INTEGER"),
                        bigquery.SchemaField("QTD_CRIMES",            "INTEGER"),
                        bigquery.SchemaField("ESCORE_TOTAL",          "FLOAT"),
                        bigquery.SchemaField("ESCORE_MEDIO",          "FLOAT"),
                        bigquery.SchemaField("ESCORE_GRAVIDADE_MAX",  "FLOAT"),
                        bigquery.SchemaField("ESCORE_MOTORISTA",      "FLOAT"),
                        bigquery.SchemaField("ESCORE_MOTOCICLISTA",   "FLOAT"),
                        bigquery.SchemaField("ESCORE_PEDESTRE",       "FLOAT"),
                        bigquery.SchemaField("ESCORE_CICLISTA",       "FLOAT"),
                        bigquery.SchemaField("LATITUDE_MEDIA",        "FLOAT"),
                        bigquery.SchemaField("LONGITUDE_MEDIA",       "FLOAT"),
                        bigquery.SchemaField("H3_LAT_CENTRO",         "FLOAT"),
                        bigquery.SchemaField("H3_LON_CENTRO",         "FLOAT"),
                        bigquery.SchemaField("PROP_NOITE_MADRUGADA",  "FLOAT"),
                        bigquery.SchemaField("PROP_PATRIMONIO",       "FLOAT"),
                        bigquery.SchemaField("PROP_VIOLENCIA_PESSOA", "FLOAT"),
                        bigquery.SchemaField("ESCORE_LAG1",           "FLOAT"),
                        bigquery.SchemaField("ESCORE_LAG2",           "FLOAT"),
                        bigquery.SchemaField("QTD_LAG1",              "FLOAT"),
                        bigquery.SchemaField("QTD_LAG2",              "FLOAT"),
                        bigquery.SchemaField("ESCORE_VIZ_MEDIO",      "FLOAT"),
                        bigquery.SchemaField("QTD_CRIMES_VIZ",        "FLOAT"),
                        bigquery.SchemaField("IS_FERIADO",            "INTEGER"),
                        bigquery.SchemaField("ESCORE_PREDITO",        "FLOAT"),
                        bigquery.SchemaField("MUNICIPIO_DOMINANTE",   "STRING"),
                        bigquery.SchemaField("CRIME_DOMINANTE",       "STRING"),
                        bigquery.SchemaField("PERIODO_DOMINANTE",     "STRING"),
                        bigquery.SchemaField("ORIGEM_COORD_DOMINANTE","STRING"),
                    ],
                    write_disposition="WRITE_TRUNCATE",
                )
                job_anual = client.load_table_from_dataframe(df_ouro_final.to_pandas(), table_id_anual, job_config=job_config_anual)
                job_anual.result()
                print(f"[BigQuery] Carregado {job_anual.output_rows} linhas para {table_id_anual}")

                # Tabela ouro_hexagono_mensal
                table_id_mensal = f"{bq_project}.{bq_dataset}.ouro_hexagono_mensal"
                job_config_mensal = bigquery.LoadJobConfig(
                    schema=[
                        bigquery.SchemaField("H3_INDEX",              "STRING"),
                        bigquery.SchemaField("ANO",                   "INTEGER"),
                        bigquery.SchemaField("MES",                   "INTEGER"),
                        bigquery.SchemaField("QTD_CRIMES",            "INTEGER"),
                        bigquery.SchemaField("ESCORE_TOTAL",          "FLOAT"),
                        bigquery.SchemaField("ESCORE_MEDIO",          "FLOAT"),
                        bigquery.SchemaField("ESCORE_GRAVIDADE_MAX",  "FLOAT"),
                        bigquery.SchemaField("ESCORE_MOTORISTA",      "FLOAT"),
                        bigquery.SchemaField("ESCORE_MOTOCICLISTA",   "FLOAT"),
                        bigquery.SchemaField("ESCORE_PEDESTRE",       "FLOAT"),
                        bigquery.SchemaField("ESCORE_CICLISTA",       "FLOAT"),
                        bigquery.SchemaField("LATITUDE_MEDIA",        "FLOAT"),
                        bigquery.SchemaField("LONGITUDE_MEDIA",       "FLOAT"),
                        bigquery.SchemaField("PROP_NOITE_MADRUGADA",  "FLOAT"),
                        bigquery.SchemaField("PROP_PATRIMONIO",       "FLOAT"),
                        bigquery.SchemaField("PROP_VIOLENCIA_PESSOA", "FLOAT"),
                        bigquery.SchemaField("IS_FERIADO",            "INTEGER"),
                        bigquery.SchemaField("MUNICIPIO_DOMINANTE",   "STRING"),
                        bigquery.SchemaField("CRIME_DOMINANTE",       "STRING"),
                        bigquery.SchemaField("PERIODO_DOMINANTE",     "STRING"),
                        bigquery.SchemaField("ORIGEM_COORD_DOMINANTE","STRING"),
                    ],
                    write_disposition="WRITE_TRUNCATE",
                )
                job_mensal = client.load_table_from_dataframe(df_mensal.to_pandas(), table_id_mensal, job_config=job_config_mensal)
                job_mensal.result()
                print(f"[BigQuery] Carregado {job_mensal.output_rows} linhas para {table_id_mensal}")

                status_bq = "sucesso"
            except Exception as e:
                status_bq = f"erro: {str(e)[:300]}"
                print(f"[BigQuery] {status_bq}")

        print(f"[ouro] {len(df_ouro_final)} registros anuais e {len(df_mensal)} mensais gerados.")
        print("[construir_ouro_fim]")

        return (df_ouro_final, mae, r2, mae_ant, melhoria, status_bq,
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
