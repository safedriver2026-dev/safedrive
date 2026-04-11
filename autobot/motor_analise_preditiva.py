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
                if "Campos da Tabela_SPDADOS" in sheet_name.upper(): # Ignorar case-insensitive
                    continue

                try:
                    # Ler tudo como string para evitar problemas de tipo na RAW
                    df_sheet = excel_file.parse(sheet_name, dtype=str)

                    # Normalizar nomes de colunas para a detecção
                    df_sheet.columns = [normalizar_texto(col) for col in df_sheet.columns]

                    colunas_presentes = [col for col in COLUNAS_CRITICAS_SSP if col in df_sheet.columns]
                    if len(colunas_presentes) >= 5:
                        print(f"[SSP] Aba '{sheet_name}' identificada como dados para o ano {ano}.")
                        df_consolidado = pd.concat([df_consolidado, df_sheet], ignore_index=True)
                    else:
                        print(f"[SSP] Aba '{sheet_name}' ignorada (poucas colunas críticas ou não é aba de dados).")
                except Exception as e:
                    print(f"[SSP] Erro ao processar aba '{sheet_name}' para {ano}: {e}")

            if not df_consolidado.empty:
                return df_consolidado
            else:
                print(f"[SSP] Nenhuma aba de dados encontrada para o ano {ano}.")
                return None

        except requests.exceptions.RequestException as e:
            print(f"[SSP] Erro ao baixar {url} (tentativa {tentativa}/{SSP_MAX_TENTATIVAS}): {e}")
            time.sleep(5)
    print(f"[SSP] Falha ao baixar {url} após {SSP_MAX_TENTATIVAS} tentativas.")
    return None

class TrackingSSP:
    def __init__(self, s3_client, bucket_name, key):
        self.s3 = s3_client
        self.bucket = bucket_name
        self.key = key
        self.dados = self._carregar_tracking()

    def _carregar_tracking(self):
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=self.key)
            content = response["Body"].read().decode("utf-8")
            tracking_data = json.loads(content)

            # Migração defensiva: se encontrar int onde deveria ser dict, corrige
            for ano, info in tracking_data.items():
                if isinstance(info, int):
                    tracking_data[ano] = {"tamanho_bytes": info, "hash_sha256": ""}
            return tracking_data
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
            Key=self.key,
            Body=json.dumps(self.dados, indent=2).encode("utf-8"),
            ContentType="application/json"
        )

    def precisa_processar(self, ano: int, tamanho_bytes: int, hash_sha256: str) -> bool:
        ano_str = str(ano)
        if ano_str not in self.dados:
            return True

        info_existente = self.dados[ano_str]
        return (info_existente.get("tamanho_bytes") != tamanho_bytes or
                info_existente.get("hash_sha256") != hash_sha256)

    def atualizar_tracking(self, ano: int, tamanho_bytes: int, hash_sha256: str):
        self.dados[str(ano)] = {
            "tamanho_bytes": tamanho_bytes,
            "hash_sha256": hash_sha256,
            "ultima_atualizacao": hora_brasilia().isoformat()
        }

class DiscordNotifier:
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url

    def _enviar_mensagem(self, titulo: str, descricao: str, cor: int):
        if not self.webhook_url:
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
            requests.post(self.webhook_url, json=payload, timeout=10)
        except requests.exceptions.RequestException as e:
            print(f"[Discord] Erro ao enviar mensagem: {e}")

    def alerta_erro(self, run_id: str, titulo: str, detalhes: str):
        desc = f"**Run ID:** `{run_id}`\n\n**Detalhes:**\n```\n{detalhes}\n```"
        self._enviar_mensagem(f"🚨 {titulo}", desc, 15158332)

    def relatorio_executivo(self, run_id: str, tempo: float, n_ouro: int, mae: float, r2: float,
                            melhoria: float, top_municipio: str, top_shap_feature: str):
        desc = (
            f"**Run ID:** `{run_id}`\n"
            f"**Tempo de Execução:** `{tempo:.1f}s`\n"
            f"**Registros Ouro:** `{n_ouro}`\n"
            f"**MAE:** `{mae:.4f}`\n"
            f"**R²:** `{r2:.4f}`\n"
            f"**Melhoria em relação ao ano anterior:** `{melhoria:.2%}`\n"
            f"**Município de Maior Risco:** `{top_municipio}`\n"
            f"**Feature Mais Impactante:** `{top_shap_feature}`"
        )
        self._enviar_mensagem(f"✅ Execução Concluída - Relatório Executivo", desc, 3066993)

    def relatorio_operacional(self, run_id: str, n_raw: int, n_prata: int, n_ouro: int,
                               mae: float, r2: float, mae_ant: float, anos_processados: list,
                               status_bq: str, shap_top3: list, r2_prefixo_raw: str):
        shap_str = "\n".join([f"- {f}: {v:.4f}" for f, v in shap_top3])
        desc = (
            f"**Run ID:** `{run_id}`\n"
            f"**Dados RAW processados:** `{n_raw}`\n"
            f"**Dados PRATA gerados:** `{n_prata}`\n"
            f"**Dados OURO gerados:** `{n_ouro}`\n"
            f"**Anos Processados:** `{', '.join(map(str, anos_processados))}`\n"
            f"**MAE Atual:** `{mae:.4f}` (Anterior: `{mae_ant:.4f}`)\n"
            f"**R² Atual:** `{r2:.4f}`\n"
            f"**Status BigQuery:** `{status_bq}`\n"
            f"**Top 3 Features SHAP:**\n```\n{shap_str}\n```\n"
            f"**Prefixo R2 RAW:** `{r2_prefixo_raw}`"
        )
        self._enviar_mensagem(f"📊 Execução Concluída - Relatório Operacional", desc, 10181046)

    def sem_novidades(self, run_id: str, tempo: float):
        desc = (
            f"**Run ID:** `{run_id}`\n"
            f"**Tempo de Execução:** `{tempo:.1f}s`\n"
            f"Nenhum dado RAW novo ou alterado foi encontrado para processamento."
        )
        self._enviar_mensagem(f"💤 Execução Concluída - Sem Novidades", desc, 16776960)

class SafeDriver:
    def __init__(self):
        self.run_id = run_id_curto()
        self.t_inicio = time.time()
        self.anos_processados = []

        r2_access_key_id = sanitizar_secret(os.environ.get("R2_ACCESS_KEY_ID", ""))
        r2_secret_access_key = sanitizar_secret(os.environ.get("R2_SECRET_ACCESS_KEY", ""))
        r2_endpoint_url = sanitizar_secret(os.environ.get("R2_ENDPOINT_URL", ""))
        self.r2_bucket = sanitizar_secret(os.environ.get("R2_BUCKET_NAME", ""))

        self.s3 = boto3.client(
            "s3",
            endpoint_url=r2_endpoint_url,
            aws_access_key_id=r2_access_key_id,
            aws_secret_access_key=r2_secret_access_key,
            config=boto3.session.Config(signature_version="s3v4")
        )
        self.tracking = TrackingSSP(self.s3, self.r2_bucket, R2_TRACKING)
        self.discord = DiscordNotifier(os.environ.get("DISCORD_WEBHOOK_URL"))
        self.lgpd_salt = sanitizar_secret(os.environ.get("LGPD_SALT", ""))

        self.df_raw = pl.DataFrame()

    def sincronizar_raw(self):
        print("[sincronizar_raw_inicio]")
        df_raw_acumulado = pl.DataFrame()

        for ano in ANOS_DISPONIVEIS:
            r2_key = f"{R2_PREFIXO_RAW}ssp_{ano}.parquet"
            df_ano = pl.DataFrame()

            try:
                response = self.s3.get_object(Bucket=self.r2_bucket, Key=r2_key)
                parquet_data = response["Body"].read()
                tamanho_bytes = len(parquet_data)
                hash_sha256 = hashlib.sha256(parquet_data).hexdigest()

                if self.tracking.precisa_processar(ano, tamanho_bytes, hash_sha256):
                    print(f"[R2] {r2_key} encontrado, mas alterado ou não rastreado — processando.")
                    df_ano = pl.read_parquet(BytesIO(parquet_data))
                    self.tracking.atualizar_tracking(ano, tamanho_bytes, hash_sha256)
                else:
                    print(f"[R2] {r2_key} encontrado e inalterado — ignorando.")
                    continue

            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    print(f"[R2] {r2_key} não existe — tentando baixar da SSP-SP.")
                    df_ssp_pandas = baixar_ssp(ano)

                    if df_ssp_pandas is not None and not df_ssp_pandas.empty:
                        # Forçar todas as colunas para string antes de converter para Polars
                        # Isso garante que o Polars não infira tipos numéricos incorretamente
                        # e evita o ShapeError na concatenação devido a tipos diferentes
                        for col in df_ssp_pandas.columns:
                            df_ssp_pandas[col] = df_ssp_pandas[col].astype(str)

                        df_ano = pl.from_pandas(df_ssp_pandas)
                        df_ano = renomear_sinonimos(df_ano) # Renomear colunas aqui

                        # Salvar no R2 após download e renomeação
                        buffer = BytesIO()
                        df_ano.write_parquet(buffer)
                        buffer.seek(0)
                        self.s3.put_object(Bucket=self.r2_bucket, Key=r2_key, Body=buffer.getvalue())

                        tamanho_bytes = buffer.tell()
                        hash_sha256 = hashlib.sha256(buffer.getvalue()).hexdigest()
                        self.tracking.atualizar_tracking(ano, tamanho_bytes, hash_sha256)
                        print(f"[R2] {r2_key} salvo no R2 após download da SSP-SP.")
                    else:
                        print(f"[SSP] Não foi possível obter dados para o ano {ano}.")
                        continue
                else:
                    raise

            if not df_ano.is_empty():
                df_raw_acumulado = pl.concat([df_raw_acumulado, df_ano])
                self.anos_processados.append(ano)

        self.tracking.salvar_tracking()
        self.df_raw = df_raw_acumulado
        print(f"[sincronizar_raw_fim] {len(self.df_raw)} registros RAW acumulados.")

    def construir_prata(self):
        print("[construir_prata_inicio]")
        if self.df_raw.is_empty():
            print("[prata] df_raw está vazio, pulando construção da camada prata.")
            return pl.DataFrame()

        df_prata = self.df_raw.clone()

        # Renomear colunas (já feito em sincronizar_raw, mas reforçar aqui para segurança)
        df_prata = renomear_sinonimos(df_prata)

        # Filtrar apenas colunas relevantes para a camada prata
        colunas_prata = [
            "DATA_OCORRENCIA_BO", "HORA_OCORRENCIA_BO", "LATITUDE", "LONGITUDE",
            "RUBRICA", "NOME_MUNICIPIO", "LOGRADOURO", "BAIRRO", "NUMERO_LOGRADOURO"
        ]
        df_prata = df_prata.select([col for col in colunas_prata if col in df_prata.columns])

        # Tipagem e tratamento de nulos/inválidos
        df_prata = df_prata.with_columns([
            pl.col("DATA_OCORRENCIA_BO").str.to_date("%d/%m/%Y", strict=False).alias("DATA_OCORRENCIA_BO"),
            pl.col("HORA_OCORRENCIA_BO").str.slice(0, 5).alias("HORA_OCORRENCIA_BO"), # Garantir HH:MM
            pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).alias("LATITUDE"),
            pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).alias("LONGITUDE"),
            pl.col("RUBRICA").apply(normalizar_texto).alias("RUBRICA_NORMALIZADA"),
            pl.col("NOME_MUNICIPIO").apply(normalizar_texto).alias("NOME_MUNICIPIO_NORMALIZADO"),
            pl.col("LOGRADOURO").apply(normalizar_texto).alias("LOGRADOURO_NORMALIZADO"),
            pl.col("BAIRRO").apply(normalizar_texto).alias("BAIRRO_NORMALIZADO"),
            pl.col("NUMERO_LOGRADOURO").str.replace(",", ".").cast(pl.Float64, strict=False).alias("NUMERO_LOGRADOURO"),
        ])

        # Remover linhas com datas nulas ou rubricas vazias
        df_prata = df_prata.filter(
            pl.col("DATA_OCORRENCIA_BO").is_not_null() &
            pl.col("RUBRICA_NORMALIZADA").is_not_null() &
            pl.col("RUBRICA_NORMALIZADA").str.len_bytes() > 0
        )

        # Filtrar rubricas que não estão em PESO_PENAL_BASE (exclui acidentes, tráfico, etc.)
        rubricas_validas = set(PESO_PENAL_BASE.keys())
        df_prata = df_prata.filter(pl.col("RUBRICA_NORMALIZADA").is_in(list(rubricas_validas)))

        # Tratamento de coordenadas inválidas/ausentes (conforme PDF)
        # Marcar como nulo se for 0.0 ou fora dos limites de SP
        df_prata = df_prata.with_columns([
            pl.when(
                (pl.col("LATITUDE").is_null()) | (pl.col("LATITUDE") == 0.0) |
                (pl.col("LATITUDE") < SP_LAT_MIN) | (pl.col("LATITUDE") > SP_LAT_MAX)
            ).then(pl.lit(None, pl.Float64))
            .otherwise(pl.col("LATITUDE")).alias("LATITUDE_F"),
            pl.when(
                (pl.col("LONGITUDE").is_null()) | (pl.col("LONGITUDE") == 0.0) |
                (pl.col("LONGITUDE") < SP_LON_MIN) | (pl.col("LONGITUDE") > SP_LON_MAX)
            ).then(pl.lit(None, pl.Float64))
            .otherwise(pl.col("LONGITUDE")).alias("LONGITUDE_F"),
        ])

        # Preencher LATITUDE_F e LONGITUDE_F nulas com a média do município para H3
        # (temporário, idealmente seria geocodificação)
        df_prata = df_prata.with_columns([
            pl.col("LATITUDE_F").fill_null(strategy="mean").over("NOME_MUNICIPIO_NORMALIZADO"),
            pl.col("LONGITUDE_F").fill_null(strategy="mean").over("NOME_MUNICIPIO_NORMALIZADO"),
        ])
        # Se ainda houver nulos (município com todas as coords nulas), preencher com a média geral
        df_prata = df_prata.with_columns([
            pl.col("LATITUDE_F").fill_null(df_prata["LATITUDE_F"].mean()),
            pl.col("LONGITUDE_F").fill_null(df_prata["LONGITUDE_F"].mean()),
        ])

        # Filtrar linhas onde LATITUDE_F ou LONGITUDE_F ainda são nulas após preenchimento
        df_prata = df_prata.filter(
            pl.col("LATITUDE_F").is_not_null() & pl.col("LONGITUDE_F").is_not_null()
        )

        # Geração de H3 Index
        df_prata = df_prata.with_columns(
            pl.struct(["LATITUDE_F", "LONGITUDE_F"])
            .apply(lambda x: h3.geo_to_h3(x["LATITUDE_F"], x["LONGITUDE_F"], H3_RESOLUCAO))
            .alias("H3_INDEX")
        )

        # Anonimização de campos sensíveis
        df_prata = df_prata.with_columns([
            pl.col("LOGRADOURO_NORMALIZADO").apply(lambda x: anonimizar_campo(x, self.lgpd_salt)).alias("LOGRADOURO_ANONIMIZADO"),
            pl.col("BAIRRO_NORMALIZADO").apply(lambda x: anonimizar_campo(x, self.lgpd_salt)).alias("BAIRRO_ANONIMIZADO"),
            pl.col("NOME_MUNICIPIO_NORMALIZADO").apply(lambda x: anonimizar_campo(x, self.lgpd_salt)).alias("NOME_MUNICIPIO_ANONIMIZADO"),
        ])

        # Features de tempo
        df_prata = df_prata.with_columns([
            pl.col("DATA_OCORRENCIA_BO").dt.year().alias("ANO"),
            pl.col("DATA_OCORRENCIA_BO").dt.month().alias("MES"),
            pl.col("DATA_OCORRENCIA_BO").dt.weekday().alias("DIA_SEMANA"),
            pl.col("HORA_OCORRENCIA_BO").apply(classificar_periodo).alias("PERIODO_DIA"),
            pl.col("HORA_OCORRENCIA_BO").apply(lambda h: 1 if classificar_periodo(h) in ["NOITE", "MADRUGADA"] else 0).alias("IS_NOITE_MADRUGADA"),
        ])

        # Feriados
        br_holidays = holidays.Brazil(state='SP', years=self.anos_processados)
        df_prata = df_prata.with_columns(
            pl.col("DATA_OCORRENCIA_BO").apply(lambda d: 1 if d in br_holidays else 0).alias("IS_FERIADO")
        )

        # Classificação de tipo de crime
        df_prata = df_prata.with_columns([
            pl.col("RUBRICA_NORMALIZADA").apply(lambda r: 1 if r in ["ROUBO", "FURTO", "ROUBO DE VEICULO", "FURTO DE VEICULO", "ROUBO DE MOTOCICLETA", "FURTO DE MOTOCICLETA", "ROUBO DE CARGA"] else 0).alias("IS_PATRIMONIO"),
            pl.col("RUBRICA_NORMALIZADA").apply(lambda r: 1 if r in ["HOMICIDIO DOLOSO", "LATROCINIO", "EXTORSAO MEDIANTE SEQUESTRO", "ESTUPRO", "LESAO CORPORAL DOLOSA"] else 0).alias("IS_VIOLENCIA_PESSOA"),
        ])

        # Cálculo de escores por perfil
        df_prata = df_prata.with_columns([
            pl.struct(["RUBRICA_NORMALIZADA", "HORA_OCORRENCIA_BO"])
            .apply(lambda x: calcular_escores_todos_perfis(x["RUBRICA_NORMALIZADA"], x["HORA_OCORRENCIA_BO"]))
            .alias("ESCORES_PERFIL")
        ])
        df_prata = df_prata.unnest("ESCORES_PERFIL")

        # Escore total da ocorrência
        df_prata = df_prata.with_columns(
            (pl.col("ESCORE_MOTORISTA") + pl.col("ESCORE_MOTOCICLISTA") +
             pl.col("ESCORE_PEDESTRE") + pl.col("ESCORE_CICLISTA")).alias("ESCORE_TOTAL_OCORRENCIA")
        )

        print(f"[construir_prata_fim] {len(df_prata)} registros PRATA gerados.")
        return df_prata

    def construir_ouro(self, df_prata: pl.DataFrame, bq_project: str, bq_dataset: str, bq_cred: str):
        print("[construir_ouro_inicio]")
        if df_prata.is_empty():
            print("[ouro] df_prata está vazio, pulando construção da camada ouro.")
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

        mae_ant = 0.0
        r2_ant = 0.0

        for fold, (train_index, test_index) in enumerate(tscv.split(X, y, anos)):
            print(f"[modelo] Treinando Fold {fold+1}...")
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            model.fit(X_train, y_train.values.ravel())

            preds = model.predict(X_test)
            oof_preds[test_index] = preds
            oof_targets[test_index] = y_test.values.ravel()

            if fold == tscv.n_splits - 1: # Último fold para métricas e SHAP
                mae = mean_absolute_error(y_test, preds)
                r2 = r2_score(y_test, preds)
                print(f"[modelo] MAE no último fold: {mae:.4f}")
                print(f"[modelo] R² no último fold: {r2:.4f}")

                # Calcular SHAP para o último fold
                try:
                    explainer = shap.TreeExplainer(model.estimators_[0]) # Usar LGBM para SHAP
                    shap_values = explainer.shap_values(X_test)

                    if isinstance(shap_values, list): # Para modelos multi-output
                        shap_values = shap_values[0]

                    shap_sum = np.abs(shap_values).mean(axis=0)
                    shap_features = pd.DataFrame(list(zip(X_test.columns, shap_sum)), columns=['feature','shap_value'])
                    shap_features = shap_features.sort_values(by='shap_value', ascending=False)

                    top_shap_feature = shap_features.iloc[0]['feature']
                    shap_top3 = shap_features.head(3).to_dict('records')
                    print(f"[modelo] Top feature SHAP: {top_shap_feature}")
                except Exception as e:
                    print(f"[modelo] Erro ao calcular SHAP: {e}")
                    top_shap_feature = "N/A"
                    shap_top3 = []

        # Comparar com o ano anterior (se houver)
        if len(self.anos_processados) > 1:
            ano_anterior = max(self.anos_processados) - 1
            preds_ano_anterior = oof_preds[anos == ano_anterior]
            targets_ano_anterior = oof_targets[anos == ano_anterior]
            if len(preds_ano_anterior) > 0:
                mae_ant = mean_absolute_error(targets_ano_anterior, preds_ano_anterior)
                print(f"[modelo] MAE do ano anterior: {mae_ant:.4f}")

        melhoria = 0.0
        if mae_ant > 0:
            melhoria = (mae_ant - mae) / mae_ant

        # Adicionar previsões ao DataFrame agregado
        df_agregado = df_agregado.with_columns(pl.Series(name="ESCORE_PREDITO", values=oof_preds))

        # Salvar modelo
        modelo_key = f"{R2_PREFIXO_MODELO}modelo_{hora_brasilia().strftime('%Y%m%d_%H%M%S')}.joblib"
        buffer_modelo = BytesIO()
        joblib.dump(model, buffer_modelo)
        buffer_modelo.seek(0)
        self.s3.put_object(Bucket=self.r2_bucket, Key=modelo_key, Body=buffer_modelo.getvalue())
        print(f"[modelo] Modelo salvo em {modelo_key}")

        # Identificar município dominante para o relatório executivo
        top_municipio = df_agregado.group_by("MUNICIPIO_DOMINANTE").agg(pl.col("QTD_CRIMES").sum().alias("TOTAL_CRIMES")).sort("TOTAL_CRIMES", descending=True).head(1).select("MUNICIPIO_DOMINANTE").item()

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
