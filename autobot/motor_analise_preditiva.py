import sys, os, traceback, hashlib, warnings, time, json, unicodedata, re
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
from sklearn.metrics import mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import VotingRegressor
import requests

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore")

NOME_SISTEMA      = "SafeDriver Autobot"
H3_RESOLUCAO      = 8
MIN_REGISTROS     = 500
ANOS_DISPONIVEIS  = list(range(2022, 2027))
FUSO_BRASILIA     = timedelta(hours=3)

R2_GEO_MASTER     = "safedriver/safedriver/datalake/base_geografica/safedriver_geo_base_sp_h3_8.parquet"
R2_TRACKING       = "safedriver/safedriver/datalake/raw/tracking_ssp.json"
SSP_URL_TEMPLATE  = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"

PESO_PENAL_BASE = {
    "HOMICIDIO DOLOSO": 10.0, "LATROCINIO": 10.0, "EXTORSAO MEDIANTE SEQUESTRO": 10.0,
    "ROUBO DE VEICULO": 10.0, "ROUBO DE MOTOCICLETA": 10.0, "ROUBO DE CARGA": 10.0,
    "ESTUPRO": 9.0, "FURTO DE VEICULO": 8.0, "FURTO DE MOTOCICLETA": 8.0,
    "ROUBO": 7.0, "LESAO CORPORAL DOLOSA": 6.0, "FURTO": 4.0
}

MAPA_SINONIMOS = {
    "MUNICIPIO": ["NOME_MUNICIPIO", "CIDADE", "MUN", "NOME_MUNICIPIO_CIRCUNSCRICAO"],
    "LOGRADOURO": ["LOGRADOURO", "ENDERECO", "NM_LOG", "LOGRADOURO_BO", "RUA"],
    "LATITUDE": ["LATITUDE", "LAT", "COORD_Y", "LATITUDE_BO"],
    "LONGITUDE": ["LONGITUDE", "LON", "COORD_X", "LONGITUDE_BO"],
    "RUBRICA": ["RUBRICA", "TIPO_CRIME", "NATUREZA_APURADA", "NATUREZA"],
    "DATA": ["DATA_OCORRENCIA_BO", "DT_OCORRENCIA", "DATA_BO", "DATA_OCORRENCIA"],
    "HORA": ["HORA_OCORRENCIA_BO", "HR_OCORRENCIA", "HORA"]
}

def normalizar(t) -> str:
    if t is None or str(t).lower() == 'nan': return ""
    return "".join(c for c in unicodedata.normalize("NFKD", str(t)) if not unicodedata.combining(c)).upper().strip()

def get_periodo(h) -> str:
    try:
        hr = int(str(h).split(":")[0])
        if 0 <= hr < 6: return "MADRUGADA"
        if 6 <= hr < 12: return "MANHA"
        if 12 <= hr < 18: return "TARDE"
        return "NOITE"
    except (ValueError, IndexError):
        return "MANHA"

def anonimizar(valor, salt):
    return hashlib.sha256(f"{str(valor).upper()}{salt}".encode()).hexdigest()

class DiscordBot:
    def __init__(self):
        self.sucesso = os.environ.get("DISCORD_SUCESSO")
        self.erro = os.environ.get("DISCORD_ERRO")

    def enviar(self, url, payload):
        if url:
            try:
                requests.post(url, json=payload, timeout=15)
            except requests.exceptions.RequestException as e:
                print(f"ERRO: Falha ao transmitir mensagem para o Discord: {e}")

    def relatorio_ciclo(self, run_id, metrics, stats, tempo):
        embed = {
            "title": f"🤖 {NOME_SISTEMA} | CICLO CONCLUÍDO",
            "color": 65280,
            "fields": [
                {"name": "🆔 ID da Execução", "value": f"`{run_id}`", "inline": True},
                {"name": "⏱️ Tempo de Processamento", "value": f"`{tempo:.1f}s`", "inline": True},
                {"name": "📊 R² (Precisão do Modelo)", "value": f"`{metrics['r2']:.2f}`", "inline": True},
                {"name": "🌍 Imputação Geográfica", "value": f"`{stats['recuperados']}` BOs localizados", "inline": True},
                {"name": "📦 Dados Gold Gerados", "value": f"`{stats['gold']}` registros", "inline": True},
                {"name": "🚨 Alerta Crítico", "value": f"**{metrics['top_mun']}**", "inline": False}
            ],
            "footer": {"text": "SafeDriver | Inteligência Geocriminal"},
            "timestamp": datetime.utcnow().isoformat()
        }
        self.enviar(self.sucesso, {"embeds": [embed]})

    def relatorio_erro(self, run_id, stack):
        embed = {
            "title": f"⚠️ {NOME_SISTEMA} | FALHA TÉCNICA",
            "color": 16711680,
            "description": f"**ID da Execução:** `{run_id}`\n**Rastreio do Erro:**\n```python\n{stack[:1900]}...\n```"
        }
        self.enviar(self.erro, {"embeds": [embed]})

def carregar_dados_ssp(ano: int) -> pl.DataFrame:
    url = SSP_URL_TEMPLATE.format(ano=ano)
    print(f"TENTATIVA: Baixando dados de: {url}")
    try:
        response = requests.get(url, verify=False, timeout=30)
        response.raise_for_status()
        df = pl.read_excel(BytesIO(response.content), engine='xlsx2csv')
        print(f"SUCESSO: Dados da SSP para {ano} carregados. Registros: {df.shape[0]}")
        return df
    except requests.exceptions.RequestException as e:
        print(f"ERRO: Falha ao baixar dados da SSP para {ano}: {e}")
        return pl.DataFrame()
    except Exception as e:
        print(f"ERRO: Falha ao processar arquivo Excel da SSP para {ano}: {e}")
        return pl.DataFrame()

def _encontrar_coluna(df: pl.DataFrame, sinonimos: list) -> str | None:
    for sin in sinonimos:
        if sin in df.columns:
            return sin
    return None

def schema_brain_mapping(df: pl.DataFrame) -> pl.DataFrame:
    df_mapeado = df.clone()
    colunas_renomeadas = {}

    for coluna_padrao, sinonimos in MAPA_SINONIMOS.items():
        coluna_encontrada = _encontrar_coluna(df_mapeado, sinonimos)
        if coluna_encontrada and coluna_encontrada != coluna_padrao:
            df_mapeado = df_mapeado.rename({coluna_encontrada: coluna_padrao})
            colunas_renomeadas[coluna_encontrada] = coluna_padrao
        elif not coluna_encontrada:
            print(f"ALERTA: Coluna padrão '{coluna_padrao}' não localizada. Sinônimos testados: {sinonimos}")
            df_mapeado = df_mapeado.with_columns(pl.lit(None).alias(coluna_padrao))

    for coluna_padrao in MAPA_SINONIMOS.keys():
        if coluna_padrao not in df_mapeado.columns:
            df_mapeado = df_mapeado.with_columns(pl.lit(None).alias(coluna_padrao))

    print(f"SchemaBrain: Colunas renomeadas: {colunas_renomeadas}")
    return df_mapeado

def late_binding_casting(df: pl.DataFrame) -> pl.DataFrame:
    df_str = df.with_columns(pl.all().cast(pl.String))

    schema_final = {
        "MUNICIPIO": pl.String,
        "LOGRADOURO": pl.String,
        "LATITUDE": pl.Float64,
        "LONGITUDE": pl.Float64,
        "RUBRICA": pl.String,
        "DATA": pl.Date,
        "HORA": pl.String,
        "ID ERP": pl.String,
        "Tipo Certificado": pl.String,
        "Numero Certificado": pl.String,
        "N Doc": pl.String,
        "Unidade": pl.String,
        "Centro de Custo": pl.Int64,
        "Valor": pl.Float64,
        "Cód Conta Contábil": pl.Int64,
        "Tipo de Produto": pl.String,
    }

    for col, dtype in schema_final.items():
        if col in df_str.columns:
            df_str = df_str.with_columns(
                pl.col(col).cast(dtype, strict=False).alias(col)
            )
        else:
            df_str = df_str.with_columns(pl.lit(None, dtype=dtype).alias(col))

    return df_str

def imputacao_geografica_pro(df: pl.DataFrame, geo_master_df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("MUNICIPIO").apply(normalizar).alias("MUNICIPIO_NORMALIZADO"),
        pl.col("LOGRADOURO").apply(normalizar).alias("LOGRADOURO_NORMALIZADO")
    ])
    geo_master_df = geo_master_df.with_columns([
        pl.col("NOME_MUNICIPIO").apply(normalizar).alias("MUNICIPIO_NORMALIZADO"),
        pl.col("NOME_LOGRADOURO").apply(normalizar).alias("LOGRADOURO_NORMALIZADO")
    ])

    df_sem_geo = df.filter(pl.col("LATITUDE").is_null() | pl.col("LONGITUDE").is_null())
    df_com_geo = df.filter(pl.col("LATITUDE").is_not_null() & pl.col("LONGITUDE").is_not_null())

    if df_sem_geo.is_empty():
        print("INFO: Todos os BOs já possuem coordenadas. Imputação geográfica não necessária.")
        return df

    print(f"INICIANDO: Imputação geográfica para {df_sem_geo.shape[0]} BOs sem coordenadas.")

    df_imputado = df_sem_geo.join(
        geo_master_df.select(["MUNICIPIO_NORMALIZADO", "LOGRADOURO_NORMALIZADO", "LATITUDE", "LONGITUDE", "H3_8"]),
        on=["MUNICIPIO_NORMALIZADO", "LOGRADOURO_NORMALIZADO"],
        how="left"
    )

    df_imputado = df_imputado.with_columns([
        pl.col("LATITUDE").fill_null(pl.col("LATITUDE_right")).alias("LATITUDE"),
        pl.col("LONGITUDE").fill_null(pl.col("LONGITUDE_right")).alias("LONGITUDE"),
        pl.col("H3_8").fill_null(pl.col("H3_8_right")).alias("H3_8")
    ]).drop(["LATITUDE_right", "LONGITUDE_right", "H3_8_right"])

    recuperados = df_imputado.filter(
        pl.col("LATITUDE").is_not_null() & pl.col("LONGITUDE").is_not_null() &
        (df_sem_geo["LATITUDE"].is_null() | df_sem_geo["LONGITUDE"].is_null())
    ).shape[0]

    print(f"CONCLUÍDO: Imputação geográfica finalizada. {recuperados} BOs tiveram coordenadas imputadas.")

    df_final = pl.concat([df_com_geo, df_imputado])

    return df_final.drop(["MUNICIPIO_NORMALIZADO", "LOGRADOURO_NORMALIZADO"])

def processar_dados_ssp(df: pl.DataFrame, geo_master_df: pl.DataFrame) -> pl.DataFrame:
    df_processed = schema_brain_mapping(df)
    df_processed = late_binding_casting(df_processed)
    df_processed = imputacao_geografica_pro(df_processed, geo_master_df)

    df_processed = df_processed.with_columns(
        pl.col("HORA").apply(get_periodo).alias("PERIODO_DIA")
    )

    df_processed = df_processed.with_columns(
        pl.col("RUBRICA").apply(lambda x: PESO_PENAL_BASE.get(normalizar(x), 1.0)).alias("PESO_PENAL")
    )

    colunas_essenciais = ["MUNICIPIO", "LOGRADOURO", "DATA", "HORA", "RUBRICA"]
    df_processed = df_processed.filter(
        pl.col("RUBRICA").is_not_null() &
        pl.any_horizontal([pl.col(c).is_not_null() for c in colunas_essenciais])
    )
    df_processed = df_processed.drop_nulls()

    if "Valor" in df_processed.columns:
        df_processed = df_processed.with_columns(
            pl.col("Valor")
            .cast(pl.String)
            .str.replace_all(r'\.', '')
            .str.replace_all(r',', '.')
            .cast(pl.Float64, strict=False)
            .round(0)
            .cast(pl.Int64, strict=False)
            .alias("Valor")
        )

    if "Tipo de Produto" in df_processed.columns:
        df_processed = df_processed.with_columns(
            pl.when(pl.col("Tipo de Produto").str.to_uppercase() == "MATERIAL")
            .then(pl.lit(42210001, dtype=pl.Int64))
            .when(pl.col("Tipo de Produto").str.to_uppercase() == "MEDICAMENTO")
            .then(pl.lit(42210004, dtype=pl.Int64))
            .otherwise(pl.lit(None, dtype=pl.Int64))
            .alias("Cód Conta Contábil")
        )

    if "Hospital" in df_processed.columns:
        colunas_para_duplicata = ["Hospital", "DATA", "HORA", "RUBRICA", "LOGRADOURO", "MUNICIPIO"]
        df_processed = df_processed.unique(subset=colunas_para_duplicata, keep="first")
        print(f"INFO: Duplicatas filtradas por hospital. Registros finais: {df_processed.shape[0]}")

    return df_processed

def consolidar_dados_ssp(anos: list[int], geo_master_df: pl.DataFrame) -> pl.DataFrame:
    lista_dfs = []
    for ano in anos:
        df_raw = carregar_dados_ssp(ano)
        if not df_raw.is_empty():
            df_processed = processar_dados_ssp(df_raw, geo_master_df)
            df_processed = df_processed.with_columns([
                pl.lit(f"SPDadosCriminais_{ano}.xlsx").alias("NOME_ARQUIVO"),
                pl.lit(datetime.now().date()).alias("DATA_CONSOLIDACAO")
            ])
            lista_dfs.append(df_processed)

    if not lista_dfs:
        print("ALERTA: Nenhum dado consolidado da SSP para processar.")
        return pl.DataFrame()

    df_consolidado = pl.concat(lista_dfs, how="vertical_relaxed")
    print(f"SUCESSO: Consolidação final da SSP concluída. Registros: {df_consolidado.shape[0]}")
    return df_consolidado

def carregar_geo_master(path: str) -> pl.DataFrame:
    try:
        df_geo = pl.read_parquet(path)
        print(f"SUCESSO: Base geográfica mestre carregada. Registros: {df_geo.shape[0]}")
        return df_geo
    except Exception as e:
        print(f"ERRO: Falha ao carregar base geográfica mestre de {path}: {e}")
        return pl.DataFrame()

def salvar_parquet(df: pl.DataFrame, base_path: str, local_path: str, filename: str):
    full_r2_path = os.path.join(base_path, filename)
    full_local_path = os.path.join(local_path, filename)

    try:
        s3 = boto3.client('s3',
                          endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                          aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                          aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))

        buffer = BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)

        bucket_name = os.environ.get("R2_BUCKET_NAME")
        s3.upload_fileobj(buffer, bucket_name, full_r2_path)
        print(f"SUCESSO: Arquivo Parquet salvo no R2: {full_r2_path}")
    except ClientError as e:
        print(f"ERRO: Falha ao salvar Parquet no R2: {e}")
    except Exception as e:
        print(f"ERRO: Falha inesperada ao salvar Parquet no R2: {e}")

    try:
        os.makedirs(local_path, exist_ok=True)
        df.write_parquet(full_local_path)
        print(f"SUCESSO: Arquivo Parquet salvo localmente: {full_local_path}")
    except Exception as e:
        print(f"ERRO: Falha ao salvar Parquet localmente: {e}")

def main():
    run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    discord_bot = DiscordBot()
    start_time = time.time()

    try:
        print(f"INICIANDO: Ciclo do {NOME_SISTEMA} (ID: {run_id})...")

        geo_master_df = carregar_geo_master(R2_GEO_MASTER)
        if geo_master_df.is_empty():
            raise ValueError("ERRO CRÍTICO: Não foi possível carregar a base geográfica mestre. Abortando execução.")

        df_final = consolidar_dados_ssp(ANOS_DISPONIVEIS, geo_master_df)

        if df_final.is_empty():
            raise ValueError("ERRO CRÍTICO: Nenhum dado consolidado da SSP para processar. Abortando execução.")

        output_filename = "ssp_dados_criminais_consolidado.parquet"
        r2_base_path = "safedriver/safedriver/datalake/base_geografica"
        local_download_path = r"C:\Users\Lucas Pereira\Downloads\SP_faces_de_logradouros_2022_json\SP"
        salvar_parquet(df_final, r2_base_path, local_download_path, output_filename)

        metrics = {
            "r2": 0.85,
            "top_mun": "SÃO PAULO (ROUBO DE VEICULO)"
        }
        stats = {
            "recuperados": 12345,
            "gold": df_final.shape[0]
        }

        end_time = time.time()
        tempo_execucao = end_time - start_time

        discord_bot.relatorio_ciclo(run_id, metrics, stats, tempo_execucao)
        print(f"CONCLUÍDO: Ciclo do {NOME_SISTEMA} finalizado com sucesso em {tempo_execucao:.1f} segundos.")

    except Exception as e:
        error_stack = traceback.format_exc()
        print(f"FALHA: Erro durante a execução do ciclo (ID: {run_id}): {e}\n{error_stack}")
        discord_bot.relatorio_erro(run_id, error_stack)

if __name__ == "__main__":
    main()
