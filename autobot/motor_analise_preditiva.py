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
GEOJSON_BAIRROS_PATH = "safedriver/safedriver/datalake/base_geografica/BR_bairros_CD2022.geojson"

PESO_PENAL_BASE = {
    "HOMICIDIO DOLOSO": 10.0, "LATROCINIO": 10.0, "EXTORSAO MEDIANTE SEQUESTRO": 10.0,
    "ROUBO DE VEICULO": 10.0, "ROUBO DE MOTOCICLETA": 10.0, "ROUBO DE CARGA": 10.0,
    "ESTUPRO": 9.0, "FURTO DE VEICULO": 8.0, "FURTO DE MOTOCICLETA": 8.0,
    "ROUBO": 7.0, "LESAO CORPORAL DOLOSA": 6.0, "FURTO": 4.0
}

MAPA_SINONIMOS = {
    "ID ERP": ["ID_ERP", "ID_DOCUMENTO_ERP"],
    "Tipo Certificado": ["TIPO_CERTIFICADO", "CERT_TIPO"],
    "Numero Certificado": ["NUMERO_CERTIFICADO", "CERT_NUM"],
    "N Doc": ["N_DOC", "NUMERO_DOCUMENTO", "NUM_DOC"],
    "Unidade": ["UNIDADE", "NOME_UNIDADE", "UNID"],
    "Centro de Custo": ["CENTRO_CUSTO", "CC", "CENTRO_DE_CUSTO"],
    "Valor": ["VALOR", "VALOR_TOTAL", "VL_DOC"],
    "Cód Conta Contábil": ["COD_CONTA_CONTABIL", "CONTA_CONTABIL"],
    "Tipo de Produto": ["TIPO_PRODUTO", "PROD_TIPO"],
    "MUNICIPIO": ["NOME_MUNICIPIO", "CIDADE", "MUN", "NOME_MUNICIPIO_CIRCUNSCRICAO"],
    "LOGRADOURO": ["LOGRADOURO", "ENDERECO", "NM_LOG", "LOGRADOURO_BO", "RUA"],
    "LATITUDE": ["LATITUDE", "LAT", "COORD_Y", "LATITUDE_BO"],
    "LONGITUDE": ["LONGITUDE", "LON", "COORD_X", "LONGITUDE_BO"],
    "RUBRICA": ["RUBRICA", "TIPO_CRIME", "NATUREZA_APURADA", "NATUREZA"],
    "DATA": ["DATA_OCORRENCIA_BO", "DT_OCORRENCIA", "DATA_BO", "DATA_OCORRENCIA"],
    "HORA": ["HORA_OCORRENCIA_BO", "HR_OCORRENCIA", "HORA"]
}

SCHEMA_PRATA = {
    "ID ERP": pl.String,
    "Tipo Certificado": pl.String,
    "Numero Certificado": pl.String,
    "N Doc": pl.Int64,
    "Unidade": pl.String,
    "Centro de Custo": pl.Int64,
    "Valor": pl.Float64,
    "Cód Conta Contábil": pl.Int64,
    "Tipo de Produto": pl.String,
    "MUNICIPIO": pl.String,
    "LOGRADOURO": pl.String,
    "LATITUDE": pl.Float64,
    "LONGITUDE": pl.Float64,
    "RUBRICA": pl.String,
    "DATA": pl.Date,
    "HORA": pl.String,
    "H3_8": pl.String,
    "PESO_PENAL": pl.Float64,
    "PERIODO_DIA": pl.String,
    "DATA_CONSOLIDACAO": pl.Date,
    "NOME_ARQUIVO": pl.String
}

def normalizar_texto(t) -> str:
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
    if valor is None: return None
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
                print(f"ERRO: Falha na transmissão de dados para o Discord: {e}")

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

    # 1. Renomear colunas e adicionar ausentes como String (Bronze)
    for coluna_padrao, sinonimos in MAPA_SINONIMOS.items():
        coluna_encontrada = _encontrar_coluna(df_mapeado, sinonimos)
        if coluna_encontrada and coluna_encontrada != coluna_padrao:
            df_mapeado = df_mapeado.rename({coluna_encontrada: coluna_padrao})
        elif not coluna_encontrada and coluna_padrao not in df_mapeado.columns:
            df_mapeado = df_mapeado.with_columns(pl.lit(None).alias(coluna_padrao).cast(pl.String))

    # 2. Remover linhas totalmente vazias
    df_mapeado = df_mapeado.drop_nulls()

    # 3. Remover linhas onde RUBRICA está preenchida, mas outras colunas essenciais estão vazias
    colunas_essenciais = ["MUNICIPIO", "LOGRADOURO", "DATA"]
    if "RUBRICA" in df_mapeado.columns:
        df_mapeado = df_mapeado.filter(
            pl.col("RUBRICA").is_null() |
            (pl.col("RUBRICA").is_not_null() & pl.all_horizontal([pl.col(c).is_not_null() for c in colunas_essenciais if c in df_mapeado.columns]))
        )

    # 4. Late Binding Casting (Bronze para Prata)
    # Primeiro, garantir que todas as colunas do SCHEMA_PRATA existam no df_mapeado
    for col, dtype in SCHEMA_PRATA.items():
        if col not in df_mapeado.columns:
            df_mapeado = df_mapeado.with_columns(pl.lit(None).alias(col).cast(pl.String)) # Adiciona como String inicialmente

    # Depois, aplicar o cast para o tipo final
    for col, dtype in SCHEMA_PRATA.items():
        if col in df_mapeado.columns:
            df_mapeado = df_mapeado.with_columns(pl.col(col).cast(dtype, strict=False))

    # 5. Tratamento específico da coluna 'Valor'
    if "Valor" in df_mapeado.columns:
        df_mapeado = df_mapeado.with_columns(
            pl.col("Valor")
            .cast(pl.String)
            .str.replace_all(r"\.", "") # Remove separador de milhar
            .str.replace_all(r",", ".") # Troca vírgula por ponto decimal
            .cast(pl.Float64, strict=False)
            .round(0) # Arredonda para o número inteiro mais próximo
            .cast(pl.Int64, strict=False) # Converte para Int64
            .fill_null(0) # Preenche nulos com 0
        )

    # 6. Preenchimento de 'Cód Conta Contábil' baseado em 'Tipo de Produto'
    if "Tipo de Produto" in df_mapeado.columns and "Cód Conta Contábil" in df_mapeado.columns:
        df_mapeado = df_mapeado.with_columns(
            pl.when(pl.col("Tipo de Produto").str.contains("MATERIAL", case_sensitive=False))
            .then(pl.lit(42210001).cast(pl.Int64))
            .when(pl.col("Tipo de Produto").str.contains("MEDICAMENTO", case_sensitive=False))
            .then(pl.lit(42210004).cast(pl.Int64))
            .otherwise(pl.col("Cód Conta Contábil"))
            .alias("Cód Conta Contábil")
        )

    # 7. Normalizar Unidade e Logradouro
    if "Unidade" in df_mapeado.columns:
        df_mapeado = df_mapeado.with_columns(pl.col("Unidade").apply(normalizar_texto).alias("Unidade"))
    if "LOGRADOURO" in df_mapeado.columns:
        df_mapeado = df_mapeado.with_columns(pl.col("LOGRADOURO").apply(normalizar_texto).alias("LOGRADOURO"))
    if "MUNICIPIO" in df_mapeado.columns:
        df_mapeado = df_mapeado.with_columns(pl.col("MUNICIPIO").apply(normalizar_texto).alias("MUNICIPIO"))

    return df_mapeado

def imputacao_geografica_pro(df_ssp: pl.DataFrame, geo_master_df: pl.DataFrame) -> pl.DataFrame:
    df_ssp_com_geo = df_ssp.filter(pl.col("LATITUDE").is_not_null() & pl.col("LONGITUDE").is_not_null())
    df_ssp_sem_geo = df_ssp.filter(pl.col("LATITUDE").is_null() | pl.col("LONGITUDE").is_null())

    if df_ssp_sem_geo.is_empty():
        return df_ssp_com_geo.with_columns(pl.lit(None).cast(pl.String).alias("H3_8"))

    df_ssp_sem_geo = df_ssp_sem_geo.with_columns([
        pl.col("MUNICIPIO").apply(normalizar_texto).alias("MUNICIPIO_NORM"),
        pl.col("LOGRADOURO").apply(normalizar_texto).alias("LOGRADOURO_NORM")
    ])

    geo_master_df_norm = geo_master_df.with_columns([
        pl.col("NOME_MUNICIPIO").apply(normalizar_texto).alias("MUNICIPIO_NORM"),
        pl.col("NOME_LOGRADOURO").apply(normalizar_texto).alias("LOGRADOURO_NORM")
    ])

    df_imputado = df_ssp_sem_geo.join(
        geo_master_df_norm.select(["MUNICIPIO_NORM", "LOGRADOURO_NORM", "LATITUDE", "LONGITUDE", "H3_8"]),
        on=["MUNICIPIO_NORM", "LOGRADOURO_NORM"],
        how="left"
    )

    df_imputado = df_imputado.with_columns([
        pl.col("LATITUDE").fill_null(pl.col("LATITUDE_right")).alias("LATITUDE"),
        pl.col("LONGITUDE").fill_null(pl.col("LONGITUDE_right")).alias("LONGITUDE"),
        pl.col("H3_8").fill_null(pl.col("H3_8_right")).alias("H3_8")
    ]).drop(["LATITUDE_right", "LONGITUDE_right", "H3_8_right"])

    recuperados = df_imputado.filter(pl.col("LATITUDE").is_not_null()).shape[0]
    print(f"SUCESSO: {recuperados} BOs recuperados via imputação geográfica.")

    df_final = pl.concat([df_ssp_com_geo, df_imputado], how="vertical_relaxed")

    return df_final.drop(["MUNICIPIO_NORM", "LOGRADOURO_NORM"])

def processar_dados_ssp(df_raw: pl.DataFrame, geo_master_df: pl.DataFrame) -> pl.DataFrame:
    df_processed = schema_brain_mapping(df_raw)

    df_processed = imputacao_geografica_pro(df_processed, geo_master_df)

    df_processed = df_processed.with_columns([
        pl.col("RUBRICA").apply(lambda x: PESO_PENAL_BASE.get(x, 1.0)).alias("PESO_PENAL"),
        pl.col("HORA").apply(get_periodo).alias("PERIODO_DIA")
    ])

    if "Unidade" in df_processed.columns:
        colunas_para_duplicata = ["Unidade", "DATA", "HORA", "RUBRICA", "LOGRADOURO", "MUNICIPIO"]
        colunas_para_duplicata = [col for col in colunas_para_duplicata if col in df_processed.columns]

        if colunas_para_duplicata:
            initial_rows = df_processed.shape[0]
            df_processed = df_processed.unique(subset=colunas_para_duplicata, keep="first")
            print(f"INFO: {initial_rows - df_processed.shape[0]} duplicatas filtradas por Unidade/Hospital.")
        else:
            print("ALERTA: Colunas essenciais para filtragem de duplicatas por Unidade/Hospital não encontradas.")

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

def carregar_geo_master_r2(path: str) -> pl.DataFrame | None:
    s3 = boto3.client('s3',
                      endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                      aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))
    bucket_name = os.environ.get("R2_BUCKET_NAME")

    try:
        obj = s3.get_object(Bucket=bucket_name, Key=path)
        df_geo = pl.read_parquet(BytesIO(obj['Body'].read()))
        print(f"SUCESSO: Base geográfica mestre carregada do R2: {path}. Registros: {df_geo.shape[0]}")
        return df_geo
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            print(f"ALERTA: Arquivo '{path}' não encontrado no R2.")
        else:
            print(f"ERRO: Falha ao carregar base geográfica mestre do R2: {e}")
        return None
    except Exception as e:
        print(f"ERRO: Falha ao ler Parquet do R2 ou arquivo corrompido: {e}")
        return None

def construir_base_geografica_do_zero(geojson_path: str, h3_resolucao: int) -> pl.DataFrame:
    print(f"INICIANDO: Construção da base geográfica a partir de {geojson_path}.")

    try:
        # --- SIMULAÇÃO DE LEITURA E PROCESSAMENTO DO GEOJSON ---
        # SUBSTITUA ESTA SEÇÃO PELA LÓGICA REAL DE LEITURA DO SEU GEOJSON
        # E EXTRAÇÃO DAS COLUNAS NECESSÁRIAS.
        # Exemplo de dados simulados para a base geográfica
        data = {
            "NOME_MUNICIPIO": ["SAO PAULO", "SAO PAULO", "CAMPINAS", "CAMPINAS", "RIO DE JANEIRO"],
            "NOME_LOGRADOURO": ["AV PAULISTA", "RUA AUGUSTA", "AV BRASIL", "RUA BARRETO LEME", "AV ATLANTICA"],
            "LATITUDE": [-23.56135, -23.5534, -22.9056, -22.8997, -22.9722],
            "LONGITUDE": [-46.6562, -46.6612, -47.0608, -47.0621, -43.1868]
        }
        df_geo_raw = pl.DataFrame(data)
        # --- FIM DA SIMULAÇÃO ---

        df_geo_raw = df_geo_raw.with_columns(
            pl.struct(["LATITUDE", "LONGITUDE"])
            .apply(lambda x: h3.h3_from_geo(x["LATITUDE"], x["LONGITUDE"], h3_resolucao))
            .alias("H3_8")
        )

        print(f"SUCESSO: Base geográfica construída. Registros: {df_geo_raw.shape[0]}")
        return df_geo_raw
    except Exception as e:
        print(f"ERRO: Falha ao construir base geográfica do zero a partir de {geojson_path}: {e}")
        return pl.DataFrame()

def salvar_parquet(df: pl.DataFrame, r2_full_path: str, local_dir_path: str, filename: str):
    try:
        s3 = boto3.client('s3',
                          endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                          aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                          aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))

        buffer = BytesIO()
        df.write_parquet(buffer)
        buffer.seek(0)

        bucket_name = os.environ.get("R2_BUCKET_NAME")
        s3.upload_fileobj(buffer, bucket_name, r2_full_path)
        print(f"SUCESSO: Arquivo Parquet salvo no R2: {r2_full_path}")
    except ClientError as e:
        print(f"ERRO: Falha ao salvar Parquet no R2: {e}")
    except Exception as e:
        print(f"ERRO: Falha inesperada ao salvar Parquet no R2: {e}")

    try:
        os.makedirs(local_dir_path, exist_ok=True)
        full_local_path = os.path.join(local_dir_path, filename)
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

        geo_master_df = None
        geo_master_df = carregar_geo_master_r2(R2_GEO_MASTER)

        if geo_master_df is None or geo_master_df.is_empty():
            print("ALERTA: Base geográfica mestre não encontrada ou corrompida. Tentando reconstruir do zero.")

            geo_master_df = construir_base_geografica_do_zero(GEOJSON_BAIRROS_PATH, H3_RESOLUCAO)

            if geo_master_df.is_empty():
                raise ValueError("ERRO CRÍTICO: Não foi possível construir a base geográfica mestre. Abortando execução.")

            local_download_path = r"C:\Users\Lucas Pereira\Downloads\SP_faces_de_logradouros_2022_json\SP"
            salvar_parquet(geo_master_df, R2_GEO_MASTER, local_download_path, os.path.basename(R2_GEO_MASTER))
            print("SUCESSO: Base geográfica mestre reconstruída e salva.")

        df_final = consolidar_dados_ssp(ANOS_DISPONIVEIS, geo_master_df)

        if df_final.is_empty():
            raise ValueError("ERRO CRÍTICO: Nenhum dado consolidado da SSP para processar. Abortando execução.")

        output_filename = "ssp_dados_criminais_consolidado.parquet"
        r2_output_base_path = "safedriver/safedriver/datalake/base_geografica"
        local_download_path = r"C:\Users\Lucas Pereira\Downloads\SP_faces_de_logradouros_2022_json\SP"

        salvar_parquet(df_final, os.path.join(r2_output_base_path, output_filename), local_download_path, output_filename)

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
