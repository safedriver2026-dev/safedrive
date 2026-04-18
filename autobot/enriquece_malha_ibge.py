import os, duckdb, polars as pl, boto3, io, requests, zipfile, unicodedata, urllib3, sys, traceback
from botocore.config import Config

# Ignora avisos de SSL do servidor do IBGE
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def remover_acentos(texto):
    if texto is None or str(texto).upper() in ["NAN", "", "NULL"]: 
        return "NAO MAPEADO"
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def pipeline_enriquecimento():
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")

    # URL oficial do Censo 2022 - Setores (SP)
    url_setores = "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_de_setores_censitarios__divisoes_intramunicipais/censo_2022/setores/shp/UF/SP/SP_Setores_2022.zip"
    
    try:
        print("🌐 1. Baixando dados de Setores direto do IBGE...")
        r = requests.get(url_setores, verify=False, timeout=120)
        r.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall("temp_setores")

        print("🛠️ 2. Extraindo Bairros e Distritos com DuckDB...")
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        
        # Lemos apenas as colunas de texto (DBF), ignorando a geometria para ser veloz
        df_ibge = con.execute("""
            SELECT 
                CAST(CD_SETOR AS VARCHAR) as setor_id,
                NM_BAIRRO as bairro_raw,
                NM_DIST as distrito,
                NM_SUBDIST as subdistrito
            FROM ST_Read('temp_setores/SP_Setores_2022.shp')
        """).pl().unique(subset=["setor_id"])

        print("📥 3. Descarregando Malha Mestra do R2...")
        r2 = boto3.client("s3", **R2_CONF)
        obj = r2.get_object(Bucket=BUCKET, Key="ouro/malha_mestra_consolidada_2025.parquet")
        df_malha = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # Limpa coluna antiga de bairro se existir
        if "bairro" in df_malha.columns:
            df_malha = df_malha.drop("bairro")

        print("🧩 4. Executando Join de Atributos...")
        df_final = df_malha.join(df_ibge, on="setor_id", how="left")
        
        # Formatação final
        df_final = df_final.with_columns([
            pl.col("bairro_raw").fill_null("NAO MAPEADO").map_elements(remover_acentos, return_dtype=pl.String).alias("bairro"),
            pl.col("distrito").map_elements(remover_acentos, return_dtype=pl.String),
            pl.col("subdistrito").map_elements(remover_acentos, return_dtype=pl.String)
        ]).drop("bairro_raw")

        print("💾 5. Fazendo upload da Malha Enriquecida...")
        buffer = io.BytesIO()
        df_final.write_parquet(buffer, compression="zstd")
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, "ouro/malha_mestra_consolidada_2025.parquet")
        
        print("✅ Processo concluído com sucesso!")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    pipeline_enriquecimento()
