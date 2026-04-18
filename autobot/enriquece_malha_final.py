import os, duckdb, polars as pl, boto3, io, unicodedata, sys, traceback
from botocore.config import Config

def remover_acentos(texto):
    if texto is None or str(texto).upper() in ["NAN", "", "NULL"]: 
        return "NAO MAPEADO"
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def pipeline_migracao_final():
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    
    # Nomes dos ficheiros que carregou no R2
    PREFIXO_IBGE = "SP_setores_CD2022" 
    CAMINHO_MALHA_ANTIGA = "ouro/malha_mestra_consolidada_2025.parquet"
    CAMINHO_MALHA_NOVA = "malha geografica/malha_mestra_consolidada_2025.parquet"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        print("🔍 [LOG PROVA] Iniciando migração e enriquecimento...")

        # 1. Download dos ficheiros Shapefile/DBF do R2 para a máquina local do GitHub
        for extensao in ['shp', 'dbf', 'shx', 'prj', 'cpg']:
            chave_r2 = f"ouro/{PREFIXO_IBGE}.{extensao}" # Assumindo que os colocou na pasta ouro
            print(f"📥 Baixando {chave_r2}...")
            r2.download_file(BUCKET, chave_r2, f"temp_ibge.{extensao}")

        # 2. Extração de Dados com DuckDB
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        
        print("📖 [LOG PROVA] Lendo atributos do Censo 2022...")
        df_ibge = con.execute("""
            SELECT 
                CAST(CD_SETOR AS VARCHAR) as setor_id,
                NM_BAIRRO as bairro_raw,
                NM_DIST as distrito,
                NM_SUBDIST as subdistrito
            FROM ST_Read('temp_ibge.shp')
        """).pl().unique(subset=["setor_id"])

        # 3. Download da Malha Mestra
        print(f"📥 Baixando malha antiga de '{CAMINHO_MALHA_ANTIGA}'...")
        obj = r2.get_object(Bucket=BUCKET, Key=CAMINHO_MALHA_ANTIGA)
        df_malha = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        if "bairro" in df_malha.columns:
            df_malha = df_malha.drop("bairro")

        # 4. Join e Higienização
        print("🧩 [LOG PROVA] Unificando dados espaciais e nomes de bairros...")
        df_final = df_malha.join(df_ibge, on="setor_id", how="left")
        
        df_final = df_final.with_columns([
            pl.col("bairro_raw").fill_null("NAO MAPEADO").map_elements(remover_acentos, return_dtype=pl.String).alias("bairro"),
            pl.col("distrito").fill_null("NAO MAPEADO").map_elements(remover_acentos, return_dtype=pl.String),
            pl.col("subdistrito").fill_null("NAO MAPEADO").map_elements(remover_acentos, return_dtype=pl.String)
        ]).drop("bairro_raw")

        # 5. Upload para a NOVA PASTA e Limpeza
        print(f"💾 Salvando nova malha em '{CAMINHO_MALHA_NOVA}'...")
        buffer = io.BytesIO()
        df_final.write_parquet(buffer, compression="zstd")
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, CAMINHO_MALHA_NOVA)

        print("🧹 [LOG PROVA] Eliminando ficheiros antigos e temporários...")
        # Apaga a malha antiga e os ficheiros de setores do R2
        r2.delete_object(Bucket=BUCKET, Key=CAMINHO_MALHA_ANTIGA)
        for extensao in ['shp', 'dbf', 'shx', 'prj', 'cpg']:
            r2.delete_object(Bucket=BUCKET, Key=f"ouro/{PREFIXO_IBGE}.{extensao}")

        print("✨ [LOG PROVA] Sucesso! Malha Geográfica consolidada e pasta 'ouro' limpa.")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    pipeline_migracao_final()
