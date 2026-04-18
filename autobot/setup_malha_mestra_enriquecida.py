import os, duckdb, polars as pl, boto3, io, unicodedata, sys, traceback, shutil
from botocore.config import Config

def remover_acentos(texto):
    if texto is None or str(texto).upper() in ["NAN", "", "NULL"]: 
        return "NAO MAPEADO"
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def pipeline_final():
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    
    # Caminhos definidos conforme os arquivos subidos
    PASTA_LOCAL = "dados_ibge/setores"
    ARQUIVO_SHP = f"{PASTA_LOCAL}/SP_setores_CD2022.shp"
    
    ORIGEM_R2 = "ouro/malha_mestra_consolidada_2025.parquet"
    DESTINO_R2 = "malha geografica/malha_mestra_consolidada_2025.parquet"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        print("🔍 [LOG PROVA] Iniciando processamento da Malha Geográfica...")

        # 1. Leitura dos Atributos Locais (Setores 2022)
        if not os.path.exists(ARQUIVO_SHP):
            print(f"🚨 [ERRO] Arquivo nao encontrado: {ARQUIVO_SHP}")
            sys.exit(1)

        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        
        print(f"📖 [LOG PROVA] Extraindo NM_BAIRRO e NM_DIST de {ARQUIVO_SHP}...")
        df_ibge = con.execute(f"""
            SELECT 
                CAST(CD_SETOR AS VARCHAR) as setor_id,
                NM_BAIRRO as bairro_raw,
                NM_DIST as distrito,
                NM_SUBDIST as subdistrito
            FROM ST_Read('{ARQUIVO_SHP}')
        """).pl().unique(subset=["setor_id"])
        
        print(f"✅ [LOG PROVA] {len(df_ibge)} setores carregados da base local.")

        # 2. Download da Malha Mestra para Enriquecimento
        print(f"📥 [LOG PROVA] Buscando malha base em '{ORIGEM_R2}'...")
        obj = r2.get_object(Bucket=BUCKET, Key=ORIGEM_R2)
        df_malha = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        if "bairro" in df_malha.columns:
            df_malha = df_malha.drop("bairro")

        # 3. Join e Higienização
        print("🧩 [LOG PROVA] Executando Join e normalizacao de strings...")
        df_final = df_malha.join(df_ibge, on="setor_id", how="left")
        
        df_final = df_final.with_columns([
            pl.col("bairro_raw").fill_null("NAO MAPEADO").map_elements(remover_acentos, return_dtype=pl.String).alias("bairro"),
            pl.col("distrito").fill_null("NAO MAPEADO").map_elements(remover_acentos, return_dtype=pl.String),
            pl.col("subdistrito").fill_null("NAO MAPEADO").map_elements(remover_acentos, return_dtype=pl.String)
        ]).drop("bairro_raw")

        # 4. Upload para a Nova Pasta e Limpeza da Antiga
        print(f"💾 [LOG PROVA] Salvando nova malha enriquecida em '{DESTINO_R2}'...")
        buffer = io.BytesIO()
        df_final.write_parquet(buffer, compression="zstd")
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, DESTINO_R2)

        print(f"🗑️ [LOG PROVA] Removendo arquivo obsoleto da pasta 'ouro'...")
        r2.delete_object(Bucket=BUCKET, Key=ORIGEM_R2)

        # 5. Expurgo dos Arquivos Locais
        if os.path.exists(PASTA_LOCAL):
            shutil.rmtree(PASTA_LOCAL)
            print(f"✨ [LOG PROVA] Pasta '{PASTA_LOCAL}' deletada. Ambiente limpo.")

        print("🏆 [LOG PROVA] Enriquecimento e migracao concluidos com sucesso!")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    pipeline_final()
