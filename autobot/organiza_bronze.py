import os, duckdb, polars as pl, boto3, io, unicodedata, sys, traceback
import h3
from botocore.config import Config

def remover_acentos(texto):
    if texto is None or str(texto).upper() in ["NAN", "", "NULL", "NONE"]: return "NAO MAPEADO"
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def get_h3_int(lat, lon):
    try: return int(h3.latlng_to_cell(lat, lon, 9), 16)
    except: return 0

def pipeline_bronze_final():
    print("🚀 [INICIO] Consolidação Bronze: Enriquecimento e Validação de Chaves")
    
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    
    # --- CAMINHOS ---
    PATH_ORIGEM_MESTRA = "malha geografica/malha_mestra_consolidada_2025.parquet"
    PATH_ORIGEM_INFRA = "datalake/bronze/malha_geografica_infraestrutura.parquet"
    
    DESTINO_BRONZE = "datalake/bronze/malha geografica/"
    FILE_MESTRA_BRONZE = f"{DESTINO_BRONZE}malha_mestra_bronze.parquet"
    FILE_INFRA_BRONZE = f"{DESTINO_BRONZE}malha_geografica_infraestrutura.parquet"
    
    PASTA_OSM = "temp_estadual"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # --- 1. CARREGAR MALHA MESTRA E VALIDAR CHAVE ---
        print("[LOG] Baixando Malha Mestra para Auditoria...")
        obj = r2.get_object(Bucket=BUCKET, Key=PATH_ORIGEM_MESTRA)
        df_base = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # Garante que a chave H3 seja UInt64 para o Join ser ultra-rápido
        df_base = df_base.with_columns(pl.col("id_h3_int").cast(pl.UInt64))
        
        stats_antes = {
            "logradouro": df_base.filter(pl.col("logradouro").is_null() | (pl.col("logradouro") == "LOGRADOURO NAO IDENTIFICADO")).height,
            "bairro": df_base.filter(pl.col("bairro") == "NAO MAPEADO").height
        }

        # --- 2. RESGATE OSM (DETECTA AUTOMATICAMENTE SP OU SUDESTE) ---
        print("[LOG] Extraindo nomes de ruas e bairros do OSM (Shapefiles)...")
        
        # O DuckDB lê qualquer arquivo .shp na pasta, independente de ser da base SP ou Sudeste
        query_roads = f"SELECT name as osm_rua, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('{PASTA_OSM}/gis_osm_roads_free_1.shp') WHERE name IS NOT NULL"
        df_roads = con.execute(query_roads).pl()

        # Mapeamento H3 (Garante consistência de tipo)
        df_roads_h3 = df_roads.with_columns(
            pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")
        ).unique("id_h3_int").select(["id_h3_int", "osm_rua"])

        # --- 3. JOIN E VALIDAÇÃO DE EFETIVIDADE ---
        print("[LOG] Cruzando bases (Left Join Inteligente)...")
        df_final = df_base.join(df_roads_h3, on="id_h3_int", how="left")

        df_final = df_final.with_columns(
            pl.when(pl.col("logradouro").is_null() | (pl.col("logradouro") == "LOGRADOURO NAO IDENTIFICADO"))
            .then(pl.col("osm_rua").fill_null("LOGRADOURO NAO IDENTIFICADO"))
            .otherwise(pl.col("logradouro")).alias("logradouro")
        ).drop("osm_rua")

        # --- 4. ORGANIZAÇÃO DOS ARQUIVOS NO R2 ---
        print("[LOG] Preparando upload da Malha Bronze Refatorada...")
        buf_mestra = io.BytesIO()
        df_final.write_parquet(buf_mestra, compression="zstd")
        buf_mestra.seek(0)
        r2.upload_fileobj(buf_mestra, BUCKET, FILE_MESTRA_BRONZE)

        # Reposicionar a Malha de Infraestrutura (Bronze)
        print("[LOG] Movendo Infraestrutura para a nova estrutura...")
        obj_inf = r2.get_object(Bucket=BUCKET, Key=PATH_ORIGEM_INFRA)
        r2.upload_fileobj(io.BytesIO(obj_inf['Body'].read()), BUCKET, FILE_INFRA_BRONZE)

        # --- 5. LIMPEZA DOS ARQUIVOS ANTIGOS ---
        print("[LOG] Limpando Landing Zone (Arquivos obsoletos)...")
        for key in [PATH_ORIGEM_MESTRA, PATH_ORIGEM_INFRA]:
            try: r2.delete_object(Bucket=BUCKET, Key=key)
            except: pass

        # --- 6. RELATÓRIO DE EFETIVIDADE ---
        stats_depois = {
            "logradouro": df_final.filter(pl.col("logradouro") == "LOGRADOURO NAO IDENTIFICADO").height,
            "bairro": df_final.filter(pl.col("bairro") == "NAO MAPEADO").height
        }
        
        recuperados = stats_antes['logradouro'] - stats_depois['logradouro']
        taxa_sucesso = (recuperados / stats_antes['logradouro'] * 100) if stats_antes['logradouro'] > 0 else 0

        print("\n" + "═"*50)
        print("🏆  RELATÓRIO DE EFETIVIDADE BRONZE")
        print("═"*50)
        print(f"📊  Total de Hexágonos:   {len(df_final)}")
        print(f"🛣️   Logradouros Iniciais: {stats_antes['logradouro']}")
        print(f"✅  Recuperados via OSM:  {recuperados}")
        print(f"📈  Efetividade:          {taxa_sucesso:.2f}%")
        print(f"📁  Caminho Final:        {FILE_MESTRA_BRONZE}")
        print("═"*50 + "\n")

    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    pipeline_bronze_final()
