import os, duckdb, polars as pl, boto3, io, unicodedata, sys, traceback
import h3
from botocore.config import Config

# --- UTILITÁRIOS ---
def remover_acentos(texto):
    if texto is None or str(texto).upper() in ["NAN", "", "NULL", "NONE"]: return None
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def get_h3_int(lat, lon):
    try: return int(h3.latlng_to_cell(lat, lon, 9), 16)
    except: return 0

def pipeline_bronze_master():
    print("🚀 [INICIO] Consolidação Bronze: Recuperação de Endereçamento Total")
    
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    PASTA_OSM = "temp_estadual"
    DEST_FOLDER = "datalake/bronze/malha_geografica/"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # 1. CARREGAR MALHA MESTRA
        obj = r2.get_object(Bucket=BUCKET, Key="malha geografica/malha_mestra_consolidada_2025.parquet")
        df_mestra = pl.read_parquet(io.BytesIO(obj['Body'].read())).with_columns(pl.col("id_h3_int").cast(pl.UInt64))

        def safe_load(file, query):
            path = f"{PASTA_OSM}/{file}"
            if os.path.exists(path): return con.execute(query.replace("PATH", path)).pl()
            return None

        # 2. RESGATE DE LOGRADOURO (ROADS)
        df_roads = safe_load("gis_osm_roads_free_1.shp", 
            "SELECT name as osm_rua, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('PATH') WHERE name IS NOT NULL")
        
        if df_roads is not None:
            df_roads_h3 = df_roads.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).unique("id_h3_int").select(["id_h3_int", "osm_rua"])
            df_mestra = df_mestra.join(df_roads_h3, on="id_h3_int", how="left").with_columns(
                pl.col("logradouro").fill_null(pl.col("osm_rua").fill_null("LOGRADOURO NAO IDENTIFICADO"))
            ).drop("osm_rua")

        # 3. RESGATE DE CIDADE E BAIRRO (PLACES)
        df_places = safe_load("gis_osm_places_free_1.shp", 
            "SELECT name as osm_nome, fclass, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('PATH') WHERE name IS NOT NULL")
        
        if df_places is not None:
            df_p_h3 = df_places.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int"))
            
            # Filtros específicos
            osm_cidades = df_p_h3.filter(pl.col("fclass").is_in(['city', 'town', 'village'])).unique("id_h3_int").select(["id_h3_int", "osm_nome"])
            osm_bairros = df_p_h3.filter(pl.col("fclass").is_in(['suburb', 'neighbourhood'])).unique("id_h3_int").select(["id_h3_int", "osm_nome"])

            # Join e Fallback Cidade
            df_mestra = df_mestra.join(osm_cidades.rename({"osm_nome": "res_city"}), on="id_h3_int", how="left").with_columns(
                pl.col("cidade_nome").fill_null(pl.col("res_city").map_elements(remover_acentos, return_dtype=pl.String))
            ).drop("res_city")

            # Join e Fallback Bairro
            df_mestra = df_mestra.join(osm_bairros.rename({"osm_nome": "res_bairro"}), on="id_h3_int", how="left").with_columns(
                pl.col("bairro").map_elements(lambda x: x if x != "NAO MAPEADO" else None, return_dtype=pl.String)
            ).with_columns(
                pl.col("bairro").fill_null(pl.col("res_bairro").map_elements(remover_acentos, return_dtype=pl.String))
            ).fill_null("NAO MAPEADO").drop("res_bairro")

        # 4. SALVAMENTO E RELATÓRIO DE METADADOS
        path_final = f"{DEST_FOLDER}malha_mestra_bronze.parquet"
        buf = io.BytesIO()
        df_mestra.write_parquet(buf, compression="zstd")
        buf.seek(0)
        r2.upload_fileobj(buf, BUCKET, path_final)

        print("\n" + "═"*70 + "\n📑 METADADOS DA MALHA RECURSIVA\n" + "═"*70)
        print(f"✅ Arquivo: {path_final}")
        print(f"📊 Shape: {df_mestra.shape}")
        print(f"🧬 Schema: {dict(df_mestra.schema)}")
        print(f"🔍 Nulos em Logradouro: {df_mestra.filter(pl.col('logradouro') == 'LOGRADOURO NAO IDENTIFICADO').height}")
        print(f"🔍 Nulos em Cidade: {df_mestra.filter(pl.col('cidade_nome').is_null()).height}")
        print("═"*70)

    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    pipeline_bronze_master()
