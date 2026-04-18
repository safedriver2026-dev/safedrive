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
    print("🚀 [INICIO] Consolidação Bronze: Idempotência + Match Agressivo")
    
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    PASTA_OSM = "temp_estadual"
    BASE_PATH = "datalake/bronze/malha_geografica/"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # --- 1. BUSCA RESILIENTE (Resolve o NoSuchKey) ---
        key_bronze = f"{BASE_PATH}malha_mestra_bronze.parquet"
        key_landing = "malha geografica/malha_mestra_consolidada_2025.parquet"
        
        try:
            print(f"[LOG] Tentando carregar da Bronze: {key_bronze}")
            obj = r2.get_object(Bucket=BUCKET, Key=key_bronze)
        except r2.exceptions.NoSuchKey:
            print(f"[LOG] Bronze não encontrada. Buscando na Landing: {key_landing}")
            obj = r2.get_object(Bucket=BUCKET, Key=key_landing)
        
        df_mestra = pl.read_parquet(io.BytesIO(obj['Body'].read())).with_columns(pl.col("id_h3_int").cast(pl.UInt64))

        # --- 2. MATCH AGRESSIVO (RUA, CIDADE, BAIRRO) ---
        def safe_query(file, query):
            path = f"{PASTA_OSM}/{file}"
            return con.execute(query.replace("PATH", path)).pl() if os.path.exists(path) else None

        # Logradouros: Pegamos Início, Fim e Meio da rua para aumentar o match
        df_roads = safe_query("gis_osm_roads_free_1.shp", """
            SELECT name as osm_rua, ST_Y(ST_StartPoint(geom)) as lat, ST_X(ST_StartPoint(geom)) as lon FROM ST_Read('PATH') WHERE name IS NOT NULL
            UNION ALL
            SELECT name as osm_rua, ST_Y(ST_EndPoint(geom)) as lat, ST_X(ST_EndPoint(geom)) as lon FROM ST_Read('PATH') WHERE name IS NOT NULL
            UNION ALL
            SELECT name as osm_rua, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('PATH') WHERE name IS NOT NULL
        """)

        if df_roads is not None:
            df_roads_h3 = df_roads.with_columns(
                pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")
            ).unique("id_h3_int").select(["id_h3_int", "osm_rua"])
            
            df_mestra = df_mestra.join(df_roads_h3, on="id_h3_int", how="left").with_columns(
                pl.col("logradouro").fill_null(pl.col("osm_rua")).fill_null("LOGRADOURO NAO IDENTIFICADO")
            ).drop("osm_rua")

        # --- 3. INFRAESTRUTURA (O QUE ESTAVA FALTANDO) ---
        infra_dfs = []
        df_pois = safe_query("gis_osm_pois_free_1.shp", "SELECT fclass, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('PATH') WHERE fclass IN ('police', 'hospital', 'bar', 'bank', 'fuel')")
        if df_pois is not None:
            infra_dfs.append(df_pois.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).pivot(on="fclass", index="id_h3_int", values="fclass", aggregate_function="len").fill_null(0))

        df_water = safe_query("gis_osm_water_free_1.shp", "SELECT 1 as presenca_agua, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('PATH')")
        if df_water is not None:
            infra_dfs.append(df_water.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).unique("id_h3_int").select(["id_h3_int", "presenca_agua"]))

        df_infra = infra_dfs[0] if infra_dfs else pl.DataFrame({"id_h3_int": [pl.Series([], dtype=pl.UInt64)]})
        for d in infra_dfs[1:]: df_infra = df_infra.join(d, on="id_h3_int", how="outer").fill_null(0)

        # --- 4. LIMPEZA DE DUPLICATAS E SALVAMENTO ---
        # Mata as 407k duplicatas garantindo uma linha por hexágono
        df_mestra = df_mestra.unique(subset=["id_h3_int"], keep="first")

        for d, name in [(df_mestra, "malha_mestra_bronze.parquet"), (df_infra, "malha_geografica_infraestrutura.parquet")]:
            buf = io.BytesIO()
            d.write_parquet(buf, compression="zstd")
            buf.seek(0)
            r2.upload_fileobj(buf, BUCKET, f"{BASE_PATH}{name}")

        print(f"\n✅ [SUCESSO] Malha consolidada: {len(df_mestra):,} registros únicos.")

    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    pipeline_bronze_master()
