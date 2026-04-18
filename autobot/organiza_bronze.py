import os, duckdb, polars as pl, boto3, io, unicodedata, sys, traceback
import h3
from botocore.config import Config

def remover_acentos(texto):
    if texto is None or str(texto).upper() in ["NAN", "", "NULL", "NONE"]: return None
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def get_h3_int(lat, lon):
    try: return int(h3.latlng_to_cell(lat, lon, 9), 16)
    except: return 0

def pipeline_bronze_master():
    print("🚀 [INICIO] Conserto Crítico: Removendo Duplicatas e Refinando Match")
    
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    PASTA_OSM = "temp_estadual"
    DEST_FILE = "datalake/bronze/malha_geografica/malha_mestra_bronze.parquet"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # 1. CARREGAR MALHA MESTRA
        obj = r2.get_object(Bucket=BUCKET, Key="malha geografica/malha_mestra_consolidada_2025.parquet")
        df_mestra = pl.read_parquet(io.BytesIO(obj['Body'].read())).with_columns(pl.col("id_h3_int").cast(pl.UInt64))

        # 2. MELHORAR O MATCH DE LOGRADOURO
        # Em vez de pegar apenas o centro da rua inteira, vamos extrair os pontos das extremidades e centro 
        # para aumentar as chances de 'bater' com os hexágonos.
        path_roads = f"{PASTA_OSM}/gis_osm_roads_free_1.shp"
        if os.path.exists(path_roads):
            print("[LOG] Processando Logradouros (Estratégia Multidirecional)...")
            df_roads = con.execute(f"""
                SELECT name as osm_rua, ST_Y(ST_StartPoint(geom)) as lat, ST_X(ST_StartPoint(geom)) as lon FROM ST_Read('{path_roads}') WHERE name IS NOT NULL
                UNION
                SELECT name as osm_rua, ST_Y(ST_EndPoint(geom)) as lat, ST_X(ST_EndPoint(geom)) as lon FROM ST_Read('{path_roads}') WHERE name IS NOT NULL
            """).pl()
            
            df_roads_h3 = df_roads.with_columns(
                pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")
            ).unique("id_h3_int").select(["id_h3_int", "osm_rua"])

            df_mestra = df_mestra.join(df_roads_h3, on="id_h3_int", how="left").with_columns(
                pl.col("logradouro").fill_null(pl.col("osm_rua")).fill_null("LOGRADOURO NAO IDENTIFICADO")
            ).drop("osm_rua")

        # 3. GARANTIR UNICIDADE ABSOLUTA (Mata as 407k duplicatas)
        print(f"[LOG] Removendo duplicatas... (Antes: {len(df_mestra)})")
        df_mestra = df_mestra.unique(subset=["id_h3_int"], keep="first")
        print(f"[LOG] Unicidade garantida. (Depois: {len(df_mestra)})")

        # 4. SALVAMENTO
        buf = io.BytesIO()
        df_mestra.write_parquet(buf, compression="zstd")
        buf.seek(0)
        r2.upload_fileobj(buf, BUCKET, DEST_FILE)
        
        print("\n✅ [SUCESSO] Malha corrigida e sem duplicatas.")

    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    pipeline_bronze_master()
