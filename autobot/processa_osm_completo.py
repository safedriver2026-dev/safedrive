import os, duckdb, polars as pl, boto3, io, requests, zipfile, unicodedata, sys, traceback, shutil
import h3
from botocore.config import Config

def remover_acentos(texto):
    if texto is None or str(texto).upper() in ["NAN", "", "NULL"]: return "NAO MAPEADO"
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def get_h3_int(lat, lon):
    try: return int(h3.latlng_to_cell(lat, lon, 9), 16)
    except: return 0

def pipeline_osm_unificado():
    R2_CONF = {"endpoint_url": os.getenv("R2_ENDPOINT_URL"), "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(), "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(), "config": Config(region_name="auto")}
    BUCKET = os.getenv("R2_BUCKET_NAME")
    CAMINHO_MALHA = "malha geografica/malha_mestra_consolidada_2025.parquet"
    CAMINHO_SOCIAL = "malha geografica social/pontos_interesse_osm.parquet"
    
    URL_OSM_SP = "https://download.geofabrik.de/south-america/brazil/sao-paulo-latest-free.shp.zip"
    PASTA_TEMP = "temp_osm_unificado"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        print("[LOG] Iniciando Enriquecimento Unificado (Download unico)...")

        # 1. DOWNLOAD UNICO
        os.makedirs(PASTA_TEMP, exist_ok=True)
        r = requests.get(URL_OSM_SP, stream=True)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            alvos = ['gis_osm_places_free_1', 'gis_osm_roads_free_1', 'gis_osm_water_a_free_1', 'gis_osm_transport_free_1', 'gis_osm_pois_free_1']
            for f in z.namelist():
                if any(x in f for x in alvos): z.extract(f, PASTA_TEMP)

        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # 2. CARREGAR MALHA GERADA PELO IBGE
        obj = r2.get_object(Bucket=BUCKET, Key=CAMINHO_MALHA)
        df_malha = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        # --- PARTE A: REFERENCIA (TOPOLOGIA) ---
        print("[LOG] Processando Hidrografia e Vias...")
        # Hidrografia
        df_agua = con.execute(f"SELECT ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('{PASTA_TEMP}/gis_osm_water_a_free_1.shp')").pl()
        df_agua = df_agua.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int"))
        df_malha = df_malha.join(df_agua.unique("id_h3_int").select([pl.col("id_h3_int"), pl.lit(1).alias("tem_agua")]), on="id_h3_int", how="left").with_columns(pl.col("tem_agua").fill_null(0))

        # --- PARTE B: TAPA-BURACOS (BAIRROS) ---
        print("[LOG] Executando Fallback de Bairros (NAO MAPEADO)...")
        df_places = con.execute(f"SELECT name, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('{PASTA_TEMP}/gis_osm_places_free_1.shp')").pl()
        df_places = df_places.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int"))
        df_places = df_places.select([pl.col("id_h3_int"), pl.col("name").map_elements(remover_acentos, return_dtype=pl.String).alias("osm_bairro")]).unique("id_h3_int")
        
        df_malha = df_malha.join(df_places, on="id_h3_int", how="left")
        df_malha = df_malha.with_columns(pl.when(pl.col("bairro") == "NAO MAPEADO").then(pl.col("osm_bairro").fill_null("NAO MAPEADO")).otherwise(pl.col("bairro")).alias("bairro")).drop("osm_bairro")

        # --- PARTE C: MALHA SOCIAL (SOCIAL) ---
        print("[LOG] Gerando Camada Social Segregada...")
        df_soc = con.execute(f"""
            SELECT fclass, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('{PASTA_TEMP}/gis_osm_transport_free_1.shp') WHERE fclass IN ('bus_stop', 'railway_station')
            UNION ALL
            SELECT fclass, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('{PASTA_TEMP}/gis_osm_pois_free_1.shp') WHERE fclass IN ('police', 'bank', 'bar', 'hospital', 'fuel')
        """).pl()
        df_soc = df_soc.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int"))
        df_social_final = df_soc.pivot(values="fclass", index="id_h3_int", columns="fclass", aggregate_function="len").fill_null(0)

        # 3. UPLOAD E LIMPEZA
        print("[LOG] Salvando tudo no R2...")
        for df, path in [(df_malha, CAMINHO_MALHA), (df_social_final, CAMINHO_SOCIAL)]:
            buf = io.BytesIO(); df.write_parquet(buf, compression="zstd"); buf.seek(0)
            r2.upload_fileobj(buf, BUCKET, path)

        shutil.rmtree(PASTA_TEMP)
        print("[LOG] Pipeline concluido com sucesso!")

    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    pipeline_osm_unificado()
