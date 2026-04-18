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

def pipeline_bronze_master():
    print("🚀 [INICIO] Processamento de Alta Performance: Malha Geográfica + Infraestrutura")
    
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    PASTA_OSM = "temp_estadual"
    
    DESTINO_FOLDER = "datalake/bronze/malha geografica/"
    FILE_MESTRA = f"{DESTINO_FOLDER}malha_mestra_bronze.parquet"
    FILE_INFRA = f"{DESTINO_FOLDER}malha_geografica_infraestrutura.parquet"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # --- 1. CARREGAR MALHA BASE ---
        obj = r2.get_object(Bucket=BUCKET, Key="malha geografica/malha_mestra_consolidada_2025.parquet")
        df_base = pl.read_parquet(io.BytesIO(obj['Body'].read())).with_columns(pl.col("id_h3_int").cast(pl.UInt64))

        # --- 2. ENRIQUECIMENTO GEOGRÁFICO (MALHA MESTRA) ---
        print("[LOG] Refinando Geografia e Endereçamento...")
        
        df_roads_raw = con.execute(f"SELECT name as osm_rua, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('{PASTA_OSM}/gis_osm_roads_free_1.shp') WHERE name IS NOT NULL").pl()
        df_roads_h3 = df_roads_raw.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).unique("id_h3_int").select(["id_h3_int", "osm_rua"])

        df_places = con.execute(f"SELECT name as osm_nome, fclass, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('{PASTA_OSM}/gis_osm_places_free_1.shp') WHERE name IS NOT NULL AND fclass IN ('suburb', 'neighbourhood')").pl()
        osm_bairros = df_places.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).unique("id_h3_int").select(["id_h3_int", "osm_nome"])

        df_mestra_final = df_base.join(df_roads_h3, on="id_h3_int", how="left").join(osm_bairros.rename({"osm_nome": "resgate_bairro"}), on="id_h3_int", how="left")
        df_mestra_final = df_mestra_final.with_columns([
            pl.when(pl.col("logradouro").is_null() | (pl.col("logradouro") == "LOGRADOURO NAO IDENTIFICADO")).then(pl.col("osm_rua").fill_null("LOGRADOURO NAO IDENTIFICADO")).otherwise(pl.col("logradouro")).alias("logradouro"),
            pl.when(pl.col("bairro") == "NAO MAPEADO").then(pl.col("resgate_bairro").map_elements(remover_acentos, return_dtype=pl.String)).otherwise(pl.col("bairro")).alias("bairro")
        ]).drop(["osm_rua", "resgate_bairro"])

        # --- 3. ENRIQUECIMENTO TÉCNICO (INFRAESTRUTURA) ---
        print("[LOG] Extraindo Indicadores Urbanos (Prédios, POIs e Água)...")

        df_buildings = con.execute(f"SELECT count(*) as qtd_predios, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('{PASTA_OSM}/gis_osm_buildings_free_1.shp') GROUP BY geom").pl()
        df_buildings_h3 = df_buildings.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).group_by("id_h3_int").agg(pl.col("qtd_predios").sum())

        df_pois = con.execute(f"SELECT fclass, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('{PASTA_OSM}/gis_osm_pois_free_1.shp') WHERE fclass IN ('police', 'hospital', 'bar', 'bank', 'fuel')").pl()
        df_pois_h3 = df_pois.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).pivot(values="fclass", index="id_h3_int", columns="fclass", aggregate_function="len").fill_null(0)

        df_water = con.execute(f"SELECT 1 as presenca_agua, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('{PASTA_OSM}/gis_osm_water_free_1.shp')").pl()
        df_water_h3 = df_water.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).unique("id_h3_int").select(["id_h3_int", "presenca_agua"])

        df_infra_final = df_buildings_h3.join(df_pois_h3, on="id_h3_int", how="outer").join(df_water_h3, on="id_h3_int", how="left").fill_null(0)

        # --- 4. PERSISTÊNCIA ---
        for df, path in [(df_mestra_final, FILE_MESTRA), (df_infra_final, FILE_INFRA)]:
            buf = io.BytesIO()
            df.write_parquet(buf, compression="zstd")
            buf.seek(0)
            r2.upload_fileobj(buf, BUCKET, path)

        # --- 5. EXIBIÇÃO DE AMOSTRA FINAL (LOG) ---
        print("\n" + "═"*70)
        print("📊 AMOSTRA: MALHA MESTRA BRONZE (GEOGRAFIA)")
        print("═"*70)
        with pl.Config(tbl_rows=8, tbl_width_chars=120):
            print(df_mestra_final.select(["id_h3_int", "setor_id", "logradouro", "bairro", "cidade_nome"]).head(8))

        print("\n" + "═"*70)
        print("📊 AMOSTRA: MALHA DE INFRAESTRUTURA BRONZE")
        print("═"*70)
        # Seleciona id e algumas colunas de infra para visualização
        cols_infra = [c for c in ["id_h3_int", "qtd_predios", "police", "bar", "presenca_agua"] if c in df_infra_final.columns]
        with pl.Config(tbl_rows=8, tbl_width_chars=120):
            # Mostra registros onde existe pelo menos alguma infraestrutura (não apenas zeros)
            print(df_infra_final.filter(pl.col("qtd_predios") > 0).select(cols_infra).head(8))
        
        print("\n" + "✅ [SUCESSO] Pipeline finalizado. Dados organizados em datalake/bronze/.")
        print("═"*70)

    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    pipeline_bronze_master()
