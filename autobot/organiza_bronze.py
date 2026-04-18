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
    print("🚀 [INICIO] Consolidação Bronze: Geografia + Infraestrutura (Foco: Estabilidade)")
    
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

        # --- FUNÇÃO DE LEITURA COM METADADOS DO SHAPEFILE ---
        def safe_load_with_meta(file_name, query):
            path = f"{PASTA_OSM}/{file_name}"
            if not os.path.exists(path):
                print(f"⚠️ AVISO: {file_name} não encontrado. Pulando camada.")
                return None
            
            print(f"\n🔍 ANALISANDO SHAPEFILE: {file_name}")
            meta = con.execute(f"DESCRIBE SELECT * FROM ST_Read('{path}')").pl()
            print(meta.select(["column_name", "column_type"]))
            
            count_raw = con.execute(f"SELECT count(*) FROM ST_Read('{path}')").fetchone()[0]
            print(f"   ↳ Total de feições: {count_raw:,}")
            
            return con.execute(query.replace("PATH", path)).pl()

        # 1. CARREGAR MALHA MESTRA ORIGINAL
        print("\n[LOG] Baixando Malha Mestra do R2...")
        obj = r2.get_object(Bucket=BUCKET, Key="malha geografica/malha_mestra_consolidada_2025.parquet")
        df_mestra = pl.read_parquet(io.BytesIO(obj['Body'].read())).with_columns(pl.col("id_h3_int").cast(pl.UInt64))

        # --- 2. ENRIQUECIMENTO GEOGRÁFICO ---
        # Logradouros (Roads)
        df_roads = safe_load_with_meta("gis_osm_roads_free_1.shp", 
            "SELECT name as osm_rua, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('PATH') WHERE name IS NOT NULL")
        
        if df_roads is not None:
            df_roads_h3 = df_roads.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).unique("id_h3_int").select(["id_h3_int", "osm_rua"])
            df_mestra = df_mestra.join(df_roads_h3, on="id_h3_int", how="left").with_columns(
                pl.col("logradouro").fill_null(pl.col("osm_rua").fill_null("LOGRADOURO NAO IDENTIFICADO"))
            ).drop("osm_rua")

        # Lugares (Places)
        df_places = safe_load_with_meta("gis_osm_places_free_1.shp", 
            "SELECT name as osm_nome, fclass, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('PATH') WHERE name IS NOT NULL")
        
        if df_places is not None:
            df_p_h3 = df_places.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int"))
            osm_cidades = df_p_h3.filter(pl.col("fclass").is_in(['city', 'town', 'village'])).unique("id_h3_int").select(["id_h3_int", "osm_nome"])
            osm_bairros = df_p_h3.filter(pl.col("fclass").is_in(['suburb', 'neighbourhood'])).unique("id_h3_int").select(["id_h3_int", "osm_nome"])
            
            df_mestra = df_mestra.join(osm_cidades.rename({"osm_nome": "res_city"}), on="id_h3_int", how="left").with_columns(
                pl.col("cidade_nome").fill_null(pl.col("res_city").map_elements(remover_acentos, return_dtype=pl.String))
            ).drop("res_city")
            
            df_mestra = df_mestra.join(osm_bairros.rename({"osm_nome": "res_bairro"}), on="id_h3_int", how="left").with_columns(
                pl.when(pl.col("bairro") == "NAO MAPEADO")
                .then(pl.col("res_bairro").map_elements(remover_acentos, return_dtype=pl.String))
                .otherwise(pl.col("bairro")).alias("bairro")
            ).fill_null("NAO MAPEADO").drop("res_bairro")

        # --- 3. INFRAESTRUTURA (EXCLUINDO BUILDINGS) ---
        infra_dfs = []
        
        # POIs (Segurança, Hospitais, Bares)
        df_pois = safe_load_with_meta("gis_osm_pois_free_1.shp", 
            "SELECT fclass, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('PATH') WHERE fclass IN ('police', 'hospital', 'bar', 'bank', 'fuel')")
        
        if df_pois is not None:
            # Correção do pivot: columns -> on
            df_pois_pivot = df_pois.with_columns(
                pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")
            ).pivot(on="fclass", index="id_h3_int", values="fclass", aggregate_function="len").fill_null(0)
            infra_dfs.append(df_pois_pivot)

        # Hidrografia (Água)
        df_water = safe_load_with_meta("gis_osm_water_free_1.shp", 
            "SELECT 1 as presenca_agua, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('PATH')")
        
        if df_water is not None:
            df_w_h3 = df_water.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).unique("id_h3_int").select(["id_h3_int", "presenca_agua"])
            infra_dfs.append(df_w_h3)

        # Unificar Infra
        if infra_dfs:
            df_infra_final = infra_dfs[0]
            for next_df in infra_dfs[1:]:
                df_infra_final = df_infra_final.join(next_df, on="id_h3_int", how="outer").fill_null(0)
        else:
            df_infra_final = pl.DataFrame({"id_h3_int": [0]})

        # --- 4. PERSISTÊNCIA ---
        print("\n[LOG] Persistindo arquivos na Camada Bronze...")
        output_files = [
            (df_mestra, f"{DEST_FOLDER}malha_mestra_bronze.parquet", "MALHA MESTRA (GEOGRAFIA)"),
            (df_infra_final, f"{DEST_FOLDER}malha_geografica_infraestrutura.parquet", "INFRAESTRUTURA")
        ]

        for df, path, label in output_files:
            buf = io.BytesIO()
            df.write_parquet(buf, compression="zstd")
            buf.seek(0)
            r2.upload_fileobj(buf, BUCKET, path)
            
            # Reporte de Metadados Finais
            print(f"\n📁 Dataset: {label}")
            print(f"   ↳ Registros: {df.shape[0]:,}")
            print(f"   ↳ Schema: {dict(df.schema)}")
            print(f"   ↳ Nulos Totais: {df.null_count().sum_horizontal()[0]}")

        # Limpeza da raiz
        try: r2.delete_object(Bucket=BUCKET, Key="malha geografica/malha_mestra_consolidada_2025.parquet")
        except: pass

        print("\n" + "✅ [SUCESSO] Camada Bronze consolidada e organizada!")

    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    pipeline_bronze_master()
