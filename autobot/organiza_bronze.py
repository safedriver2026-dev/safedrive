import os, duckdb, polars as pl, boto3, io, unicodedata, sys, traceback
import h3
from botocore.config import Config

# --- UTILITÁRIOS ---
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
    
    # Estrutura de Pastas conforme solicitado
    DEST_FOLDER = "datalake/bronze/malha_geografica/"
    FILE_MESTRA = f"{DEST_FOLDER}malha_mestra_bronze.parquet"
    FILE_INFRA = f"{DEST_FOLDER}malha_geografica_infraestrutura.parquet"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # --- 1. CARREGAR MALHA BASE ---
        print("[LOG] Baixando Malha Mestra do R2...")
        obj = r2.get_object(Bucket=BUCKET, Key="malha geografica/malha_mestra_consolidada_2025.parquet")
        df_mestra = pl.read_parquet(io.BytesIO(obj['Body'].read())).with_columns(pl.col("id_h3_int").cast(pl.UInt64))

        # --- 2. FUNÇÃO DE LEITURA SEGURA (RESOLVE O ERRO DE IO) ---
        def safe_load_shp(file_name, query):
            path = f"{PASTA_OSM}/{file_name}"
            if os.path.exists(path):
                print(f"✅ Processando: {file_name}")
                return con.execute(query.replace("FILE_PATH", path)).pl()
            else:
                print(f"⚠️ AVISO: {file_name} não encontrado. Ignorando esta camada.")
                return None

        # --- 3. ENRIQUECIMENTO GEOGRÁFICO (MALHA MESTRA) ---
        print("\n[LOG] Refinando Geografia e Endereçamento...")
        
        # Ruas (Logradouros)
        df_roads = safe_load_shp("gis_osm_roads_free_1.shp", 
            "SELECT name as osm_rua, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('FILE_PATH') WHERE name IS NOT NULL")
        
        if df_roads is not None:
            df_roads_h3 = df_roads.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).unique("id_h3_int").select(["id_h3_int", "osm_rua"])
            df_mestra = df_mestra.join(df_roads_h3, on="id_h3_int", how="left").with_columns(
                pl.col("logradouro").fill_null(pl.col("osm_rua").fill_null("LOGRADOURO NAO IDENTIFICADO"))
            ).drop("osm_rua")

        # Bairros (Places)
        df_places = safe_load_shp("gis_osm_places_free_1.shp", 
            "SELECT name as osm_nome, fclass, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('FILE_PATH') WHERE name IS NOT NULL AND fclass IN ('suburb', 'neighbourhood')")
        
        if df_places is not None:
            osm_bairros = df_places.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).unique("id_h3_int").select(["id_h3_int", "osm_nome"])
            df_mestra = df_mestra.join(osm_bairros.rename({"osm_nome": "resgate_bairro"}), on="id_h3_int", how="left").with_columns(
                pl.col("bairro").map_elements(lambda x: x if x != "NAO MAPEADO" else None, return_dtype=pl.String)
            ).with_columns(
                pl.col("bairro").fill_null(pl.col("resgate_bairro").map_elements(remover_acentos, return_dtype=pl.String))
            ).fill_null("NAO MAPEADO").drop("resgate_bairro")

        # --- 4. INFRAESTRUTURA (ARQUIVO SEPARADO) ---
        print("\n[LOG] Extraindo Indicadores de Infraestrutura...")
        
        # Prédios
        df_builds = safe_load_shp("gis_osm_buildings_free_1.shp", "SELECT count(*) as qtd, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('FILE_PATH') GROUP BY geom")
        
        # POIs
        df_pois = safe_load_shp("gis_osm_pois_free_1.shp", "SELECT fclass, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('FILE_PATH') WHERE fclass IN ('police', 'hospital', 'bar', 'bank', 'fuel')")
        
        # Água
        df_water = safe_load_shp("gis_osm_water_free_1.shp", "SELECT 1 as presenca_agua, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('FILE_PATH')")

        # Consolidar Infraestrutura
        infra_dfs = []
        if df_builds is not None:
            infra_dfs.append(df_builds.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).group_by("id_h3_int").agg(pl.col("qtd").sum().alias("qtd_predios")))
        if df_pois is not None:
            infra_dfs.append(df_pois.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).pivot(values="fclass", index="id_h3_int", columns="fclass", aggregate_function="len").fill_null(0))
        if df_water is not None:
            infra_dfs.append(df_water.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).unique("id_h3_int").select(["id_h3_int", "presenca_agua"]))

        # Join final da infra
        df_infra_final = infra_dfs[0] if infra_dfs else pl.DataFrame({"id_h3_int": [0]})
        for extra_df in infra_dfs[1:]:
            df_infra_final = df_infra_final.join(extra_df, on="id_h3_int", how="outer").fill_null(0)

        # --- 5. PERSISTÊNCIA E LIMPEZA ---
        print("\n[LOG] Salvando arquivos na Camada Bronze...")
        for df, path in [(df_mestra, FILE_MESTRA), (df_infra_final, FILE_INFRA)]:
            buf = io.BytesIO()
            df.write_parquet(buf, compression="zstd")
            buf.seek(0)
            r2.upload_fileobj(buf, BUCKET, path)
            print(f"✅ Upload concluído: {path}")

        # Limpeza da raiz
        r2.delete_object(Bucket=BUCKET, Key="malha geografica/malha_mestra_consolidada_2025.parquet")

        # --- 6. METADADOS E AMOSTRAS ---
        print("\n" + "═"*70 + "\n📑 RELATÓRIO DE METADADOS (BRONZE)\n" + "═"*70)
        
        for name, df in [("MALHA MESTRA", df_mestra), ("INFRAESTRUTURA", df_infra_final)]:
            print(f"\n📁 Dataset: {name}")
            print(f"   ↳ Dimensões: {df.shape}")
            print(f"   ↳ Colunas: {df.columns}")
            print(f"   ↳ Nulos totais: {df.null_count().sum_horizontal()[0]}")
            print(f"   ↳ Tipos: {dict(df.schema)}")
        
        print("\n" + "📊 AMOSTRA DA MALHA MESTRA")
        with pl.Config(tbl_rows=5, tbl_width_chars=120):
            print(df_mestra.select(["id_h3_int", "logradouro", "bairro", "cidade_nome"]).head(5))

        print("\n" + "✅ [SUCESSO] Camada Bronze organizada!")

    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    pipeline_bronze_master()
