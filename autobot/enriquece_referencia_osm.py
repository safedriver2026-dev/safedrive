import os, duckdb, polars as pl, boto3, io, requests, zipfile, sys, traceback, shutil
import h3
from botocore.config import Config

def get_h3_int(lat, lon):
    try:
        return int(h3.latlng_to_cell(lat, lon, 9), 16)
    except:
        return 0

def pipeline_referencia_osm():
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    CAMINHO_MALHA = "malha geografica/malha_mestra_consolidada_2025.parquet"
    
    URL_OSM_SP = "https://download.geofabrik.de/south-america/brazil/sao-paulo-latest-free.shp.zip"
    PASTA_TEMP_OSM = "temp_osm_referencia"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        print("[LOG] Iniciando Enriquecimento de Referencia com OSM...")

        # 1. DOWNLOAD E EXTRACAO
        print("[LOG] Baixando base completa do OSM para SP...")
        os.makedirs(PASTA_TEMP_OSM, exist_ok=True)
        r = requests.get(URL_OSM_SP, stream=True)
        r.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            # Extraímos rios/lagos (water) e estradas (roads)
            alvos = ['gis_osm_water_a_free_1', 'gis_osm_roads_free_1']
            for f in z.namelist():
                if any(x in f for x in alvos):
                    z.extract(f, PASTA_TEMP_OSM)

        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # 2. CARREGAR MALHA MESTRA
        print("[LOG] Lendo Malha Mestra...")
        obj = r2.get_object(Bucket=BUCKET, Key=CAMINHO_MALHA)
        df_malha = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        # ==========================================
        # PARTE A: HIDROGRAFIA (CORPOS D'AGUA)
        # ==========================================
        print("[LOG] Processando Hidrografia (Rios, Represas, Lagos)...")
        # O _a_ no nome significa area (poligono). Pegamos o centroide (ST_Centroid) do poligono.
        df_agua = con.execute(f"""
            SELECT ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon
            FROM ST_Read('{PASTA_TEMP_OSM}/gis_osm_water_a_free_1.shp')
            WHERE fclass IN ('river', 'water', 'reservoir')
        """).pl()

        df_agua = df_agua.with_columns(
            pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")
        ).unique(subset=["id_h3_int"])

        # Criamos a coluna 'tem_agua'
        df_agua = df_agua.with_columns(pl.lit(1, dtype=pl.UInt8).alias("tem_agua")).drop(["lat", "lon"])
        df_malha = df_malha.join(df_agua, on="id_h3_int", how="left").with_columns(pl.col("tem_agua").fill_null(0))

        # ==========================================
        # PARTE B: MALHA VIARIA E TIPOLOGIA DE AREA
        # ==========================================
        print("[LOG] Processando Malha Viaria (Classificacao de Ruas)...")
        # As vias sao linhas. O DuckDB permite pegar um ponto dentro dessa linha.
        df_ruas = con.execute(f"""
            SELECT fclass, ST_Y(ST_PointN(geom, 1)) as lat, ST_X(ST_PointN(geom, 1)) as lon
            FROM ST_Read('{PASTA_TEMP_OSM}/gis_osm_roads_free_1.shp')
        """).pl()

        df_ruas = df_ruas.with_columns(
            pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")
        )

        # Classificacao da via principal de cada hexagono
        print("[LOG] Determinando o tipo de zona via OSM...")
        # 1. Contamos o total de vias (segmentos) para medir densidade
        df_densidade = df_ruas.group_by("id_h3_int").len().rename({"len": "qtd_segmentos_via"})
        
        # 2. Descobrimos a via "mais importante" do hexagono (prioridade rodovia > primaria > secundaria > residencial)
        ordem_importancia = {
            'motorway': 1, 'motorway_link': 1, 'trunk': 2, 'primary': 3,
            'secondary': 4, 'tertiary': 5, 'residential': 6, 'unclassified': 7
        }
        
        df_ruas_imp = df_ruas.with_columns(
            pl.col("fclass").replace(ordem_importancia, default=99).cast(pl.UInt8).alias("peso_via")
        ).group_by("id_h3_int").min()
        
        df_ruas_imp = df_ruas_imp.with_columns(
            pl.when(pl.col("peso_via") <= 2).then(pl.lit("RODOVIA"))
            .when(pl.col("peso_via") <= 5).then(pl.lit("ARTERIA_PRINCIPAL"))
            .when(pl.col("peso_via") == 6).then(pl.lit("RESIDENCIAL"))
            .otherwise(pl.lit("ACESSO_LOCAL")).alias("tipo_zona_viaria")
        ).drop(["peso_via", "fclass", "lat", "lon"])

        # Juntando as informacoes de vias na malha mestra
        df_malha = df_malha.join(df_densidade, on="id_h3_int", how="left").with_columns(pl.col("qtd_segmentos_via").fill_null(0).cast(pl.UInt32))
        df_malha = df_malha.join(df_ruas_imp, on="id_h3_int", how="left").with_columns(pl.col("tipo_zona_viaria").fill_null("ISOLADO"))

        # 3. SALVAR RESULTADOS E LIMPAR
        print("[LOG] Guardando a Malha Mestra (Referencia) atualizada...")
        buffer = io.BytesIO()
        df_malha.write_parquet(buffer, compression="zstd")
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, CAMINHO_MALHA)

        print(f"[LOG] Removendo pasta temporaria '{PASTA_TEMP_OSM}'...")
        if os.path.exists(PASTA_TEMP_OSM):
            shutil.rmtree(PASTA_TEMP_OSM)
            
        print("[LOG] Enriquecimento de Referencia concluido. A malha agora entende topologia.")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    pipeline_referencia_osm()
