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

def pipeline_osm_datalake():
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    
    # DEFINICAO DOS CAMINHOS COM A NOVA NOMENCLATURA
    CAMINHO_MALHA_BASE = "malha geografica/malha_mestra_consolidada_2025.parquet"
    CAMINHO_BRONZE_INFRA = "datalake/bronze/malha_geografica_infraestrutura.parquet"
    
    URL_OSM_SP = "https://download.geofabrik.de/south-america/brazil/sao-paulo-latest-free.shp.zip"
    PASTA_TEMP = "temp_osm_dados"
    ARQUIVO_ZIP = "dados_osm.zip"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        print("[LOG] Iniciando Processamento para Camada Bronze (Infraestrutura)...")

        # 1. DOWNLOAD ROBUSTO (STREAMING PARA DISCO - ISSO EVITA O ERRO BADZIPFILE)
        headers = {'User-Agent': 'Mozilla/5.0'}
        with requests.get(URL_OSM_SP, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(ARQUIVO_ZIP, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # 2. EXTRACAO
        os.makedirs(PASTA_TEMP, exist_ok=True)
        with zipfile.ZipFile(ARQUIVO_ZIP, 'r') as z:
            alvos = ['gis_osm_places_free_1', 'gis_osm_transport_free_1', 'gis_osm_pois_free_1']
            for f in z.namelist():
                if any(x in f for x in alvos):
                    z.extract(f, PASTA_TEMP)
        
        os.remove(ARQUIVO_ZIP)
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # 3. ATUALIZACAO DA MALHA GEOGRAFICA (BASE)
        obj = r2.get_object(Bucket=BUCKET, Key=CAMINHO_MALHA_BASE)
        df_base = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        df_places = con.execute(f"SELECT name, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('{PASTA_TEMP}/gis_osm_places_free_1.shp')").pl()
        df_places = df_places.with_columns(
            pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")
        ).select([
            pl.col("id_h3_int"), 
            pl.col("name").map_elements(remover_acentos, return_dtype=pl.String).alias("osm_bairro")
        ]).unique("id_h3_int")

        df_base = df_base.join(df_places, on="id_h3_int", how="left")
        df_base = df_base.with_columns(
            pl.when(pl.col("bairro") == "NAO MAPEADO").then(pl.col("osm_bairro").fill_null("NAO MAPEADO"))
            .otherwise(pl.col("bairro")).alias("bairro")
        ).drop("osm_bairro")

        # 4. CRIACAO DA MALHA DE INFRAESTRUTURA
        print("[LOG] Gerando arquivo de Infraestrutura Urbana...")
        df_inf = con.execute(f"""
            SELECT fclass, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('{PASTA_TEMP}/gis_osm_transport_free_1.shp') WHERE fclass IN ('bus_stop', 'railway_station')
            UNION ALL
            SELECT fclass, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('{PASTA_TEMP}/gis_osm_pois_free_1.shp') WHERE fclass IN ('police', 'bank', 'bar', 'hospital', 'fuel')
        """).pl()
        
        df_infra = df_inf.with_columns(
            pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")
        ).pivot(values="fclass", index="id_h3_int", columns="fclass", aggregate_function="len").fill_null(0)

        # 5. UPLOAD
        # Base atualizada (Bairros corrigidos)
        buf_base = io.BytesIO()
        df_base.write_parquet(buf_base, compression="zstd")
        buf_base.seek(0)
        r2.upload_fileobj(buf_base, BUCKET, CAMINHO_MALHA_BASE)

        # Infraestrutura (Novas features)
        print(f"[LOG] Salvando Camada de Infraestrutura em '{CAMINHO_BRONZE_INFRA}'...")
        buf_inf = io.BytesIO()
        df_infra.write_parquet(buf_inf, compression="zstd")
        buf_inf.seek(0)
        r2.upload_fileobj(buf_inf, BUCKET, CAMINHO_BRONZE_INFRA)

        # 6. LIMPEZA
        shutil.rmtree(PASTA_TEMP)
        print("[LOG] Processo concluido com sucesso.")

    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    pipeline_osm_datalake()
