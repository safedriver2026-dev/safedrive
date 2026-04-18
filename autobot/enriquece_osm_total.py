import os, duckdb, polars as pl, boto3, io, unicodedata, sys, traceback, shutil
import h3
from botocore.config import Config

def remover_acentos(texto):
    if texto is None or str(texto).upper() in ["NAN", "", "NULL"]: return "NAO MAPEADO"
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def get_h3_int(lat, lon):
    try: return int(h3.latlng_to_cell(lat, lon, 9), 16)
    except: return 0

def pipeline_processamento_local():
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    
    CAMINHO_MALHA_BASE = "malha geografica/malha_mestra_consolidada_2025.parquet"
    CAMINHO_BRONZE_INFRA = "datalake/bronze/malha_geografica_infraestrutura.parquet"
    PASTA_DADOS = "temp_estadual"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        print("🚀 [LOG] Iniciando Processamento de Dados Locais...")

        # 1. DUCKDB: LEITURA DIRETA DO DISCO
        print("[LOG] DuckDB lendo Shapefiles extraídos...")
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        
        # Filtramos POIs e Transportes
        query_infra = f"""
            SELECT fclass, ST_Y(geom) as lat, ST_X(geom) as lon 
            FROM ST_Read('{PASTA_DADOS}/gis_osm_pois_free_1.shp')
            WHERE fclass IN ('police', 'bank', 'bar', 'hospital', 'fuel')
            UNION ALL
            SELECT fclass, ST_Y(geom) as lat, ST_X(geom) as lon 
            FROM ST_Read('{PASTA_DADOS}/gis_osm_transport_free_1.shp')
            WHERE fclass IN ('bus_stop', 'railway_station')
        """
        df_infra_raw = con.execute(query_infra).pl()

        # 2. POLARS: MAPEAMENTO H3
        print(f"[LOG] Mapeando {len(df_infra_raw)} pontos para a grade H3...")
        df_infra = df_infra_raw.with_columns(
            pl.struct(["lat", "lon"])
            .map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64)
            .alias("id_h3_int")
        ).drop(["lat", "lon"])

        df_pivot = df_infra.pivot(
            values="fclass", index="id_h3_int", columns="fclass", aggregate_function="len"
        ).fill_null(0)

        # 3. ATUALIZAÇÃO DA MALHA BASE (FALLBACK DE BAIRROS)
        print("[LOG] Buscando Malha Mestra no R2 para atualizar nomes...")
        obj = r2.get_object(Bucket=BUCKET, Key=CAMINHO_MALHA_BASE)
        df_base = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        df_places = con.execute(f"""
            SELECT name, ST_Y(geom) as lat, ST_X(geom) as lon 
            FROM ST_Read('{PASTA_DADOS}/gis_osm_places_free_1.shp')
            WHERE fclass IN ('suburb', 'neighbourhood', 'town', 'village') AND name IS NOT NULL
        """).pl()

        df_places = df_places.with_columns(
            pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int"),
            pl.col("name").map_elements(remover_acentos, return_dtype=pl.String).alias("osm_bairro")
        ).unique("id_h3_int").select(["id_h3_int", "osm_bairro"])

        df_base = df_base.join(df_places, on="id_h3_int", how="left")
        df_base = df_base.with_columns(
            pl.when(pl.col("bairro") == "NAO MAPEADO").then(pl.col("osm_bairro").fill_null("NAO MAPEADO"))
            .otherwise(pl.col("bairro")).alias("bairro")
        ).drop("osm_bairro")

        # 4. UPLOAD DOS RESULTADOS
        print("[LOG] Enviando arquivos consolidados para o Cloudflare R2...")
        
        # Infraestrutura
        buf_inf = io.BytesIO()
        df_pivot.write_parquet(buf_inf, compression="zstd")
        buf_inf.seek(0)
        r2.upload_fileobj(buf_inf, BUCKET, CAMINHO_BRONZE_INFRA)

        # Malha Base
        buf_base = io.BytesIO()
        df_base.write_parquet(buf_base, compression="zstd")
        buf_base.seek(0)
        r2.upload_fileobj(buf_base, BUCKET, CAMINHO_MALHA_BASE)

        print(f"✅ [LOG] Sucesso! Processamento estadual concluído.")

    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    pipeline_processamento_local()
