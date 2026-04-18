import os, h3, polars as pl, duckdb, boto3, io
from botocore.config import Config

def criar_dim_logradouro():
    R2_CONFIG = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    try:
        print("☁️ Carregando Malha Base H3...")
        r2 = boto3.client("s3", **R2_CONFIG)
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        print("🛣️ Extraindo logradouros do OSM (PBF)...")
        # O DuckDB lê PBF nativamente com ST_Read
        df_ruas = con.execute("""
            SELECT 
                tags['name'] as nome_rua, 
                tags['highway'] as tipo_via,
                ST_Y(ST_Centroid(geom)) as lat, 
                ST_X(ST_Centroid(geom)) as lon
            FROM ST_Read('sao-paulo-latest.osm.pbf')
            WHERE tags['name'] IS NOT NULL AND tags['highway'] IS NOT NULL
        """).pl()

        print("⬢ Mapeando H3 para cada rua...")
        dim_logradouros = df_ruas.with_columns(
            pl.struct(["lat", "lon"]).map_elements(
                lambda x: h3.geo_to_h3(x["lat"], x["lon"], 9), return_dtype=pl.Utf8
            ).alias("id_h3_h9")
        ).select(["id_h3_h9", "nome_rua", "tipo_via"]).unique()

        # Filtragem interna
        dim_logradouros = dim_logradouros.join(df_hex.select("id_h3_h9"), on="id_h3_h9", how="inner")

        buffer = io.BytesIO()
        dim_logradouros.write_parquet(buffer)
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, "referencia/dim_logradouro.parquet")
        print("✅ Dimensão Logradouro concluída.")

    except Exception as e:
        print(f"🚨 Erro em Logradouros: {e}")
        raise e

if __name__ == "__main__":
    criar_dim_logradouro()
