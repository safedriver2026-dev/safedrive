import os, h3, polars as pl, duckdb, boto3, io
from botocore.config import Config

def criar_dim_logradouro():
    R2_URL = os.getenv("R2_ENDPOINT_URL")
    R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
    BUCKET = os.getenv("R2_BUCKET_NAME")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    try:
        print("☁️ Carregando Malha Base...")
        r2 = boto3.client("s3", endpoint_url=R2_URL, aws_access_key_id=R2_KEY.strip(), 
                          aws_secret_access_key=R2_SECRET.strip(), config=Config(region_name="auto"))
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        print("🛣️ Extraindo ruas do OSM...")
        df_ruas = con.execute("""
            SELECT tags['name'] as nome_rua, tags['highway'] as tipo_via,
                   ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon
            FROM st_read('sao-paulo-latest.osm.pbf')
            WHERE tags['name'] IS NOT NULL AND tags['highway'] IS NOT NULL
        """).pl()

        dim_logradouros = df_ruas.with_columns(
            pl.struct(["lat", "lon"]).map_elements(
                lambda x: h3.geo_to_h3(x["lat"], x["lon"], 9), return_dtype=pl.Utf8
            ).alias("id_h3_h9")
        ).select(["id_h3_h9", "nome_rua", "tipo_via"]).unique()

        # Garante que apenas ruas dentro da malha de SP sejam mantidas
        dim_logradouros = dim_logradouros.join(df_hex.select("id_h3_h9"), on="id_h3_h9", how="inner")

        buffer = io.BytesIO()
        dim_logradouros.write_parquet(buffer)
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, "referencia/dim_logradouro.parquet")
        print(f"✅ Dimensão Logradouro concluída: {len(dim_logradouros)} registos.")
    except Exception as e:
        print(f"🚨 Erro: {e}")
        raise e

if __name__ == "__main__":
    criar_dim_logradouro()
