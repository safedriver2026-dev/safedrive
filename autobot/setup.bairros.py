import os, h3, requests, polars as pl, duckdb, boto3, io
from botocore.config import Config

def criar_dim_bairro():
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
        print("🏘️ Obtendo malha de distritos (Bairros)...")
        url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?intrarregiao=distrito&formato=application/vnd.geo+json&qualidade=minima"
        try:
            res = requests.get(url, timeout=30)
            res.raise_for_status()
            with open("distritos.json", "wb") as f: f.write(res.content)
        except:
            print("⚠️ Usando Mirror alternativo...")
            url_mirror = "https://raw.githubusercontent.com/giuliano-macedo/geodata-br-geojson/master/sp/sp_distritos.json"
            res = requests.get(url_mirror)
            with open("distritos.json", "wb") as f: f.write(res.content)

        r2 = boto3.client("s3", **R2_CONFIG)
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        print("⚔️ Cruzamento Espacial (Bairros)...")
        dim_bairro = con.execute("""
            SELECT 
                h.id_h3_h9, 
                COALESCE(g.nm_dis, g.nm_distrito, g.nomedistrito, g.nm_mun) as nome_bairro
            FROM df_hex h
            JOIN ST_Read('distritos.json') g 
            ON ST_Within(ST_Point(h.lon, h.lat), g.geom)
        """).pl()

        buffer = io.BytesIO()
        dim_bairro.write_parquet(buffer)
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, "referencia/dim_bairro.parquet")
        print(f"✅ Dimensão Bairro concluída: {len(dim_bairro)} registros.")

    except Exception as e:
        print(f"🚨 Erro em Bairros: {e}")
        raise e

if __name__ == "__main__":
    criar_dim_bairro()
