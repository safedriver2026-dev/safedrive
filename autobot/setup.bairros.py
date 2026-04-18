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
        print("🏘️ Ingerindo bairros/distritos...")
        url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?intrarregiao=distrito&formato=application/vnd.geo+json&qualidade=minima"
        
        try:
            res = requests.get(url, timeout=30).json()
        except:
            url_mirror = "https://raw.githubusercontent.com/giuliano-macedo/geodata-br-geojson/master/sp/sp_distritos.json"
            res = requests.get(url_mirror).json()

        features = []
        for f in res['features']:
            props = f['properties']
            features.append({
                "nome_bairro": props.get('nm_dis') or props.get('nm_distrito') or props.get('nomedistrito') or "Desconhecido",
                "wkt": con.execute("SELECT ST_AsText(ST_ReadGeoJSON(?))", [str(f['geometry'])]).fetchone()[0]
            })
        df_geo = pl.DataFrame(features)

        r2 = boto3.client("s3", **R2_CONFIG)
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        dim_bairro = con.execute("""
            SELECT h.id_h3_h9, g.nome_bairro
            FROM df_hex h
            JOIN df_geo g ON ST_Within(ST_Point(h.lon, h.lat), ST_GeomFromText(g.wkt))
        """).pl()

        buffer = io.BytesIO()
        dim_bairro.write_parquet(buffer)
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, "referencia/dim_bairro.parquet")
        print(f"✅ Dimensão Bairro: {len(dim_bairro)} registros.")

    except Exception as e:
        print(f"🚨 Erro em Bairros: {e}")
        raise e

if __name__ == "__main__":
    criar_dim_bairro()
