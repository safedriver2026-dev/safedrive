import os, h3, requests, polars as pl, duckdb, boto3, io, json
from botocore.config import Config

def criar_dim_bairro():
    R2_URL = os.getenv("R2_ENDPOINT_URL")
    R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
    BUCKET = os.getenv("R2_BUCKET_NAME")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    try:
        print("🏘️ Obtendo malha de distritos...")
        url_distritos = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?intrarregiao=distrito&formato=application/vnd.geo+json&qualidade=minima"
        try:
            res = requests.get(url_distritos, timeout=30).json()
            with open("distritos.json", "w") as f: json.dump(res, f)
        except:
            print("⚠️ Usando Mirror para distritos...")
            url_mirror = "https://raw.githubusercontent.com/giuliano-macedo/geodata-br-geojson/master/sp/sp_distritos.json"
            res = requests.get(url_mirror).json()
            with open("distritos.json", "w") as f: json.dump(res, f)

        r2 = boto3.client("s3", endpoint_url=R2_URL, aws_access_key_id=R2_KEY.strip(), 
                          aws_secret_access_key=R2_SECRET.strip(), config=Config(region_name="auto"))
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        dim_bairro = con.execute("""
            SELECT h.id_h3_h9, COALESCE(d.nm_dis, d.nm_distrito, 'Desconhecido') as nome_bairro
            FROM df_hex h
            JOIN st_read('distritos.json') d ON ST_Within(ST_Point(h.lon, h.lat), d.geom)
        """).pl()

        buffer = io.BytesIO()
        dim_bairro.write_parquet(buffer)
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, "referencia/dim_bairro.parquet")
        print(f"✅ Dimensão Bairro concluída: {len(dim_bairro)} registos.")
    except Exception as e:
        print(f"🚨 Erro: {e}")
        raise e

if __name__ == "__main__":
    criar_dim_bairro()
