import os, h3, requests, polars as pl, duckdb, boto3, io, json
from botocore.config import Config

def criar_dim_cidade():
    R2_URL = os.getenv("R2_ENDPOINT_URL")
    R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
    BUCKET = os.getenv("R2_BUCKET_NAME")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    try:
        print("🏙️ Obtendo malha municipal do IBGE...")
        url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?intrarregiao=municipio&formato=application/vnd.geo+json&qualidade=minima"
        res = requests.get(url).json()
        with open("municipios.json", "w") as f: json.dump(res, f)

        print("☁️ Carregando Malha Base H3...")
        r2 = boto3.client("s3", endpoint_url=R2_URL, aws_access_key_id=R2_KEY.strip(), 
                          aws_secret_access_key=R2_SECRET.strip(), config=Config(region_name="auto"))
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        dim_cidade = con.execute("""
            SELECT h.id_h3_h9, d.nm_mun as nome_cidade, d.cd_mun as codigo_ibge_cidade
            FROM df_hex h
            JOIN st_read('municipios.json') d ON ST_Within(ST_Point(h.lon, h.lat), d.geom)
        """).pl()

        buffer = io.BytesIO()
        dim_cidade.write_parquet(buffer)
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, "referencia/dim_cidade.parquet")
        print(f"✅ Dimensão Cidade concluída: {len(dim_cidade)} registos.")
    except Exception as e:
        print(f"🚨 Erro: {e}")
        raise e

if __name__ == "__main__":
    criar_dim_cidade()
