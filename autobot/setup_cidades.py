import os, h3, requests, polars as pl, duckdb, boto3, io
from botocore.config import Config

def criar_dim_cidade():
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
        print("🏙️ Baixando malha de cidades do IBGE...")
        url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?intrarregiao=municipio&formato=application/vnd.geo+json&qualidade=minima"
        res = requests.get(url)
        with open("municipios.json", "wb") as f: f.write(res.content)

        print("☁️ Carregando Malha Base H3 do R2...")
        r2 = boto3.client("s3", **R2_CONFIG)
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        print("⚔️ Cruzamento Espacial de Alta Performance...")
        # O DuckDB ST_Read já achata o GeoJSON. Usamos COALESCE para os nomes.
        dim_cidade = con.execute("""
            SELECT 
                h.id_h3_h9, 
                COALESCE(g.nm_mun, g.nm_municipio, g.nomemunicipio) as nome_cidade,
                COALESCE(g.cd_mun, g.cd_municipio, g.codarea) as codigo_ibge_cidade
            FROM df_hex h
            JOIN ST_Read('municipios.json') g 
            ON ST_Within(ST_Point(h.lon, h.lat), g.geom)
        """).pl()

        buffer = io.BytesIO()
        dim_cidade.write_parquet(buffer)
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, "referencia/dim_cidade.parquet")
        print(f"✅ Dimensão Cidade concluída: {len(dim_cidade)} registros.")

    except Exception as e:
        print(f"🚨 Erro em Cidades: {e}")
        raise e

if __name__ == "__main__":
    criar_dim_cidade()
