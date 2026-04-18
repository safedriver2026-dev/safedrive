import os, h3, requests, polars as pl, duckdb, boto3, io
from botocore.config import Config

def criar_dim_cidade():
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    
    # Busca a malha municipal do IBGE (mais estável que a de distritos)
    url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?intrarregiao=municipio&formato=application/vnd.geo+json&qualidade=minima"
    res = requests.get(url).json()
    with open("municipios.json", "w") as f: f.write(json.dumps(res))

    # Carrega os 2.3M de hexágonos que você já gerou
    r2 = boto3.client("s3", endpoint_url=os.getenv("R2_ENDPOINT_URL"), 
                      aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID").strip(),
                      aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY").strip(),
                      config=Config(region_name="auto"))
    obj = r2.get_object(Bucket=os.getenv("R2_BUCKET_NAME"), Key="referencia/dim_hex_sp.parquet")
    df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

    dim_cidade = con.execute("""
        SELECT h.id_h3_h9, d.nm_mun as nome_cidade, d.cd_mun as codigo_ibge_cidade
        FROM df_hex h
        JOIN st_read('municipios.json') d ON ST_Within(ST_Point(h.lon, h.lat), d.geom)
    """).pl()

    # Salva no R2
    buffer = io.BytesIO()
    dim_cidade.write_parquet(buffer)
    buffer.seek(0)
    r2.upload_fileobj(buffer, os.getenv("R2_BUCKET_NAME"), "referencia/dim_cidade.parquet")
    print(f"✅ Dimensão Cidade: {len(dim_cidade)} registros.")
