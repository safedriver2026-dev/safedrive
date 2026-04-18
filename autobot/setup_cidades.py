import os, h3, requests, polars as pl, duckdb, boto3, io
from botocore.config import Config

def criar_dim_cidade():
    # Credenciais
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
        print("🏙️ Ingerindo cidades (IBGE)...")
        url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?intrarregiao=municipio&formato=application/vnd.geo+json&qualidade=minima"
        res = requests.get(url).json()

        # Extraímos os dados via Python para não depender do "humor" do DuckDB com o JSON
        features = []
        for f in res['features']:
            props = f['properties']
            features.append({
                "codigo_ibge": props.get('cd_mun') or props.get('codarea'),
                "nome_cidade": props.get('nm_mun') or props.get('nomemunicipio') or "Desconhecido",
                "wkt": duckdb.execute("SELECT ST_AsText(ST_ReadGeoJSON(?))", [str(f['geometry'])]).fetchone()[0]
            })
        
        df_geo = pl.DataFrame(features)

        print("☁️ Carregando Malha Base H3...")
        r2 = boto3.client("s3", **R2_CONFIG)
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        print("⚔️ Cruzamento Espacial...")
        dim_cidade = con.execute("""
            SELECT h.id_h3_h9, g.nome_cidade, g.codigo_ibge
            FROM df_hex h
            JOIN df_geo g ON ST_Within(ST_Point(h.lon, h.lat), ST_GeomFromText(g.wkt))
        """).pl()

        # Upload
        buffer = io.BytesIO()
        dim_cidade.write_parquet(buffer)
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, "referencia/dim_cidade.parquet")
        print(f"✅ Dimensão Cidade: {len(dim_cidade)} registros.")

    except Exception as e:
        print(f"🚨 Erro em Cidades: {e}")
        raise e

if __name__ == "__main__":
    criar_dim_cidade()
