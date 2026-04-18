import os
import h3
import requests
import polars as pl
import boto3
from botocore.config import Config
import io

def gerar_malha_pura():
    R2_URL = os.getenv("R2_ENDPOINT_URL")
    R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
    BUCKET = os.getenv("R2_BUCKET_NAME")

    try:
        print("⬢ Gerando IDs H3 puros para o Estado de SP...")
        # Pegamos apenas o contorno do estado para o preenchimento
        url_sp = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?formato=application/vnd.geo+json&qualidade=minima"
        res = requests.get(url_sp).json()
        coords = res['features'][0]['geometry']
        
        hexs = set()
        geometrias = coords['coordinates'] if coords['type'] == 'MultiPolygon' else [coords['coordinates']]
        for p in geometrias:
            hexs.update(h3.polyfill({'type': 'Polygon', 'coordinates': [[[c[1], c[0]] for c in p[0]]]}, 9))

        df_hex = pl.DataFrame({"id_h3_h9": list(hexs)}).with_columns([
            pl.col("id_h3_h9").map_elements(lambda x: h3.h3_to_parent(x, 8), return_dtype=pl.Utf8).alias("id_h3_h8"),
            pl.col("id_h3_h9").map_elements(lambda x: h3.h3_to_geo(x)[0], return_dtype=pl.Float64).alias("lat"),
            pl.col("id_h3_h9").map_elements(lambda x: h3.h3_to_geo(x)[1], return_dtype=pl.Float64).alias("lon")
        ])

        buffer = io.BytesIO()
        df_hex.write_parquet(buffer)
        buffer.seek(0)
        
        r2 = boto3.client("s3", endpoint_url=R2_URL, aws_access_key_id=R2_KEY.strip(), 
                          aws_secret_access_key=R2_SECRET.strip(), config=Config(region_name="auto"))
        r2.upload_fileobj(buffer, BUCKET, "referencia/dim_hex_sp.parquet")
        print(f"✅ Malha Base Gerada: {len(df_hex)} hexágonos.")

    except Exception as e:
        print(f"🚨 Erro: {e}")
        raise e

if __name__ == "__main__":
    gerar_malha_pura()
