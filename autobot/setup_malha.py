import os
import h3
import requests
import polars as pl
import duckdb
import boto3
from botocore.config import Config
import io

def criar_malha_mestra():
    # Credenciais limpas do ambiente
    R2_URL = os.getenv("R2_ENDPOINT_URL")
    R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
    BUCKET = os.getenv("R2_BUCKET_NAME")

    # DuckDB para join espacial eficiente
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        # 1. Gerar Geometria H3 (Nível 9 - Quarteirão)
        print("⬢ Gerando malha H3 (Resolução 9) para São Paulo...")
        url_sp = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?formato=application/vnd.geo+json"
        res_sp = requests.get(url_sp, headers=headers)
        res_sp.raise_for_status()
        
        coords = res_sp.json()['features'][0]['geometry']
        hexs = set()
        geos = coords['coordinates'] if coords['type'] == 'MultiPolygon' else [coords['coordinates']]
        for p in geos:
            poly = {'type': 'Polygon', 'coordinates': [[[c[1], c[0]] for c in p[0]]]}
            hexs.update(h3.polyfill(poly, 9))

        df_h3 = pl.DataFrame({"id_h3_h9": list(hexs)}).with_columns([
            pl.col("id_h3_h9").map_elements(lambda x: h3.h3_to_parent(x, 8), return_dtype=pl.Utf8).alias("id_h3_h8"),
            pl.col("id_h3_h9").map_elements(lambda x: h3.h3_to_geo(x)[0], return_dtype=pl.Float64).alias("lat"),
            pl.col("id_h3_h9").map_elements(lambda x: h3.h3_to_geo(x)[1], return_dtype=pl.Float64).alias("lon")
        ])

        # 2. Baixar Fronteiras de Bairros (Distritos)
        print("🏘️ Mapeando identidades administrativas (Bairros/Cidades)...")
        url_dist = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?intrarregiao=distrito&formato=application/vnd.geo+json"
        res_dist = requests.get(url_dist, headers=headers)
        res_dist.raise_for_status()
        with open("distritos.json", "wb") as f: f.write(res_dist.content)

        # 3. Join Espacial: Cada hexágono descobre onde "nasceu"
        # Usamos UNNEST e properties para evitar o erro de coluna não encontrada
        df_mestra = con.execute("""
            SELECT 
                h.id_h3_h9, 
                h.id_h3_h8, 
                h.lat, 
                h.lon,
                COALESCE(d.nm_mun, d.nm_municipio, 'Desconhecido') as municipio,
                COALESCE(d.nm_dis, d.nm_distrito, d.nm_mun, 'Desconhecido') as bairro
            FROM df_h3 h
            JOIN (SELECT * FROM st_read('distritos.json')) d 
            ON ST_Within(ST_Point(h.lon, h.lat), d.geom)
        """).pl()

        print(f"📊 Malha consolidada: {len(df_mestra)} registros.")

        # 4. Salvar no R2
        print("☁️ Subindo arquivo mestre para o Cloudflare R2...")
        buffer = io.BytesIO()
        df_mestra.write_parquet(buffer)
        buffer.seek(0)
        
        # O parâmetro 'auto' na região é vital para o R2
        r2 = boto3.client("s3", endpoint_url=R2_URL, aws_access_key_id=R2_KEY.strip(), 
                          aws_secret_access_key=R2_SECRET.strip(), config=Config(region_name="auto"))
        
        r2.upload_fileobj(buffer, BUCKET, "referencia/malha_mestra.parquet")
        print("✅ Sucesso! Malha Mestra disponível no R2.")

    except Exception as e:
        print(f"🚨 Erro: {e}")
        raise e

if __name__ == "__main__":
    criar_malha_mestra()
