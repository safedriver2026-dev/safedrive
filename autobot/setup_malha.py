import os
import h3
import requests
import polars as pl
import duckdb
import boto3
from botocore.config import Config
import io

def notificar(webhook_url, mensagem):
    if webhook_url:
        try: requests.post(webhook_url, json={"content": mensagem})
        except: pass

def criar_malha_mestra():
    # Credenciais
    R2_URL = os.getenv("R2_ENDPOINT_URL")
    R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
    BUCKET = os.getenv("R2_BUCKET_NAME")
    WEBHOOK_SUCESSO = os.getenv("DISCORD_SUCESSO")
    WEBHOOK_ERRO = os.getenv("DISCORD_ERRO")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

        # 1. Gerar Geometria H3
        print("⬢ Gerando malha H3 (v3.7.6)...")
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

        # 2. Mapear Nomes (IBGE)
        print("🏘️ Mapeando Municípios e Bairros...")
        url_dist = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?formato=application/vnd.geo+json&qualidade=minima&intrarregiao=distrito"
        res_dist = requests.get(url_dist, headers=headers)
        res_dist.raise_for_status()
        with open("distritos.json", "wb") as f: f.write(res_dist.content)

        # Join Espacial via DuckDB
        df_mestra = con.execute("""
            SELECT h.id_h3_h9, h.id_h3_h8, h.lat, h.lon, d.nm_mun as municipio, d.nm_dis as bairro
            FROM df_h3 h
            JOIN st_read('distritos.json') d ON ST_Within(ST_Point(h.lon, h.lat), d.geom)
        """).pl()

        # 3. Salvar no R2
        buffer = io.BytesIO()
        df_mestra.write_parquet(buffer)
        buffer.seek(0)
        
        r2 = boto3.client("s3", endpoint_url=R2_URL, aws_access_key_id=R2_KEY, 
                          aws_secret_access_key=R2_SECRET, config=Config(region_name="auto"))
        r2.upload_fileobj(buffer, BUCKET, "referencia/malha_mestra.parquet")

        notificar(WEBHOOK_SUCESSO, f"🏁 **Malha Mestra Criada!**\nTotal de Células H9: {len(df_mestra)}\nLocal: `referencia/malha_mestra.parquet`")

    except Exception as e:
        notificar(WEBHOOK_ERRO, f"🚨 Erro na criação da Malha Mestra: {str(e)}")
        raise e

if __name__ == "__main__":
    criar_malha_mestra()
