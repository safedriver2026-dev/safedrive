import os
import h3
import requests
import polars as pl
import duckdb
import boto3
from botocore.config import Config
import io

def notificar_discord(webhook_url, mensagem):
    if webhook_url:
        try:
            requests.post(webhook_url, json={"content": mensagem})
        except:
            pass

def processar_malha_referencia():
    R2_URL = os.getenv("R2_ENDPOINT_URL")
    R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
    BUCKET = os.getenv("R2_BUCKET_NAME")
    WEBHOOK_SUCESSO = os.getenv("DISCORD_SUCESSO")
    WEBHOOK_ERRO = os.getenv("DISCORD_ERRO")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    try:
        print("⬢ Gerando geometria H3 (v3.7.6) para o estado de São Paulo...")
        url_sp = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?formato=application/vnd.geo+json"
        res_sp = requests.get(url_sp).json()
        coords_ibge = res_sp['features'][0]['geometry']
        
        hexs = set()
        geometrias = coords_ibge['coordinates'] if coords_ibge['type'] == 'MultiPolygon' else [coords_ibge['coordinates']]
        for p in geometrias:
            # Padrão H3 v3.7.x
            poligono = {'type': 'Polygon', 'coordinates': [[[c[1], c[0]] for c in p[0]]]}
            hexs.update(h3.polyfill(poligono, 9))

        # Mapeamento usando as funções consolidadas do H3 v3
        df_h3 = pl.DataFrame({"id_h3_h9": list(hexs)}).with_columns([
            pl.col("id_h3_h9").map_elements(lambda x: h3.h3_to_parent(x, 8), return_dtype=pl.Utf8).alias("id_h3_h8"),
            pl.col("id_h3_h9").map_elements(lambda x: h3.h3_to_geo(x)[0], return_dtype=pl.Float64).alias("lat"),
            pl.col("id_h3_h9").map_elements(lambda x: h3.h3_to_geo(x)[1], return_dtype=pl.Float64).alias("lon")
        ])

        print("🏘️ Mapeando divisões de Bairros e Municípios...")
        url_distritos = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?formato=application/vnd.geo+json&qualidade=minima&intrarregiao=distrito"
        with open("distritos.json", "wb") as f:
            f.write(requests.get(url_distritos).content)

        df_base = con.execute("""
            SELECT h.id_h3_h9, h.id_h3_h8, h.lat, h.lon, d.nm_mun as municipio, d.nm_dis as bairro
            FROM df_h3 h
            JOIN st_read('distritos.json') d ON ST_Within(ST_Point(h.lon, h.lat), d.geom)
        """).pl()

        print("🛣️ Vinculando nomes de ruas aos hexágonos...")
        df_ruas_raw = con.execute("""
            SELECT tags['name'] as nome_rua, ST_Y(ST_Centroid(geom)) as r_lat, ST_X(ST_Centroid(geom)) as r_lon
            FROM st_read('sao-paulo-latest.osm.pbf')
            WHERE tags['highway'] IS NOT NULL AND tags['name'] IS NOT NULL
        """).pl()

        # Usando a função h3.geo_to_h3 para converter as ruas
        df_ruas_h3 = df_ruas_raw.with_columns(
            pl.struct(["r_lat", "r_lon"]).map_elements(
                lambda x: h3.geo_to_h3(x["r_lat"], x["r_lon"], 9), 
                return_dtype=pl.Utf8
            ).alias("id_h3_h9")
        ).group_by("id_h3_h9").agg(pl.col("nome_rua").unique().alias("logradouros"))

        df_referencia = df_base.join(df_ruas_h3, on="id_h3_h9", how="left").with_columns(
            pl.col("logradouros").fill_null([])
        )

        print("☁️ Enviando para o Cloudflare R2...")
        buffer = io.BytesIO()
        df_referencia.write_parquet(buffer)
        buffer.seek(0)
        
        r2 = boto3.client("s3", endpoint_url=R2_URL, aws_access_key_id=R2_KEY, 
                          aws_secret_access_key=R2_SECRET, config=Config(region_name="auto"))
        r2.upload_fileobj(buffer, BUCKET, "referencia/malha_referencia.parquet")

        notificar_discord(WEBHOOK_SUCESSO, f"🏁 **Sucesso:** Malha de Referência Gerada com {len(df_referencia)} células.")

    except Exception as e:
        notificar_discord(WEBHOOK_ERRO, f"🚨 **Erro no Processamento:** {str(e)}")
        print(f"Erro: {e}")
        raise e

if __name__ == "__main__":
    processar_malha_referencia()
