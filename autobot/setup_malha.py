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
    R2_URL = os.getenv("R2_ENDPOINT_URL")
    R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
    BUCKET = os.getenv("R2_BUCKET_NAME")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    try:
        headers = {"User-Agent": "Mozilla/5.0"}

        # 1. Gerar Geometria H3
        print("⬢ Gerando malha H3 (Resolução 9) para São Paulo...")
        url_sp = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?formato=application/vnd.geo+json&qualidade=minima"
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

        # 2. Mapear Bairros (Distritos) - ESTRATÉGIA RESILIENTE
        print("🏘️ Mapeando Bairros (Tentando IBGE com Qualidade Mínima)...")
        # Adicionamos 'qualidade=minima' para evitar o erro 400
        url_distritos = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?intrarregiao=distrito&formato=application/vnd.geo+json&qualidade=minima"
        
        try:
            res_dist = requests.get(url_distritos, headers=headers, timeout=30)
            res_dist.raise_for_status()
            with open("distritos.json", "wb") as f: f.write(res_dist.content)
        except Exception as e:
            print(f"⚠️ IBGE falhou ({e}). Usando fonte alternativa (GitHub Mirror)...")
            # Fonte alternativa: Repositório que mantém GeoJSONs do IBGE atualizados
            url_mirror = "https://raw.githubusercontent.com/giuliano-macedo/geodata-br-geojson/master/sp/sp_distritos.json"
            res_mirror = requests.get(url_mirror, timeout=30)
            with open("distritos.json", "wb") as f: f.write(res_mirror.content)

        # 3. Join Espacial via DuckDB
        print("⚔️ Executando cruzamento espacial...")
        df_mestra = con.execute("""
            SELECT 
                h.id_h3_h9, h.id_h3_h8, h.lat, h.lon, 
                COALESCE(d.nm_mun, d.nm_municipio, 'São Paulo') as municipio, 
                COALESCE(d.nm_dis, d.nm_distrito, d.nm_mun, 'Bairro Desconhecido') as bairro
            FROM df_h3 h
            JOIN (SELECT * FROM st_read('distritos.json')) d ON ST_Within(ST_Point(h.lon, h.lat), d.geom)
        """).pl()

        # 4. Salvar no R2
        buffer = io.BytesIO()
        df_mestra.write_parquet(buffer)
        buffer.seek(0)
        r2 = boto3.client("s3", endpoint_url=R2_URL, aws_access_key_id=R2_KEY.strip(), 
                          aws_secret_access_key=R2_SECRET.strip(), config=Config(region_name="auto"))
        r2.upload_fileobj(buffer, BUCKET, "referencia/malha_mestra.parquet")

        print(f"✅ Malha Mestra concluída: {len(df_mestra)} registros.")

    except Exception as e:
        print(f"🚨 Erro Fatal: {e}")
        raise e

if __name__ == "__main__":
    criar_malha_mestra()
