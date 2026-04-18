import os
import h3
import requests
import polars as pl
import duckdb
import boto3
from botocore.config import Config
import io
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class ArquitetoMalhaMestra:
    def __init__(self):
        self.bucket = os.getenv("R2_BUCKET_NAME")
        self.r2 = boto3.client(
            "s3",
            endpoint_url=os.getenv("R2_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            config=Config(region_name="auto")
        )
        self.con = duckdb.connect()
        self.con.execute("INSTALL spatial; LOAD spatial;")

    def gerar_grid_base(self):
        logger.info("🗺️ Gerando Grid Mestre H9/H8 via IBGE...")
        url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?formato=application/vnd.geo+json"
        geo_sp = requests.get(url).json()
        coords = geo_sp['features'][0]['geometry']['coordinates'][0][0]
        polygon_h3 = {"type": "Polygon", "coordinates": [[ [c[1], c[0]] for c in coords ]]}
        
        hexs_h9 = list(h3.polyfill(polygon_h3, 9))
        return pl.DataFrame({"H3_INDEX_H9": hexs_h9}).with_columns(
            pl.col("H3_INDEX_H9").map_elements(lambda x: h3.cell_to_parent(x, 8), return_dtype=pl.Utf8).alias("H3_INDEX_H8")
        )

    def extrair_nomes_osm(self):
        logger.info("🏘️ Extraindo Bairros e Ruas do OSM (DuckDB)...")
        # Extrai ruas e bairros com suas coordenadas centrais
        df_geo = self.con.execute("""
            SELECT tags['name'] as nome, 'rua' as cat, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon
            FROM st_read('sao-paulo-latest.osm.pbf', layer='lines')
            WHERE tags['highway'] IS NOT NULL AND tags['name'] IS NOT NULL
            UNION ALL
            SELECT tags['name'] as nome, 'bairro' as cat, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon
            FROM st_read('sao-paulo-latest.osm.pbf', layer='multipolygons')
            WHERE tags['place'] IN ('suburb', 'neighbourhood') AND tags['name'] IS NOT NULL
        """).pl()
        
        # Mapeia cada nome para um H9
        return df_geo.with_columns(
            pl.struct(["lat", "lon"]).map_elements(lambda x: h3.latlng_to_cell(x["lat"], x["lon"], 9), return_dtype=pl.Utf8).alias("H3_INDEX_H9")
        )

    def construir_malha_mestra(self):
        grid_base = self.gerar_grid_base()
        nomes_osm = self.extrair_nomes_osm()
        
        logger.info("📦 Consolidando Hierarquia Geográfica...")
        
        # Agrupa os nomes por H9 (quarteirão)
        meta_agrupada = nomes_osm.group_by("H3_INDEX_H9").agg([
            pl.col("nome").filter(pl.col("cat") == "bairro").unique().alias("bairros"),
            pl.col("nome").filter(pl.col("cat") == "rua").unique().alias("logradouros")
        ])
        
        # Faz o Join com a Malha Virgem
        malha_mestra = grid_base.join(meta_agrupada, on="H3_INDEX_H9", how="left").with_columns([
            pl.lit("São Paulo").alias("estado"),
            pl.col("bairros").fill_null([]),
            pl.col("logradouros").fill_null([])
        ])
        
        # Upload para o R2
        buffer = io.BytesIO()
        malha_mestra.write_parquet(buffer)
        buffer.seek(0)
        
        self.r2.upload_fileobj(buffer, self.bucket, "referencia/malha_mestra_sp.parquet")
        logger.info(f"✨ Malha Mestra concluída com {len(malha_mestra)} células H9.")

if __name__ == "__main__":
    ArquitetoMalhaMestra().construir_malha_mestra()
