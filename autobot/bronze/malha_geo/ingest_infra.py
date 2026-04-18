import os, polars as pl, duckdb, boto3, sys, traceback, io, h3
from botocore.config import Config

def processar_infraestrutura():
    print("🚀 [INICIO] Processamento Puro: INFRAESTRUTURA OSM (Arquivos Locais)")
    
    # 1. Configuração do Data Lake
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    r2 = boto3.client("s3", **R2_CONF)
    BUCKET = os.getenv("R2_BUCKET_NAME")
    
    try:
        # 2. Setup do Motor Espacial
        con = duckdb.connect()
        con.execute("PRAGMA memory_limit='4GB'; PRAGMA threads=2;")
        con.execute("INSTALL spatial; LOAD spatial;")
        
        # 3. Leitura e Filtro (O arquivo já foi baixado pelo YAML)
        print("🏢 Lendo Shapefile de POIs (Pontos de Interesse) do OpenStreetMap...")
        shp_file = "temp_osm/gis_osm_pois_free_1.shp"
        
        # Filtramos apenas o que importa para risco e segurança
        df_infra = con.execute(f"""
            SELECT fclass, CAST(ST_Y(geom) AS FLOAT) as lat, CAST(ST_X(geom) AS FLOAT) as lon 
            FROM ST_Read('{shp_file}') 
            WHERE fclass IN ('police', 'hospital', 'bar', 'bank', 'fuel')
        """).pl()
        
        # 4. Enriquecimento H3 (Convertendo Coordenada para Hexágono)
        print("⬢ Convertendo coordenadas para Hexágonos H3 (Resolução 9)...")
        df_infra = df_infra.with_columns(
            pl.struct(["lat", "lon"]).map_elements(
                lambda x: int(h3.latlng_to_cell(x["lat"], x["lon"], 9), 16), 
                return_dtype=pl.UInt64
            ).alias("id_h3_int")
        )

        # 5. Pivotagem (De Linhas para Colunas)
        print("📊 Pivotando tabela (Agregando contagem de infraestrutura por Hexágono)...")
        # Transforma os fclass em colunas: qtd_police, qtd_bar, etc.
        df_infra = df_infra.pivot(
            on="fclass", 
            index="id_h3_int", 
            values="fclass", 
            aggregate_function="len"
        ).fill_null(0) # Se não tem bar, é 0 (não nulo)

        # 6. Upload para o R2
        print("💾 Enviando malha de infraestrutura para a Camada Bronze (R2)...")
        buf = io.BytesIO()
        df_infra.write_parquet(buf, compression="zstd")
        buf.seek(0)
        
        r2.upload_fileobj(buf, BUCKET, "datalake/bronze/malha_geografica/malha_infraestrutura_bronze.parquet")
        
        print(f"🏁 [SUCESSO] Processamento concluído. {len(df_infra):,} hexágonos receberam infraestrutura.")

    except Exception:
        print("\n🚨 ERRO NO PROCESSAMENTO DA INFRAESTRUTURA:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    processar_infraestrutura()
