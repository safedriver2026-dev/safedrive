import os, duckdb, polars as pl, boto3, io, requests, zipfile, unicodedata, sys, traceback, shutil
import h3
from botocore.config import Config

def remover_acentos(texto):
    if texto is None or str(texto).upper() in ["NAN", "", "NULL"]: 
        return "NAO MAPEADO"
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def get_h3_int(lat, lon):
    try:
        return int(h3.latlng_to_cell(lat, lon, 9), 16)
    except:
        return 0

def pipeline_osm_total():
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    CAMINHO_MALHA = "malha geografica/malha_mestra_consolidada_2025.parquet"
    
    URL_OSM_SP = "https://download.geofabrik.de/south-america/brazil/sao-paulo-latest-free.shp.zip"
    PASTA_OSM = "temp_osm_dados"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        print("[LOG] Iniciando Processamento Unificado do OpenStreetMap...")

        # 1. DOWNLOAD E EXTRACAO SEGREGADA
        print("[LOG] Baixando a base completa do OSM para SP...")
        os.makedirs(PASTA_OSM, exist_ok=True)
        r = requests.get(URL_OSM_SP, stream=True)
        r.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            arquivos_alvo = ['gis_osm_places_free_1', 'gis_osm_transport_free_1', 'gis_osm_pois_free_1']
            for arq_zip in z.namelist():
                if any(alvo in arq_zip for alvo in arquivos_alvo):
                    z.extract(arq_zip, PASTA_OSM)

        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # 2. DOWNLOAD DA MALHA MESTRA
        print(f"[LOG] Baixando Malha Mestra de '{CAMINHO_MALHA}'...")
        obj = r2.get_object(Bucket=BUCKET, Key=CAMINHO_MALHA)
        df_malha = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        # ==========================================
        # FASE A: TAPA-BURACOS (BAIRROS)
        # ==========================================
        print("[LOG] FASE A: Processando lugares para Fallback de Bairros...")
        df_places = con.execute(f"""
            SELECT name as osm_bairro_raw, ST_Y(geom) as lat, ST_X(geom) as lon
            FROM ST_Read('{PASTA_OSM}/gis_osm_places_free_1.shp')
            WHERE fclass IN ('suburb', 'neighbourhood', 'town', 'village') AND name IS NOT NULL
        """).pl()

        df_places = df_places.with_columns(
            pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int"),
            pl.col("osm_bairro_raw").map_elements(remover_acentos, return_dtype=pl.String).alias("osm_bairro")
        ).unique(subset=["id_h3_int"], keep="first").drop(["lat", "lon", "osm_bairro_raw"])

        df_malha = df_malha.join(df_places, on="id_h3_int", how="left")
        df_malha = df_malha.with_columns(
            pl.when(pl.col("bairro") == "NAO MAPEADO").then(pl.col("osm_bairro").fill_null("NAO MAPEADO"))
            .otherwise(pl.col("bairro")).alias("bairro")
        ).drop("osm_bairro")

        # ==========================================
        # FASE B: FEATURES (STOPS E POIS)
        # ==========================================
        print("[LOG] FASE B: Processando Stops de Transporte e Locais de Risco...")
        # Lemos transportes (paradas de onibus/metro) e POIs de risco juntos
        df_features = con.execute(f"""
            SELECT fclass, ST_Y(geom) as lat, ST_X(geom) as lon 
            FROM ST_Read('{PASTA_OSM}/gis_osm_transport_free_1.shp')
            WHERE fclass IN ('bus_stop', 'tram_stop', 'railway_station')
            UNION ALL
            SELECT fclass, ST_Y(geom) as lat, ST_X(geom) as lon 
            FROM ST_Read('{PASTA_OSM}/gis_osm_pois_free_1.shp')
            WHERE fclass IN ('police', 'bank', 'atm', 'bar', 'nightclub', 'hospital')
        """).pl()

        df_features = df_features.with_columns(
            pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")
        )

        print("[LOG] Agregando features por Hexagono...")
        df_agregado = df_features.pivot(
            values="fclass", index="id_h3_int", columns="fclass", aggregate_function="len"
        ).fill_null(0)

        # Dicionario para traduzir as colunas do OSM para o nosso padrao
        traducao = {
            "bus_stop": "qtd_pontos_onibus",
            "railway_station": "qtd_estacoes_trem_metro",
            "tram_stop": "qtd_paradas_vlt",
            "police": "qtd_delegacias",
            "bank": "qtd_bancos",
            "atm": "qtd_caixas_eletronicos",
            "bar": "qtd_bares",
            "nightclub": "qtd_baladas",
            "hospital": "qtd_hospitais"
        }
        renomeios_validos = {k: v for k, v in traducao.items() if k in df_agregado.columns}
        df_agregado = df_agregado.rename(renomeios_validos)

        # Junta as features na malha
        df_malha = df_malha.join(df_agregado, on="id_h3_int", how="left")
        colunas_novas = list(renomeios_validos.values())
        df_malha = df_malha.with_columns([
            pl.col(c).fill_null(0).cast(pl.UInt32) for c in colunas_novas if c in df_malha.columns
        ])

        # ==========================================
        # FINALIZACAO E LIMPEZA
        # ==========================================
        print("[LOG] Guardando a Malha Definitiva Enriquecida no R2...")
        buffer = io.BytesIO()
        df_malha.write_parquet(buffer, compression="zstd")
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, CAMINHO_MALHA)

        print(f"[LOG] Limpando a pasta segregada '{PASTA_OSM}' do ambiente...")
        if os.path.exists(PASTA_OSM):
            shutil.rmtree(PASTA_OSM)
            
        print("[LOG] Pipeline do OpenStreetMap concluido com sucesso. Ambiente limpo.")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    pipeline_osm_total()
