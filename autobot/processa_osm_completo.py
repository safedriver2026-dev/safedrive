import os, duckdb, polars as pl, boto3, io, requests, zipfile, unicodedata, sys, traceback, shutil
import h3
from botocore.config import Config

def remover_acentos(texto):
    if texto is None or str(texto).upper() in ["NAN", "", "NULL"]: return "NAO MAPEADO"
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def get_h3_int(lat, lon):
    try: return int(h3.latlng_to_cell(lat, lon, 9), 16)
    except: return 0

def pipeline_osm_bbbike():
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    
    CAMINHO_MALHA_BASE = "malha geografica/malha_mestra_consolidada_2025.parquet"
    CAMINHO_BRONZE_INFRA = "datalake/bronze/malha_geografica_infraestrutura.parquet"
    
    # NOVA URL: Repositorio BBBike (Foco em Sao Paulo)
    URL_OSM_SP = "https://download.bbbike.org/osm/bbbike/SaoPaulo/SaoPaulo.osm.shp.zip"
    PASTA_TEMP = "temp_osm_dados"
    ARQUIVO_ZIP = "dados_osm.zip"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        print("🚀 [LOG] Iniciando Processamento via BBBIKE (Nova Fonte)...")

        # 1. DOWNLOAD ROBUSTO DO BBBIKE
        print("[LOG] Baixando base do BBBike (Sao Paulo)...")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        with requests.get(URL_OSM_SP, headers=headers, stream=True) as r:
            r.raise_for_status()
            with open(ARQUIVO_ZIP, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        # VERIFICACAO DE SEGURANCA: O arquivo tem o tamanho correto?
        tamanho_mb = os.path.getsize(ARQUIVO_ZIP) / (1024 * 1024)
        print(f"[LOG] Download concluido. Tamanho do arquivo: {tamanho_mb:.2f} MB")
        
        if tamanho_mb < 1.0:
            print("❌ [ERRO] O arquivo baixado e muito pequeno. O servidor bloqueou o download!")
            with open(ARQUIVO_ZIP, 'r', encoding='utf-8', errors='ignore') as f:
                print("Conteudo retornado pelo servidor:\n", f.read()[:500])
            sys.exit(1)

        # 2. EXTRACAO INTELIGENTE (Procurando places e points)
        print("[LOG] Extraindo shapefiles especificos...")
        os.makedirs(PASTA_TEMP, exist_ok=True)
        places_shp_path = ""
        points_shp_path = ""
        
        with zipfile.ZipFile(ARQUIVO_ZIP, 'r') as z:
            for f in z.namelist():
                nome_arquivo = f.split('/')[-1]
                if nome_arquivo.startswith('places.') or nome_arquivo.startswith('points.'):
                    z.extract(f, PASTA_TEMP)
                    if nome_arquivo == 'places.shp': places_shp_path = f
                    if nome_arquivo == 'points.shp': points_shp_path = f
        
        os.remove(ARQUIVO_ZIP)
        
        if not places_shp_path or not points_shp_path:
            raise Exception("Arquivos places.shp ou points.shp nao encontrados no ZIP.")

        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # 3. ATUALIZACAO DA MALHA GEOGRAFICA (BASE)
        print("[LOG] Executando Fallback de Bairros...")
        obj = r2.get_object(Bucket=BUCKET, Key=CAMINHO_MALHA_BASE)
        df_base = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        # No BBBike, a coluna de classificacao chama-se 'type'
        df_places = con.execute(f"SELECT name, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('{PASTA_TEMP}/{places_shp_path}') WHERE type IN ('suburb', 'neighbourhood', 'town', 'village') AND name IS NOT NULL").pl()
        df_places = df_places.with_columns(
            pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")
        ).select([
            pl.col("id_h3_int"), 
            pl.col("name").map_elements(remover_acentos, return_dtype=pl.String).alias("osm_bairro")
        ]).unique("id_h3_int")

        df_base = df_base.join(df_places, on="id_h3_int", how="left")
        df_base = df_base.with_columns(
            pl.when(pl.col("bairro") == "NAO MAPEADO").then(pl.col("osm_bairro").fill_null("NAO MAPEADO"))
            .otherwise(pl.col("bairro")).alias("bairro")
        ).drop("osm_bairro")

        # 4. CRIACAO DA MALHA DE INFRAESTRUTURA
        print("[LOG] Gerando arquivo de Infraestrutura Urbana...")
        # O BBBike junta tudo (POIs e Transportes) no points.shp
        df_inf = con.execute(f"""
            SELECT type as fclass, ST_Y(geom) as lat, ST_X(geom) as lon 
            FROM ST_Read('{PASTA_TEMP}/{points_shp_path}') 
            WHERE type IN ('bus_stop', 'station', 'railway_station', 'police', 'bank', 'bar', 'hospital', 'fuel')
        """).pl()
        
        df_infra = df_inf.with_columns(
            pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")
        ).pivot(values="fclass", index="id_h3_int", columns="fclass", aggregate_function="len").fill_null(0)

        # 5. UPLOAD
        print("[LOG] Fazendo upload dos arquivos finais para o R2...")
        buf_base = io.BytesIO()
        df_base.write_parquet(buf_base, compression="zstd")
        buf_base.seek(0)
        r2.upload_fileobj(buf_base, BUCKET, CAMINHO_MALHA_BASE)

        buf_inf = io.BytesIO()
        df_infra.write_parquet(buf_inf, compression="zstd")
        buf_inf.seek(0)
        r2.upload_fileobj(buf_inf, BUCKET, CAMINHO_BRONZE_INFRA)

        # 6. LIMPEZA
        shutil.rmtree(PASTA_TEMP)
        print("[LOG] Processo concluido com sucesso. Data Lake atualizado.")

    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    pipeline_osm_bbbike()
