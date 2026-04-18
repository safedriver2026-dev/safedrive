import os, duckdb, polars as pl, boto3, io, unicodedata, sys, traceback
import h3
from botocore.config import Config

def remover_acentos(texto):
    if texto is None or str(texto).upper() in ["NAN", "", "NULL", "NONE"]: return "NAO MAPEADO"
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def get_h3_int(lat, lon):
    try: return int(h3.latlng_to_cell(lat, lon, 9), 16)
    except: return 0

def pipeline_bronze_final():
    print("🚀 [INICIO] Organização da Camada Bronze - SafeDriver Autobot")
    
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    
    # --- MAPEAMENTO DE CAMINHOS ---
    PATH_ORIGEM_MESTRA = "malha geografica/malha_mestra_consolidada_2025.parquet"
    PATH_ORIGEM_INFRA = "datalake/bronze/malha_geografica_infraestrutura.parquet"
    
    # Destinos Finais conforme solicitado
    DESTINO_BRONZE = "datalake/bronze/malha geografica/"
    FILE_MESTRA = f"{DESTINO_BRONZE}malha_mestra_bronze.parquet"
    FILE_INFRA = f"{DESTINO_BRONZE}malha_geografica_infraestrutura.parquet"
    
    PASTA_OSM = "temp_estadual"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # --- 1. CARREGAR E ENRIQUECER MALHA MESTRA ---
        print("[LOG] Baixando e Refatorando Malha Mestra...")
        obj = r2.get_object(Bucket=BUCKET, Key=PATH_ORIGEM_MESTRA)
        df_base = pl.read_parquet(io.BytesIO(obj['Body'].read()))
        
        # Auditoria prévia
        logradouro_nulo_pre = df_base.filter(pl.col("logradouro").is_null() | (pl.col("logradouro") == "LOGRADOURO NAO IDENTIFICADO")).height

        # Resgate OSM (Ruas)
        df_roads = con.execute(f"SELECT name as osm_rua, ST_Y(ST_Centroid(geom)) as lat, ST_X(ST_Centroid(geom)) as lon FROM ST_Read('{PASTA_OSM}/gis_osm_roads_free_1.shp') WHERE name IS NOT NULL").pl()
        df_roads_h3 = df_roads.with_columns(pl.struct(["lat", "lon"]).map_elements(lambda x: get_h3_int(x["lat"], x["lon"]), return_dtype=pl.UInt64).alias("id_h3_int")).unique("id_h3_int").select(["id_h3_int", "osm_rua"])

        # Aplicação do Enriquecimento
        df_final = df_base.join(df_roads_h3, on="id_h3_int", how="left")
        df_final = df_final.with_columns(
            pl.when(pl.col("logradouro").is_null() | (pl.col("logradouro") == "LOGRADOURO NAO IDENTIFICADO"))
            .then(pl.col("osm_rua").fill_null("LOGRADOURO NAO IDENTIFICADO"))
            .otherwise(pl.col("logradouro")).alias("logradouro")
        ).drop("osm_rua")

        # --- 2. MOVER INFRAESTRUTURA PARA O NOVO CAMINHO ---
        print("[LOG] Reposicionando Malha de Infraestrutura...")
        obj_inf = r2.get_object(Bucket=BUCKET, Key=PATH_ORIGEM_INFRA)
        df_infra = pl.read_parquet(io.BytesIO(obj_inf['Body'].read()))

        # --- 3. UPLOAD PARA A ESTRUTURA FINAL ---
        print(f"📤 Enviando para: {DESTINO_BRONZE}")
        
        # Upload Malha Mestra Bronze
        buf_mestra = io.BytesIO()
        df_final.write_parquet(buf_mestra, compression="zstd")
        buf_mestra.seek(0)
        r2.upload_fileobj(buf_mestra, BUCKET, FILE_MESTRA)

        # Upload Infraestrutura Bronze
        buf_infra = io.BytesIO()
        df_infra.write_parquet(buf_infra, compression="zstd")
        buf_infra.seek(0)
        r2.upload_fileobj(buf_infra, BUCKET, FILE_INFRA)

        # --- 4. LIMPEZA (DELETE DOS ANTIGOS) ---
        print("[LOG] Iniciando limpeza de arquivos obsoletos...")
        chaves_para_deletar = [PATH_ORIGEM_MESTRA, PATH_ORIGEM_INFRA]
        for key in chaves_para_deletar:
            try:
                r2.delete_object(Bucket=BUCKET, Key=key)
                print(f"🗑️ Deletado com sucesso: {key}")
            except:
                print(f"⚠️ Não foi possível deletar {key} (pode já ter sido removido)")

        # --- 5. LOG DETALHADO ---
        logradouro_nulo_pos = df_final.filter(pl.col("logradouro") == "LOGRADOURO NAO IDENTIFICADO").height
        print("\n" + "="*45)
        print("✅ CAMADA BRONZE CONSOLIDADA")
        print(f"🛣️  Logradouros Recuperados: {logradouro_nulo_pre - logradouro_nulo_pos}")
        print(f"📂 Malha Mestra: {FILE_MESTRA}")
        print(f"📂 Infraestrutura: {FILE_INFRA}")
        print("="*45)

    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__":
    pipeline_bronze_final()
