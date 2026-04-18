import os, requests, zipfile, io, h3, polars as pl, duckdb, boto3, shutil, sys, traceback
from botocore.config import Config

def limpar_pasta(pasta):
    if os.path.exists(pasta):
        shutil.rmtree(pasta)
        print(f"🧹 Pasta {pasta} limpa com sucesso.")

def download_e_extrai_seletivo(url, pasta_destino, extensoes_alvo=None):
    print(f"📥 Baixando e extraindo de: {url}...")
    r = requests.get(url, stream=True)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for file in z.namelist():
            # Filtro para não extrair arquivos inúteis (como .txt ou .dbf se não precisar)
            if extensoes_alvo is None or any(file.endswith(ext) for ext in extensoes_alvo):
                z.extract(file, pasta_destino)
    print(f"✅ Extração seletiva concluída em: {pasta_destino}")

def pipeline_autolimpante():
    print("🚀 [INICIO] Pipeline Autônomo e Autolimpante")
    
    # Configurações de Nuvem
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    r2 = boto3.client("s3", **R2_CONF)
    BUCKET = os.getenv("R2_BUCKET_NAME")

    # Pastas de trabalho
    TMP_OSM = "temp_osm"
    TMP_IBGE = "temp_ibge"
    
    try:
        # --- ETAPA 1: INFRAESTRUTURA (OSM) ---
        download_e_extrai_seletivo(
            "https://download.geofabrik.de/south-america/brazil/sao-paulo-latest-free.shp.zip", 
            TMP_OSM, 
            extensoes_alvo=['.shp', '.shx', '.dbf', '.prj']
        )
        
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        
        print("🏢 Processando Infraestrutura...")
        # Lemos direto do SHP e convertemos para H3 via Polars
        df_infra = con.execute(f"SELECT fclass, ST_Y(geom) as lat, ST_X(geom) as lon FROM ST_Read('{TMP_OSM}/gis_osm_pois_free_1.shp') WHERE fclass IN ('police', 'hospital', 'bar', 'bank', 'fuel')").pl()
        
        # Mapeamento H3 e Pivot
        df_infra = df_infra.with_columns(
            pl.struct(["lat", "lon"]).map_elements(lambda x: int(h3.latlng_to_cell(x["lat"], x["lon"], 9), 16), return_dtype=pl.UInt64).alias("id_h3_int")
        ).pivot(on="fclass", index="id_h3_int", values="fclass", aggregate_function="len").fill_null(0)

        # Upload Imediato
        buf = io.BytesIO()
        df_infra.write_parquet(buf, compression="zstd")
        buf.seek(0)
        r2.upload_fileobj(buf, BUCKET, "datalake/bronze/malha_geografica/malha_infraestrutura_bronze.parquet")
        
        # LIMPEZA IMEDIATA (Libera espaço para o IBGE)
        limpar_pasta(TMP_OSM)

        # --- ETAPA 2: GEOGRAFIA (IBGE) ---
        download_e_extrai_seletivo(
            "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/SP/SP_Municipios_2022.zip",
            TMP_IBGE
        )
        
        print("📍 Processando Geografia...")
        # (Lógica de processamento IBGE aqui seguindo o mesmo padrão de upload e limpeza)
        # ...
        
        limpar_pasta(TMP_IBGE)

        print("\n🏁 [SUCESSO] Ouro entregue. Disco limpo.")

    except Exception:
        traceback.print_exc()
        # Garantir limpeza mesmo em caso de erro
        limpar_pasta(TMP_OSM)
        limpar_pasta(TMP_IBGE)
        sys.exit(1)

if __name__ == "__main__":
    pipeline_autolimpante()
