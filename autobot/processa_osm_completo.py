import os, duckdb, polars as pl, boto3, io, requests, unicodedata, sys, traceback
import h3
from botocore.config import Config

def remover_acentos(texto):
    if texto is None or str(texto).upper() in ["NAN", "", "NULL"]: return "NAO MAPEADO"
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def get_h3_int(lat, lon):
    try: return int(h3.latlng_to_cell(lat, lon, 9), 16)
    except: return 0

def pipeline_osm_api():
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    
    CAMINHO_MALHA_BASE = "malha geografica/malha_mestra_consolidada_2025.parquet"
    CAMINHO_BRONZE_INFRA = "datalake/bronze/malha_geografica_infraestrutura.parquet"

    # AREA 3600059470 e o codigo exato do Estado de Sao Paulo no OpenStreetMap
    query_overpass = """
    [out:json][timeout:900];
    area(3600059470)->.sp;
    (
      nwr["place"~"suburb|neighbourhood|town|village"](area.sp);
      nwr["amenity"~"police|bank|bar|hospital|fuel"](area.sp);
      nwr["highway"="bus_stop"](area.sp);
      nwr["railway"="station"](area.sp);
    );
    out center;
    """

    try:
        r2 = boto3.client("s3", **R2_CONF)
        print("[LOG] Iniciando Processamento via Overpass API (Sem download de ZIP)...")

        # 1. CHAMADA A API (Direto para a Memoria)
        print("[LOG] Consultando servidores do OpenStreetMap (Isto pode levar 1 a 2 minutos)...")
        url = "https://overpass-api.de/api/interpreter"
        # Um User-Agent customizado evita bloqueios
        headers = {'User-Agent': 'SafeDriver-Autobot-Pipeline/1.0 (github actions)'}
        
        response = requests.post(url, data={'data': query_overpass}, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        elementos = data.get('elements', [])
        print(f"[LOG] Dados recebidos com sucesso! {len(elementos)} locais de interesse encontrados.")

        # 2. PROCESSAMENTO DO JSON PARA H3
        print("[LOG] Mapeando coordenadas JSON para Hexagonos H3...")
        lugares = []
        infra = []

        for el in elementos:
            # A API devolve lat/lon diretamente para Nodes, ou dentro de 'center' para Areas/Vias
            lat = el.get('lat')
            lon = el.get('lon')
            
            if not lat or not lon:
                center = el.get('center', {})
                lat = center.get('lat')
                lon = center.get('lon')
                
            if not lat or not lon: continue

            h3_id = get_h3_int(lat, lon)
            if h3_id == 0: continue

            tags = el.get('tags', {})

            # Separar Fallback de Bairros
            if 'place' in tags and 'name' in tags:
                lugares.append({'id_h3_int': h3_id, 'osm_bairro_raw': tags['name']})

            # Separar Infraestrutura
            fclass = None
            if 'amenity' in tags: fclass = tags['amenity']
            elif tags.get('highway') == 'bus_stop': fclass = 'bus_stop'
            elif tags.get('railway') == 'station': fclass = 'railway_station'

            if fclass:
                infra.append({'id_h3_int': h3_id, 'fclass': fclass})

        # 3. ATUALIZACAO DA MALHA GEOGRAFICA (BASE)
        print("[LOG] Lendo Malha Mestra do R2 e aplicando Fallback de Bairros...")
        obj = r2.get_object(Bucket=BUCKET, Key=CAMINHO_MALHA_BASE)
        df_base = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        if lugares:
            df_places = pl.DataFrame(lugares)
            df_places = df_places.select([
                pl.col("id_h3_int"),
                pl.col("osm_bairro_raw").map_elements(remover_acentos, return_dtype=pl.String).alias("osm_bairro")
            ]).unique("id_h3_int")

            df_base = df_base.join(df_places, on="id_h3_int", how="left")
            df_base = df_base.with_columns(
                pl.when(pl.col("bairro") == "NAO MAPEADO").then(pl.col("osm_bairro").fill_null("NAO MAPEADO"))
                .otherwise(pl.col("bairro")).alias("bairro")
            ).drop("osm_bairro")

        # 4. CRIACAO DA MALHA DE INFRAESTRUTURA
        print("[LOG] Pivotando dados para a Tabela de Infraestrutura...")
        if infra:
            df_infra = pl.DataFrame(infra)
            df_infra = df_infra.pivot(values="fclass", index="id_h3_int", columns="fclass", aggregate_function="len").fill_null(0)
        else:
            df_infra = pl.DataFrame({"id_h3_int": [0]}) # Fallback de seguranca

        # 5. UPLOAD DIRETO DA MEMORIA
        print("[LOG] Fazendo upload dos ficheiros Parquet para o R2...")
        buf_base = io.BytesIO()
        df_base.write_parquet(buf_base, compression="zstd")
        buf_base.seek(0)
        r2.upload_fileobj(buf_base, BUCKET, CAMINHO_MALHA_BASE)

        buf_inf = io.BytesIO()
        df_infra.write_parquet(buf_inf, compression="zstd")
        buf_inf.seek(0)
        r2.upload_fileobj(buf_inf, BUCKET, CAMINHO_BRONZE_INFRA)

        print("[LOG] Pipeline da API concluido! 100% executado em memoria (Sem ZIPs).")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    pipeline_osm_api()
