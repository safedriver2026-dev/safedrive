import os, h3, polars as pl, duckdb, boto3, io, glob, unicodedata, sys, traceback
from botocore.config import Config

def remover_acentos(texto):
    if texto is None or texto == "": 
        return "NAO MAPEADO"
    # Normaliza para decompor caracteres acentuados e remove a acentuação
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def build_malha_mestra():
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    try:
        print("🔍 --- DEBUG: MAPEANDO ESTRUTURA DE ARQUIVOS ---")
        for root, dirs, files in os.walk("dados_ibge"):
            for f in files:
                print(f"📄 Encontrado: {os.path.join(root, f)}")

        print("\n⬢ Lendo Malha Base H3 do R2...")
        r2 = boto3.client("s3", **R2_CONF)
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        # Busca recursiva para garantir que achamos os arquivos mesmo em subpastas
        def find_file(pattern):
            files = glob.glob(pattern, recursive=True)
            if not files:
                print(f"🚨 ERRO: Nenhum arquivo encontrado para o padrao: {pattern}")
                sys.exit(1)
            return files[0]

        path_faces = find_file("dados_ibge/faces/**/*.json")
        path_mun = find_file("dados_ibge/municipios/**/*.shp")
        path_rgi = find_file("dados_ibge/imediata/**/*.shp")
        path_rint = find_file("dados_ibge/intermediaria/**/*.shp")

        print(f"🎯 Arquivos selecionados:\n - Faces: {path_faces}\n - Mun: {path_mun}")

        print("⚔️ Executando Cruzamento Espacial Múltiplo...")
        # Cruzamos o hexágono com polígonos (Cidades/Regiões) e a rua mais próxima (Faces)
        df_raw = con.execute(f"""
            SELECT 
                h.id_h3_h9,
                CAST(h.lat AS FLOAT) as latitude,
                CAST(h.lon AS FLOAT) as longitude,
                m.NM_MUN as cidade_nome,
                rgi.NM_RGI as regiao_imediata,
                rint.NM_RGINT as regiao_intermediaria,
                f.properties->>'NM_LOGRAD' as logradouro,
                f.properties->>'NM_BAIRRO' as bairro,
                CAST(f.properties->>'CD_SETOR' AS UINT64) as setor_id
            FROM df_hex h
            JOIN ST_Read('{path_mun}') m ON ST_Within(ST_Point(h.lon, h.lat), m.geom)
            JOIN ST_Read('{path_rgi}') rgi ON ST_Within(ST_Point(h.lon, h.lat), rgi.geom)
            JOIN ST_Read('{path_rint}') rint ON ST_Within(ST_Point(h.lon, h.lat), rint.geom)
            LEFT JOIN ST_Read('{path_faces}') f ON ST_DWithin(ST_Point(h.lon, h.lat), f.geom, 0.0001)
            QUALIFY ROW_NUMBER() OVER(PARTITION BY h.id_h3_h9 ORDER BY ST_Distance(ST_Point(h.lon, h.lat), f.geom) ASC) = 1
        """).pl()

        print("🔠 Formatando Textos (MAIÚSCULO E SEM ACENTO)...")
        cols_texto = ["cidade_nome", "regiao_imediata", "regiao_intermediaria", "logradouro", "bairro"]
        
        df_final = df_raw.with_columns([
            pl.col(c).map_elements(remover_acentos, return_dtype=pl.String) for c in cols_texto
        ]).with_columns([
            pl.col("id_h3_h9").map_elements(h3.string_to_int, return_dtype=pl.UInt64).alias("id_h3_int"),
            pl.col("cidade_nome").cast(pl.Categorical)
        ]).drop("id_h3_h9")

        print(f"📊 Preview do documento Final:\n{df_final.head(3)}")

        print("💾 Gravando Malha Mestra no R2 (Parquet ZSTD)...")
        buffer = io.BytesIO()
        df_final.write_parquet(buffer, compression="zstd", compression_level=3)
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, "ouro/malha_mestra_consolidada_2025.parquet")
        
        print("✅ Malha Mestra consolidada com sucesso!")

    except Exception:
        print("🚨 ERRO DETALHADO NO SCRIPT:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    build_malha_mestra()
