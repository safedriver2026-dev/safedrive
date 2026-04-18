import os, h3, polars as pl, duckdb, boto3, io, glob, unicodedata, sys, traceback
from botocore.config import Config

def remover_acentos(texto):
    if texto is None or texto == "" or str(texto).upper() == "NAN": 
        return "NAO MAPEADO"
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

    # 1. Defesas do GitHub Actions
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='6GB'")
    con.execute("PRAGMA preserve_insertion_order=false")
    con.execute("INSTALL spatial; LOAD spatial;")

    try:
        def find_file(pattern):
            files = glob.glob(pattern, recursive=True)
            if not files:
                print(f"🚨 ERRO: Padrao nao encontrado: {pattern}")
                sys.exit(1)
            return files[0]

        paths = {
            "municipios": find_file("dados_ibge/municipios/**/*.shp"),
            "imediata": find_file("dados_ibge/imediata/**/*.shp"),
            "intermediaria": find_file("dados_ibge/intermediaria/**/*.shp"),
            "faces": find_file("dados_ibge/faces/**/*.json")
        }

        print("\n📦 Materializando Tabelas Espaciais na Memória (Gerando Índices)...")
        # Isso impede o Produto Cartesiano de 78GB
        con.execute(f"CREATE TABLE tb_mun AS SELECT * FROM ST_Read('{paths['municipios']}')")
        con.execute(f"CREATE TABLE tb_rgi AS SELECT * FROM ST_Read('{paths['imediata']}')")
        con.execute(f"CREATE TABLE tb_rint AS SELECT * FROM ST_Read('{paths['intermediaria']}')")
        con.execute(f"CREATE TABLE tb_faces AS SELECT * FROM ST_Read('{paths['faces']}')")

        print("\n⬢ Lendo Malha Base H3 do R2...")
        r2 = boto3.client("s3", **R2_CONF)
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        # 2. Processamento em Lotes (Chunking)
        lote_tamanho = 250000
        total_linhas = len(df_hex)
        resultados = []

        print(f"\n⚔️ Iniciando Cruzamento Espacial em Lotes (Total: {total_linhas} linhas)...")
        
        for i in range(0, total_linhas, lote_tamanho):
            print(f"🔄 Processando Lote: {i} até {i + lote_tamanho}...")
            
            # Corta um pedaço do dataframe e registra no DuckDB
            df_lote = df_hex.slice(i, lote_tamanho)
            con.register("tb_lote_hex", df_lote.to_arrow())
            
            # Executa o Join apenas para este lote
            lote_raw = con.execute("""
                SELECT 
                    h.id_h3_h9,
                    CAST(h.lat AS FLOAT) as latitude,
                    CAST(h.lon AS FLOAT) as longitude,
                    m.NM_MUN as cidade_nome,
                    rgi.NM_RGI as regiao_imediata,
                    rint.NM_RGINT as regiao_intermediaria,
                    CONCAT_WS(' ', f.NM_TIP_LOG, f.NM_TIT_LOG, f.NM_LOG) as logradouro,
                    'NAO MAPEADO' as bairro,
                    CAST(f.CD_SETOR AS UINT64) as setor_id
                FROM tb_lote_hex h
                LEFT JOIN tb_mun m ON ST_Within(ST_Point(h.lon, h.lat), m.geom)
                LEFT JOIN tb_rgi rgi ON ST_Within(ST_Point(h.lon, h.lat), rgi.geom)
                LEFT JOIN tb_rint rint ON ST_Within(ST_Point(h.lon, h.lat), rint.geom)
                LEFT JOIN tb_faces f ON ST_DWithin(ST_Point(h.lon, h.lat), f.geom, 0.0001)
                QUALIFY ROW_NUMBER() OVER(PARTITION BY h.id_h3_h9 ORDER BY ST_Distance(ST_Point(h.lon, h.lat), f.geom) ASC) = 1
            """).pl()
            
            resultados.append(lote_raw)
            # Limpa o lote da memória do DuckDB
            con.unregister("tb_lote_hex")

        print("\n🧩 Unindo os lotes processados...")
        df_completo = pl.concat(resultados)

        print("🔠 Formatando Textos (MAIÚSCULO E SEM ACENTO)...")
        cols_texto = ["cidade_nome", "regiao_imediata", "regiao_intermediaria", "logradouro", "bairro"]
        
        df_final = df_completo.with_columns([
            pl.col(c).map_elements(remover_acentos, return_dtype=pl.String) for c in cols_texto
        ]).with_columns([
            pl.col("id_h3_h9").map_elements(h3.string_to_int, return_dtype=pl.UInt64).alias("id_h3_int"),
            pl.col("cidade_nome").cast(pl.Categorical)
        ]).drop("id_h3_h9")

        print(f"📊 Preview Final:\n{df_final.head(5)}")

        print("💾 Gravando Malha Mestra no R2 (Parquet ZSTD)...")
        buffer = io.BytesIO()
        df_final.write_parquet(buffer, compression="zstd", compression_level=3)
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, "ouro/malha_mestra_consolidada_2025.parquet")
        
        print("✅ Malha Mestra consolidada com sucesso e sem estourar a memória!")

    except Exception:
        print("\n🚨 ERRO DETALHADO NO SCRIPT:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    build_malha_mestra()
