import os, polars as pl, duckdb, boto3, io, glob, unicodedata, sys, traceback, requests, zipfile
from botocore.config import Config

def remover_acentos(texto):
    if texto is None or texto == "" or str(texto).upper() == "NAN": 
        return "NAO MAPEADO"
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def get_dicionario_bairros():
    print("🌐 Buscando dicionario de bairros no IBGE...")
    url = "https://ftp.ibge.gov.br/Censos/Censo_Demografico_2022/Malha_de_Setores_Censitarios/Cadastro_de_Setores_Censitarios/SP_Cadastro_de_Setores_Censitarios_2022.zip"
    try:
        r = requests.get(url, verify=False)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            arq = [f for f in z.namelist() if f.endswith('.txt') or f.endswith('.csv')][0]
            with z.open(arq) as f:
                return pl.read_csv(f, separator=";", infer_schema_length=10000, ignore_errors=True).select([
                    pl.col("CD_SETOR").cast(pl.Utf8).alias("setor_id"),
                    pl.col("NM_BAIRRO").alias("bairro_oficial")
                ]).unique(subset=["setor_id"])
    except:
        return pl.DataFrame({"setor_id": [], "bairro_oficial": []})

def build_malha_mestra():
    R2_CONF = {"endpoint_url": os.getenv("R2_ENDPOINT_URL"), "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(), 
               "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(), "config": Config(region_name="auto")}
    BUCKET = os.getenv("R2_BUCKET_NAME")

    # 1. Preparar dicionario de bairros
    df_bairros = get_dicionario_bairros()

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial; PRAGMA memory_limit='6GB';")

    try:
        def find_file(pattern):
            f = glob.glob(pattern, recursive=True)
            return f[0] if f else sys.exit(1)

        paths = {"mun": find_file("dados_ibge/municipios/**/*.shp"), "faces": find_file("dados_ibge/faces/**/*.json")}
        con.execute(f"CREATE TABLE tb_mun AS SELECT * FROM ST_Read('{paths['mun']}')")
        con.execute(f"CREATE TABLE tb_faces AS SELECT * FROM ST_Read('{paths['faces']}')")

        r2 = boto3.client("s3", **R2_CONF)
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        lote_tamanho, resultados = 50000, []
        print(f"\n⚔️ Iniciando Processamento Enriquecido...")
        
        for i in range(0, len(df_hex), lote_tamanho):
            print(f"🔄 Lote: {i}...")
            df_lote = df_hex.slice(i, lote_tamanho)
            con.register("tb_lote_hex", df_lote.to_arrow())
            
            lote_raw = con.execute("""
                SELECT 
                    h.id_h3_h9, m.NM_MUN as cidade,
                    CONCAT_WS(' ', f.NM_TIP_LOG, f.NM_TIT_LOG, f.NM_LOG) as logradouro,
                    CAST(f.CD_SETOR AS VARCHAR) as setor_id
                FROM tb_lote_hex h
                LEFT JOIN tb_mun m ON ST_Within(ST_Point(h.lon, h.lat), m.geom)
                LEFT JOIN tb_faces f ON ST_DWithin(ST_Point(h.lon, h.lat), f.geom, 0.0001)
                QUALIFY ROW_NUMBER() OVER(PARTITION BY h.id_h3_h9 ORDER BY ST_Distance(ST_Point(h.lon, h.lat), f.geom) ASC) = 1
            """).pl()
            resultados.append(lote_raw)
            con.unregister("tb_lote_hex")

        print("\n🧩 Consolidando e Fazendo Join com Dicionario de Bairros...")
        df_completo = pl.concat(resultados)
        
        # O Join Magico: Trazendo o nome do bairro pelo setor_id
        df_final = df_completo.join(df_bairros, on="setor_id", how="left")
        
        df_final = df_final.with_columns([
            pl.col("cidade").map_elements(remover_acentos, return_dtype=pl.String),
            pl.col("logradouro").map_elements(remover_acentos, return_dtype=pl.String),
            pl.col("bairro_oficial").fill_null("NAO MAPEADO").map_elements(remover_acentos, return_dtype=pl.String).alias("bairro"),
            pl.col("id_h3_h9").map_elements(lambda x: int(x, 16), return_dtype=pl.UInt64).alias("id_h3_int")
        ]).drop(["id_h3_h9", "bairro_oficial"])

        print("💾 Gravando Malha Mestra Ouro no R2...")
        buffer = io.BytesIO()
        df_final.write_parquet(buffer, compression="zstd")
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, "ouro/malha_mestra_consolidada_2025.parquet")
        print("✅ Malha Mestra Enriquecida concluida!")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    build_malha_mestra()
