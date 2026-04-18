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

    if not os.path.exists('duckdb_temp'): os.makedirs('duckdb_temp')
    con = duckdb.connect()
    con.execute("PRAGMA temp_directory='duckdb_temp'; PRAGMA memory_limit='6GB'; PRAGMA threads=2;")
    con.execute("INSTALL spatial; LOAD spatial;")

    try:
        def find_file(pattern):
            f = glob.glob(pattern, recursive=True)
            if not f: sys.exit(1)
            return f[0]

        paths = {
            "mun": find_file("dados_ibge/municipios/**/*.shp"),
            "rgi": find_file("dados_ibge/imediata/**/*.shp"),
            "rint": find_file("dados_ibge/intermediaria/**/*.shp"),
            "faces": find_file("dados_ibge/faces/**/*.json")
        }

        con.execute(f"CREATE TABLE tb_mun AS SELECT * FROM ST_Read('{paths['mun']}')")
        con.execute(f"CREATE TABLE tb_rgi AS SELECT * FROM ST_Read('{paths['rgi']}')")
        con.execute(f"CREATE TABLE tb_rint AS SELECT * FROM ST_Read('{paths['rint']}')")
        con.execute(f"CREATE TABLE tb_faces AS SELECT * FROM ST_Read('{paths['faces']}')")

        r2 = boto3.client("s3", **R2_CONF)
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        lote_tamanho, resultados = 50000, []
        for i in range(0, len(df_hex), lote_tamanho):
            df_lote = df_hex.slice(i, lote_tamanho)
            con.register("tb_lote_hex", df_lote.to_arrow())
            
            con.execute("""
                CREATE TEMP TABLE lote_admin AS 
                SELECT h.id_h3_h9, CAST(h.lat AS FLOAT) as lat, CAST(h.lon AS FLOAT) as lon,
                       m.NM_MUN as cidade, rgi.NM_RGI as rgi, rint.NM_RGINT as rint
                FROM tb_lote_hex h
                LEFT JOIN tb_mun m ON ST_Within(ST_Point(h.lon, h.lat), m.geom)
                LEFT JOIN tb_rgi rgi ON ST_Within(ST_Point(h.lon, h.lat), rgi.geom)
                LEFT JOIN tb_rint rint ON ST_Within(ST_Point(h.lon, h.lat), rint.geom)
            """)
            
            lote_raw = con.execute("""
                SELECT a.*, CONCAT_WS(' ', f.NM_TIP_LOG, f.NM_TIT_LOG, f.NM_LOG) as logradouro,
                       'NAO MAPEADO' as bairro, CAST(f.CD_SETOR AS VARCHAR) as setor_id
                FROM lote_admin a
                LEFT JOIN tb_faces f ON ST_DWithin(ST_Point(a.lon, a.lat), f.geom, 0.0001)
                QUALIFY ROW_NUMBER() OVER(PARTITION BY a.id_h3_h9 ORDER BY ST_Distance(ST_Point(a.lon, a.lat), f.geom) ASC) = 1
            """).pl()
            resultados.append(lote_raw)
            con.execute("DROP TABLE lote_admin"); con.unregister("tb_lote_hex")

        df_final = pl.concat(resultados).with_columns([
            pl.col(c).map_elements(remover_acentos, return_dtype=pl.String) for c in ["cidade", "rgi", "rint", "logradouro", "bairro"]
        ]).with_columns([
            # AQUI: string_to_int para a versão 3.7.6
            pl.col("id_h3_h9").map_elements(h3.string_to_int, return_dtype=pl.UInt64).alias("id_h3_int")
        ]).drop("id_h3_h9")

        buffer = io.BytesIO()
        df_final.write_parquet(buffer, compression="zstd")
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, "ouro/malha_mestra_consolidada_2025.parquet")
        print("✅ Malha Mestra concluída!")

    except Exception:
        traceback.print_exc(); sys.exit(1)

if __name__ == "__main__": build_malha_mestra()
