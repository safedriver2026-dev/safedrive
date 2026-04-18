import os, h3, polars as pl, duckdb, boto3, io, glob, unicodedata
from botocore.config import Config

def remover_acentos(texto):
    if texto is None: return None
    # Normaliza para decompor caracteres acentuados e remove os marks
    nfkd_form = unicodedata.normalize('NFKD', texto)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper()

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
        print("⬢ Lendo Malha Base H3...")
        r2 = boto3.client("s3", **R2_CONF)
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        # Caminhos dos arquivos extraídos pelo workflow
        path_mun = "dados_ibge/municipios/SP_Municipios_2025.shp"
        path_rgi = "dados_ibge/imediata/SP_RG_Imediatas_2025.shp"
        path_rint = "dados_ibge/intermediaria/SP_RG_Intermediarias_2025.shp"
        path_faces = glob.glob("dados_ibge/faces/*.json")[0]

        print("⚔️ Cruzamento Espacial Múltiplo...")
        df_raw = con.execute(f"""
            SELECT 
                h.id_h3_h9,
                CAST(h.lat AS FLOAT) as latitude,
                CAST(h.lon AS FLOAT) as longitude,
                CAST(m.CD_MUN AS UINT32) as cidade_id,
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

        print("🔠 Padronizando textos (MAIÚSCULO E SEM ACENTO)...")
        # Colunas que precisam de limpeza
        cols_texto = ["cidade_nome", "regiao_imediata", "regiao_intermediaria", "logradouro", "bairro"]
        
        # Aplicando a limpeza e convertendo H3 para Inteiro (Baixa Latência)
        df_final = df_raw.with_columns([
            pl.col(c).map_elements(remover_acentos, return_dtype=pl.String) for c in cols_texto
        ]).with_columns([
            pl.col("id_h3_h9").map_elements(h3.string_to_int, return_dtype=pl.UInt64).alias("id_h3_int"),
            pl.col("cidade_nome").cast(pl.Categorical)
        ]).drop("id_h3_h9")

        print("💾 Gravando Malha Mestra Consolidada (ZSTD)...")
        buffer = io.BytesIO()
        df_final.write_parquet(buffer, compression="zstd", compression_level=3)
        buffer.seek(0)
        
        r2.upload_fileobj(buffer, BUCKET, "ouro/malha_mestra_consolidada_2025.parquet")
        print(f"✅ Sucesso! Malha criada com {len(df_final)} hexágonos.")

    except Exception as e:
        print(f"🚨 Erro no Script: {e}")
        raise e

if __name__ == "__main__":
    build_malha_mestra()
