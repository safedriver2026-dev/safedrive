import os, h3, polars as pl, duckdb, boto3, io, glob
from botocore.config import Config

def processar_faces_ibge():
    R2_CONFIG = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")

    try:
        print("☁️ Carregando Malha Base H3 do R2...")
        r2 = boto3.client("s3", **R2_CONFIG)
        obj = r2.get_object(Bucket=BUCKET, Key="referencia/dim_hex_sp.parquet")
        df_hex = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        # Encontra os arquivos JSON extraídos (o IBGE pode dividir em vários)
        arquivos_json = glob.glob("faces_data/*.json")
        print(f"📂 Arquivos encontrados para processar: {len(arquivos_json)}")

        for arq in arquivos_json:
            print(f"⚙️ Processando: {arq}")
            
            # O DuckDB lê o JSON das Faces e já faz o Join com nossos hexágonos
            # Mapeamento provável das Faces 2022:
            # NM_LOGRAD (Rua), NM_BAIRRO (Bairro), NM_MUN (Cidade)
            df_faces_h3 = con.execute(f"""
                SELECT 
                    h.id_h3_h9,
                    f.properties->>'NM_LOGRAD' as nome_rua,
                    f.properties->>'NM_BAIRRO' as nome_bairro,
                    f.properties->>'NM_MUN' as nome_cidade,
                    f.properties->>'TIPO_LOGRA' as tipo_via
                FROM df_hex h
                JOIN ST_Read('{arq}') f 
                ON ST_Within(ST_Point(h.lon, h.lat), f.geom)
            """).pl()

            # Aqui geramos as 3 dimensões separadas como você pediu
            print("💾 Separando e salvando dimensões...")
            
            for dim_name, cols in {
                "dim_cidade": ["id_h3_h9", "nome_cidade"],
                "dim_bairro": ["id_h3_h9", "nome_bairro"],
                "dim_logradouro": ["id_h3_h9", "nome_rua", "tipo_via"]
            }.items():
                dim_df = df_faces_h3.select(cols).unique()
                
                buffer = io.BytesIO()
                dim_df.write_parquet(buffer)
                buffer.seek(0)
                r2.upload_fileobj(buffer, BUCKET, f"referencia/{dim_name}.parquet")

        print("🏁 Gênesis das Faces concluído com sucesso!")

    except Exception as e:
        print(f"🚨 Erro no processamento das Faces: {e}")
        raise e

if __name__ == "__main__":
    processar_faces_ibge()
