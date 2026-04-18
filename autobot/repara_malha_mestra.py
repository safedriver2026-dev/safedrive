import os, duckdb, polars as pl, boto3, io, sys, traceback
from botocore.config import Config

def repara_malha_mestra():
    print("🚀 --- INICIANDO REPARAÇÃO DA MALHA MESTRA ---")
    
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL").strip(),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME").strip()
    
    try:
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # 1. Carregar o Ground Truth (Shapefile do IBGE)
        # O DuckDB já lê o .prj e entende que é EPSG:4674
        print("[LOG] Carregando setores do IBGE...")
        con.execute("CREATE VIEW v_setores AS SELECT CD_SETOR, geom FROM ST_Read('shp_original/SP_setores_CD2022.shp')")

        # 2. Carregar a sua Malha Mestra com erro (pegando as coordenadas brutas)
        print("[LOG] Carregando malha atual para reparo...")
        df_erro = pl.read_parquet('malha_mestra.parquet')
        con.register("v_malha_erro", df_erro)

        # 3. O JOIN ESPACIAL CORRIGIDO
        # Criamos o ponto a partir de lat/lon e garantimos que o DuckDB o veja no mesmo sistema do IBGE
        print("[LOG] Executando Join Espacial de alto desempenho...")
        df_corrigido = con.execute("""
            SELECT 
                m.* EXCLUDE (setor_id), -- Pegamos tudo da malha original, menos o setor_id nulo
                s.CD_SETOR as setor_id   -- Injetamos o ID correto vindo do Shapefile
            FROM v_malha_erro m
            LEFT JOIN v_setores s ON ST_Within(
                ST_Point(m.longitude, m.latitude), 
                s.geom
            )
        """).pl()

        # 4. VALIDAÇÃO IMEDIATA
        nulos = df_corrigido.select(pl.col("setor_id").is_null().sum())[0,0]
        sucesso = len(df_corrigido) - nulos
        print(f"📊 Resultado do Reparo: {sucesso} hexágonos vinculados com sucesso!")
        print(f"⚠️ Restaram {nulos} nulos (provavelmente áreas de mar ou fora do estado).")

        # 5. UPLOAD DA MALHA CORRIGIDA
        print("[LOG] Enviando malha corrigida para o R2...")
        buf = io.BytesIO()
        df_corrigido.write_parquet(buf, compression="zstd")
        buf.seek(0)
        
        r2 = boto3.client("s3", **R2_CONF)
        r2.upload_fileobj(buf, BUCKET, "malha geografica/malha_mestra_consolidada_2025.parquet")

        print("✅ REPARAÇÃO CONCLUÍDA!")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    repara_malha_mestra()
