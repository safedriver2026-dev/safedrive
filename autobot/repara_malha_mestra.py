import os, duckdb, polars as pl, boto3, io, sys, traceback
from botocore.config import Config

def executar_reparo():
    print("🚀 --- INICIANDO RECONSTRUÇÃO DA MALHA MESTRA ---")
    
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL").strip(),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME").strip()
    
    # Nomes dos arquivos alinhados com o YAML
    ARQUIVO_ENTRADA = 'malha_mestra_erro.parquet'
    PASTA_SHP = 'shp_original'
    ARQUIVO_SHP = f'{PASTA_SHP}/SP_setores_CD2022.shp'

    try:
        # Verificação de segurança: O arquivo existe?
        if not os.path.exists(ARQUIVO_ENTRADA):
            print(f"❌ ERRO: O arquivo {ARQUIVO_ENTRADA} não foi encontrado no Runner.")
            sys.exit(1)

        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # 1. Carregar Shapefile do IBGE
        print("[LOG] Lendo polígonos do IBGE (SIRGAS 2000)...")
        con.execute(f"""
            CREATE VIEW v_setores AS 
            SELECT CAST(CD_SETOR AS VARCHAR) as cd_setor_limpo, geom 
            FROM ST_Read('{ARQUIVO_SHP}')
        """)

        # 2. Carregar a Malha Parquet (com o nome correto agora)
        print(f"[LOG] Lendo {ARQUIVO_ENTRADA}...")
        df_erro = pl.read_parquet(ARQUIVO_ENTRADA)
        con.register("v_malha_erro", df_erro)

        # 3. O JOIN ESPACIAL DEFINITIVO (Com correção de Projeção)
        print("[LOG] Executando Join Espacial (ST_Within) com ST_Transform...")
        # Nota: WGS84 (4326) para SIRGAS 2000 (4674)
        df_corrigido = con.execute("""
            SELECT 
                m.* EXCLUDE (setor_id),
                s.cd_setor_limpo as setor_id
            FROM v_malha_erro m
            LEFT JOIN v_setores s ON ST_Within(
                ST_Transform(ST_Point(m.longitude, m.latitude), 'EPSG:4326', 'EPSG:4674'), 
                s.geom
            )
        """).pl()

        # 4. Auditoria de Sucesso
        total = len(df_corrigido)
        vinculados = df_corrigido.select(pl.col("setor_id").is_not_null().sum()).item()
        percentual = (vinculados / total) * 100
        
        print(f"\n📊 --- RELATÓRIO DE REPARAÇÃO ---")
        print(f"✅ Total Processado: {total}")
        print(f"✅ Sucesso no Vínculo: {vinculados} ({percentual:.2f}%)")
        
        # 5. Upload para o R2
        print("\n[LOG] Enviando malha corrigida para o Cloudflare R2...")
        buf = io.BytesIO()
        df_corrigido.write_parquet(buf, compression="zstd")
        buf.seek(0)
        
        r2 = boto3.client("s3", **R2_CONF)
        r2.upload_fileobj(buf, BUCKET, "malha geografica/malha_mestra_consolidada_2025.parquet")
        
        print("🎉 Tabela finalizada e salva no R2!")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    executar_reparo()
