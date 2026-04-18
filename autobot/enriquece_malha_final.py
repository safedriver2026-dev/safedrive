import os, duckdb, polars as pl, boto3, io, unicodedata, sys, traceback
from botocore.config import Config

def remover_acentos(texto):
    # Se estiver vazio ou nulo, preenche com um valor padrão
    if texto is None or str(texto).upper() in ["NAN", "", "NULL"]: 
        return "NAO MAPEADO"
    # Normaliza a string para remover os acentos
    nfkd_form = unicodedata.normalize('NFKD', str(texto))
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).upper().strip()

def pipeline_migracao_final():
    # Configuração das credenciais do Cloudflare R2
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL"),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME")
    
    # === PARAMETRIZAÇÃO DAS PASTAS E FICHEIROS ===
    # O nome exato dos ficheiros que carregou (sem a extensão)
    PREFIXO_IBGE = "SP_setores_CD2022" 
    
    # Partimos do princípio que carregou os ficheiros do IBGE e a malha na pasta 'ouro'
    PASTA_ORIGEM_R2 = "ouro" 
    CAMINHO_MALHA_ANTIGA = f"{PASTA_ORIGEM_R2}/malha_mestra_consolidada_2025.parquet"
    
    # O novo caminho solicitado para a malha final
    CAMINHO_MALHA_NOVA = "malha geografica/malha_mestra_consolidada_2025.parquet"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        print("🔍 [LOG PROVA] A iniciar o processo de enriquecimento e migração...")

        # 1. Descarregar os ficheiros IBGE do R2 para a máquina temporária do GitHub
        extensoes = ['shp', 'dbf', 'shx', 'prj', 'cpg']
        for ext in extensoes:
            chave_r2 = f"{PASTA_ORIGEM_R2}/{PREFIXO_IBGE}.{ext}"
            ficheiro_local = f"temp_ibge.{ext}"
            print(f"📥 [LOG PROVA] A transferir o ficheiro '{chave_r2}' para leitura...")
            r2.download_file(BUCKET, chave_r2, ficheiro_local)

        # 2. Leitura e extração do DBF com DuckDB
        print("📖 [LOG PROVA] A extrair os atributos dos Setores Censitários de 2022...")
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")
        
        # Lemos o Shapefile, mas o DuckDB é inteligente o suficiente para puxar só os textos do DBF
        df_ibge = con.execute("""
            SELECT 
                CAST(CD_SETOR AS VARCHAR) as setor_id,
                NM_BAIRRO as bairro_raw,
                NM_DIST as distrito,
                NM_SUBDIST as subdistrito
            FROM ST_Read('temp_ibge.shp')
        """).pl().unique(subset=["setor_id"])
        
        print(f"✅ [LOG PROVA] {len(df_ibge)} setores extraídos com sucesso da base do IBGE.")

        # 3. Descarregar a Malha Mestra a ser atualizada
        print(f"📥 [LOG PROVA] A descarregar a malha base do caminho: '{CAMINHO_MALHA_ANTIGA}'...")
        obj_malha = r2.get_object(Bucket=BUCKET, Key=CAMINHO_MALHA_ANTIGA)
        df_malha = pl.read_parquet(io.BytesIO(obj_malha['Body'].read()))
        
        # Remover a coluna 'bairro' antiga, caso esta exista, para evitar conflitos
        if "bairro" in df_malha.columns:
            df_malha = df_malha.drop("bairro")

        # 4. Cruzamento dos Dados (Join)
        print("🧩 [LOG PROVA] A injetar os Bairros, Distritos e Subdistritos na Malha Mestra...")
        df_final = df_malha.join(df_ibge, on="setor_id", how="left")
        
        # Tratamento e higienização dos textos (maiúsculas e sem acentos)
        df_final = df_final.with_columns([
            pl.col("bairro_raw").fill_null("NAO MAPEADO").map_elements(remover_acentos, return_dtype=pl.String).alias("bairro"),
            pl.col("distrito").fill_null("NAO MAPEADO").map_elements(remover_acentos, return_dtype=pl.String),
            pl.col("subdistrito").fill_null("NAO MAPEADO").map_elements(remover_acentos, return_dtype=pl.String)
        ]).drop("bairro_raw")

        # 5. Carregar para o NOVO caminho (malha geografica)
        print(f"💾 [LOG PROVA] A guardar a nova Malha Enriquecida em '{CAMINHO_MALHA_NOVA}'...")
        buffer = io.BytesIO()
        df_final.write_parquet(buffer, compression="zstd")
        buffer.seek(0)
        r2.upload_fileobj(buffer, BUCKET, CAMINHO_MALHA_NOVA)

        # 6. Limpeza e Expurgo (Ficheiros do IBGE + Malha Antiga)
        print("🧹 [LOG PROVA] A iniciar o expurgo dos documentos utilizados na pasta 'ouro'...")
        
        # Apaga a malha antiga
        r2.delete_object(Bucket=BUCKET, Key=CAMINHO_MALHA_ANTIGA)
        print(f"🗑️ [LOG PROVA] Ficheiro eliminado do R2: '{CAMINHO_MALHA_ANTIGA}'")
        
        # Apaga os 5 ficheiros do Shapefile
        for ext in extensoes:
            chave_r2 = f"{PASTA_ORIGEM_R2}/{PREFIXO_IBGE}.{ext}"
            r2.delete_object(Bucket=BUCKET, Key=chave_r2)
            print(f"🗑️ [LOG PROVA] Ficheiro eliminado do R2: '{chave_r2}'")
            
            # Limpeza também na máquina virtual local (GitHub Actions)
            if os.path.exists(f"temp_ibge.{ext}"):
                os.remove(f"temp_ibge.{ext}")

        print("✨ [LOG PROVA] Processo finalizado com excelência! A base geográfica está 100% pronta e o ambiente limpo.")

    except Exception:
        print("\n🚨 [ERRO CRÍTICO] Falha durante a execução:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    pipeline_migracao_final()
