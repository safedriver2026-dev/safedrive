import os, duckdb, polars as pl, sys, traceback

def auditoria_detalhada_shp():
    print("🕵️ --- INICIANDO INSPEÇÃO PROFUNDA DE METADADOS (SHP) ---")
    PASTA_SHP = 'shp_original'
    ARQUIVO_SHP = f'{PASTA_DADOS}/SP_setores_CD2022.shp' if 'PASTA_DADOS' in locals() else f'{PASTA_SHP}/SP_setores_CD2022.shp'

    try:
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # 1. INSPEÇÃO TÉCNICA DO SHAPEFILE
        print(f"\n📂 [CAMADA: SHAPEFILE ORIGINAL]")
        con.execute(f"CREATE VIEW v_shp AS SELECT * FROM ST_Read('{ARQUIVO_SHP}')")
        
        # Extrair Esquema (DNA das colunas)
        esquema_shp = con.execute("DESCRIBE v_shp").pl()
        print("📋 Esquema de Colunas (Tipos de Dados):")
        print(esquema_shp)

        # Amostra de Dados (Ver como o ID está escrito)
        print("\n👀 Amostra dos primeiros 5 registros do SHP:")
        amostra_shp = con.execute("SELECT * FROM v_shp LIMIT 5").pl()
        print(amostra_shp)

        # 2. INSPEÇÃO DA MALHA MESTRA (PARQUET)
        print(f"\n📂 [CAMADA: MALHA MESTRA PARQUET]")
        df_p = pl.read_parquet('malha_mestra.parquet')
        
        print("📋 Esquema de Colunas (Parquet):")
        print(df_p.schema)

        print("\n👀 Amostra dos registros do Parquet (setor_id):")
        # Mostramos os que NÃO são nulos (se houver) ou apenas os primeiros
        print(df_p.select(["id_h3_int", "setor_id", "bairro"]).head(5))

        # 3. ANÁLISE DE CAUSA RAIZ
        print("\n🧠 [ANÁLISE DE CAUSA RAIZ]")
        
        # Teste 1: Contagem de caracteres no SHP
        # Geralmente CD_SETOR do IBGE tem 15 caracteres
        tamanho_id_shp = con.execute("""
            SELECT length(CD_SETOR::VARCHAR) as len, COUNT(*) 
            FROM v_shp 
            GROUP BY len
        """).pl()
        print("📏 Tamanho dos IDs no SHP (esperado: 15):")
        print(tamanho_id_shp)

        # Teste 2: Verificar se existem zeros à esquerda que sumiram
        zeros_esquerda = con.execute("""
            SELECT COUNT(*) 
            FROM v_shp 
            WHERE CD_SETOR::VARCHAR LIKE '0%'
        """).fetchone()[0]
        print(f"🔢 IDs que começam com zero no SHP: {zeros_esquerda}")

        # Teste 3: Por que o Parquet está 100% nulo?
        # Vamos ver se a coluna existe mas tem outro nome
        print(f"❓ Colunas disponíveis no Parquet: {df_p.columns}")

        print("\n--- FIM DA INSPEÇÃO ---")
        
        # Não deletamos nada agora para você poder ler o log
        sys.exit(1) 

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    auditoria_detalhada_shp()
