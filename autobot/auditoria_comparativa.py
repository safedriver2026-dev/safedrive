import os, duckdb, polars as pl, sys, traceback, shutil

def auditoria_confronto():
    print("🕵️ --- INICIANDO AUDITORIA DE RECONCILIAÇÃO (SHP vs PARQUET) ---")
    PASTA_SHP = 'shp_original'
    ARQUIVO_SHP = f'{PASTA_SHP}/SP_setores_CD2022.shp'
    ARQUIVO_PARQUET = 'malha_mestra.parquet'

    try:
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # 1. VALIDAR LEITURA DO SHAPEFILE (Precisa de todos os arquivos na pasta)
        print("[LOG] Validando integridade do Shapefile original...")
        # ST_Read falha se o .dbf ou .shx estiverem corrompidos
        con.execute(f"CREATE VIEW v_shp AS SELECT * FROM ST_Read('{ARQUIVO_SHP}')")
        
        info_shp = con.execute("SELECT COUNT(*), COUNT(DISTINCT CD_SETOR) FROM v_shp").fetchone()
        total_shp = info_shp[0]
        distintos_shp = info_shp[1]

        # 2. CARREGAR PARQUET
        print("[LOG] Lendo Malha Mestra do Data Lake...")
        df_p = pl.read_parquet(ARQUIVO_PARQUET)
        con.register("v_parquet", df_p)

        # 3. CONFRONTO DE DADOS
        nulls_parquet = df_p.select(pl.col("setor_id").null_count()).item()
        
        print(f"\n📊 --- RESULTADOS DA AUDITORIA ---")
        print(f"✅ Shapefile Original: {total_shp} setores encontrados.")
        print(f"✅ Setores Únicos no SHP: {distintos_shp}")
        print(f"⚠️ Nulos na Malha Mestra (setor_id): {nulls_parquet}")

        # 4. TESTE DE JOIN (A PROVA REAL)
        # Verificamos se o ID do SHP (CD_SETOR) bate com o setor_id do Parquet
        match_count = con.execute("""
            SELECT COUNT(*) 
            FROM v_shp s 
            INNER JOIN v_parquet p ON s.CD_SETOR::VARCHAR = p.setor_id::VARCHAR
        """).fetchone()[0]

        print(f"🔗 Cruzamento de chaves: {match_count} registros encontrados em ambos.")

        # Lógica de Decisão para Limpeza
        if match_count > 0 and nulls_parquet < (len(df_p) * 0.1): # Se bater e tiver menos de 10% de erro
            print("\n✅ AUDITORIA SUCESSO: Os dados foram reconciliados. Iniciando limpeza...")
            # SÓ DELETA SE TUDO ESTIVER OK
            if os.path.exists(PASTA_SHP): shutil.rmtree(PASTA_SHP)
            if os.path.exists(ARQUIVO_PARQUET): os.remove(ARQUIVO_PARQUET)
            print("🧹 Limpeza concluída.")
        else:
            print("\n🚨 FALHA NA RECONCILIAÇÃO!")
            if match_count == 0:
                print("👉 Causa: O setor_id no Parquet não tem correspondência no SHP original. Verifique se houve perda de zeros à esquerda.")
            if nulls_parquet > 0:
                print(f"👉 Causa: Existem {nulls_parquet} registros órfãos na malha.")
            
            print("📂 Arquivos mantidos para debug manual.")
            sys.exit(1)

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    auditoria_confronto()
