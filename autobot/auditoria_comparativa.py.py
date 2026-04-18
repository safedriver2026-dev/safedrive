import os, duckdb, polars as pl, sys, traceback

def auditoria_comparativa():
    print("🕵️ --- INICIANDO AUDITORIA DE RECONCILIAÇÃO (SHP vs PARQUET) ---")
    
    try:
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial;")

        # 1. Carregar a fonte original (SHP)
        print("[LOG] Lendo Ground Truth (Shapefile IBGE)...")
        # Ajuste o nome do arquivo .shp conforme o que está dentro do seu zip
        con.execute("CREATE VIEW shp_source AS SELECT * FROM ST_Read('shp_original/*.shp')")
        
        # 2. Carregar a Malha Mestra gerada
        print("[LOG] Lendo Malha Mestra (Parquet)...")
        df_parquet = pl.read_parquet("malha_mestra.parquet")
        con.register("parquet_data", df_parquet)

        # --- TESTE 1: DIFERENÇA DE CONTAGEM ---
        count_shp = con.execute("SELECT COUNT(*) FROM shp_source").fetchone()[0]
        count_parquet = len(df_parquet)
        diff_perc = abs(count_shp - count_parquet) / count_shp * 100
        
        print(f"\n📊 [ESTATÍSTICA DE COBERTURA]")
        print(f"   - Registros no SHP Original: {count_shp}")
        print(f"   - Registros na Malha Mestra: {count_parquet}")
        print(f"   - Diferença: {diff_perc:.2f}%")

        # --- TESTE 2: DIAGNÓSTICO DO SETOR_ID (POR QUE ESTÁ NULO?) ---
        print("\n🔍 [DIAGNÓSTICO DE JUNÇÃO]")
        # Verifica se o problema é o tipo de dado (String vs Int)
        col_type_shp = con.execute("DESCRIBE SELECT CD_SETOR FROM shp_source").fetchall()
        print(f"   - Tipo da coluna CD_SETOR no SHP: {col_type_shp[0][1]}")
        
        # Tenta um join de amostra para ver se as chaves batem
        match_test = con.execute("""
            SELECT COUNT(*) 
            FROM shp_source s
            JOIN parquet_data p ON s.CD_SETOR::VARCHAR = p.setor_id::VARCHAR
            WHERE p.setor_id IS NOT NULL
        """).fetchone()[0]
        
        if match_test == 0:
            print("   🚨 ERRO CRÍTICO: Zero matches entre SHP e Parquet via ID.")
            print("   👉 Provável causa: Formatação dos IDs (zeros à esquerda perdidos ou tipos diferentes).")
        else:
            print(f"   ✅ Sucesso de Join: {match_test} registros batem corretamente.")

        # --- TESTE 3: ANÁLISE DE VAZIOS (NULLS) ---
        nulls = df_parquet.select(pl.col("setor_id").null_count()).item()
        if nulls > 0:
            print(f"\n❌ [INTEGRIDADE] A Malha Mestra possui {nulls} registros sem setor_id.")
            # Identifica se são regiões específicas
            regioes_afetadas = df_parquet.filter(pl.col("setor_id").is_null())\
                                        .group_by("cidade_nome").count()\
                                        .sort("count", descending=True).head(5)
            print("   📍 Top 5 cidades com mais falhas de ID:")
            print(regioes_afetadas)

        print("\n--- AUDITORIA FINALIZADA ---")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    auditoria_comparativa()
