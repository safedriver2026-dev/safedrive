import duckdb
import os

def investigar():
    print("🕵️ Iniciando varredura total de metadados no PBF...")
    con = duckdb.connect()
    
    # Configuração de ambiente Linux (GHA)
    con.execute("SET extension_directory='./duckdb_ext';")
    con.execute("INSTALL spatial; LOAD spatial;")
    
    pbf = "data.osm.pbf"

    # 1. EXPLOSÃO DE CHAVES (O Dicionário Real)
    # Aqui vamos descobrir cada 'Key' que os usuários do OSM mapearam no Sudeste
    print("\n📚 [DICIONÁRIO] Contagem de TODAS as Chaves (Tags) presentes:")
    query_keys = f"""
        SELECT 
            key, 
            count(*) as ocorrencias
        FROM (
            SELECT unnest(map_keys(tags)) as key 
            FROM ST_Read('{pbf}')
        ) 
        GROUP BY key 
        ORDER BY ocorrencias DESC 
        LIMIT 50
    """
    df_keys = con.execute(query_keys).pl()
    print(df_keys)

    # 2. ANÁLISE DE VALORES (O que tem dentro das chaves principais)
    # Vamos espiar o que as pessoas colocam em 'amenity' (infraestrutura)
    print("\n🏥 [INFRAESTRUTURA] Distribuição da chave 'amenity':")
    query_amenity = f"""
        SELECT 
            tags->'amenity' as tipo, 
            count(*) as total
        FROM ST_Read('{pbf}')
        WHERE tags->'amenity' IS NOT NULL
        GROUP BY 1 
        ORDER BY 2 DESC 
        LIMIT 20
    """
    df_amenity = con.execute(query_amenity).pl()
    print(df_amenity)

if __name__ == "__main__":
    investigar()
