import duckdb
import os

def criar_config_osm():
    """Cria um arquivo osmconf.ini básico para o GDAL não reclamar"""
    config_content = """
[points]
n_all_tags=yes
[lines]
n_all_tags=yes
[multipolygons]
n_all_tags=yes
[multilinestrings]
n_all_tags=yes
[other_relations]
n_all_tags=yes
"""
    with open("osmconf.ini", "w") as f:
        f.write(config_content)
    # Define a variável de ambiente para o motor espacial encontrar o arquivo
    os.environ["OSM_CONFIG_FILE"] = os.path.abspath("osmconf.ini")
    print("✅ Arquivo de configuração osmconf.ini gerado com sucesso.")

def investigar():
    print("🕵️ Iniciando varredura total de metadados no PBF...")
    
    # Prepara o terreno antes de ligar o DuckDB
    criar_config_osm()
    
    con = duckdb.connect()
    con.execute("SET extension_directory='./duckdb_ext';")
    con.execute("INSTALL spatial; LOAD spatial;")
    
    pbf = "data.osm.pbf"

    # 1. EXPLOSÃO DE CHAVES (O seu Dicionário Real)
    print("\n📚 [DICIONÁRIO] Contagem de TODAS as Chaves (Tags) presentes no Sudeste:")
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
        LIMIT 40
    """
    try:
        df_keys = con.execute(query_keys).pl()
        print(df_keys)
    except Exception as e:
        print(f"❌ Erro na varredura de chaves: {e}")

    # 2. RADIOGRAFIA DA INFRAESTRUTURA (Amenity)
    print("\n🏥 [SERVIÇOS] O que foi mapeado como 'amenity':")
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
    try:
        df_amenity = con.execute(query_amenity).pl()
        print(df_amenity)
    except Exception as e:
        print(f"❌ Erro na varredura de serviços: {e}")

if __name__ == "__main__":
    investigar()
