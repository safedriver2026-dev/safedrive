import duckdb
import os
import pathlib

# Criar pasta de relatórios
pathlib.Path("reports").mkdir(exist_ok=True)

def investigar():
    print("🕵️ Analisando a estrutura bruta do PBF...")
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    
    pbf = "data.osm.pbf"
    
    # 1. Extrair TODAS as chaves (Keys) existentes no arquivo
    # Isso é o "Dicionário" que você quer: o que foi mapeado?
    print("📚 Gerando Dicionário de Chaves...")
    query_keys = f"""
        COPY (
            SELECT key, count(*) as ocorrencias
            FROM (
                SELECT unnest(map_keys(tags)) as key 
                FROM ST_Read('{pbf}')
            ) 
            GROUP BY key 
            ORDER BY ocorrencias DESC
        ) TO 'reports/dicionario_chaves.csv' (HEADER, DELIMITER ',');
    """
    con.execute(query_keys)

    # 2. Ver os tipos de "Amenity" (Onde mora o risco do SafeDriver)
    print("🏥 Mapeando Infraestrutura (Amenities)...")
    query_amenities = f"""
        COPY (
            SELECT tags->'amenity' as tipo, count(*) as total
            FROM ST_Read('{pbf}')
            WHERE tags->'amenity' IS NOT NULL
            GROUP BY 1 ORDER BY 2 DESC
        ) TO 'reports/tipos_amenities.csv' (HEADER, DELIMITER ',');
    """
    con.execute(query_amenities)

    print("✅ Relatórios gerados na pasta /reports")

if __name__ == "__main__":
    investigar()
