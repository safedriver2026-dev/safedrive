import duckdb
import polars as pl

def debug_pbf():
    print("🚀 Iniciando Diagnóstico Profundo no GitHub Actions...")
    
    con = duckdb.connect()
    # No Linux do GHA, isso instala os drivers necessários automaticamente
    con.execute("INSTALL spatial; LOAD spatial;")
    
    pbf_path = "sudeste.osm.pbf"
    
    print("📊 1. Esquema Geral (Colunas disponíveis):")
    schema = con.execute(f"DESCRIBE SELECT * FROM ST_Read('{pbf_path}') LIMIT 1").pl()
    print(schema)

    print("\n🔍 2. Top 15 Amenidades (O 'Dicionário' do SafeDriver):")
    # Aqui descobrimos o que o arquivo tem de hospitais, bares, polícia, etc.
    amenities = con.execute(f"""
        SELECT 
            tags->'amenity' as categoria, 
            count(*) as total
        FROM ST_Read('{pbf_path}')
        WHERE categoria IS NOT NULL
        GROUP BY 1 
        ORDER BY 2 DESC 
        LIMIT 15
    """).pl()
    print(amenities)

    print("\n🛣️ 3. Malha Viária (Hierarquia de Ruas):")
    roads = con.execute(f"""
        SELECT 
            tags->'highway' as tipo_via, 
            count(*) as total
        FROM ST_Read('{pbf_path}')
        WHERE tipo_via IS NOT NULL
        GROUP BY 1 
        ORDER BY 2 DESC 
        LIMIT 10
    """).pl()
    print(roads)

if __name__ == "__main__":
    debug_pbf()
