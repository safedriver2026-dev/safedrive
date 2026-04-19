import duckdb
import os

def investigar():
    print("🕵️ Iniciando Diagnóstico de Estrutura (Discovery Mode)...")
    
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    
    pbf = "data.osm.pbf"
    
    if not os.path.exists(pbf):
        print("❌ Arquivo data.osm.pbf não encontrado!")
        return

    # 1. DESCOBRINDO O ESQUEMA REAL (Colunas)
    print("\n🏛️ 1. COLUNAS DETECTADAS NO ARQUIVO (O Dicionário de Colunas):")
    # O DuckDB vai ler o cabeçalho e nos dizer o que ele extraiu como coluna fixa
    try:
        # Consultamos os metadados das colunas
        esquema = con.execute(f"SELECT * FROM ST_Read('{pbf}') LIMIT 0").df()
        colunas = esquema.columns.tolist()
        print(f"✅ O motor detectou as seguintes colunas: {colunas}")
    except Exception as e:
        print(f"❌ Erro ao ler colunas: {e}")

    # 2. INVESTIGANDO O "LIXÃO" (other_tags)
    # No OSM, o que não virou coluna fixa vai para a 'other_tags' como texto
    if 'other_tags' in colunas:
        print("\n📦 2. ESPIANDO O CONTEÚDO DE 'other_tags' (Dados não estruturados):")
        try:
            # Pegamos uma amostra de 10 linhas para você ver o formato do dicionário
            amostra = con.execute(f"SELECT other_tags FROM ST_Read('{pbf}') WHERE other_tags IS NOT NULL LIMIT 10").df()
            print(amostra)
        except Exception as e:
            print(f"❌ Erro ao ler other_tags: {e}")
    else:
        print("\n⚠️ A coluna 'other_tags' não foi encontrada. Verificando o que existe no lugar...")

    # 3. CONTAGEM POR TIPO DE OBJETO
    # Vamos ver se o arquivo é feito de pontos, linhas ou polígonos
    print("\n📐 3. RESUMO DE GEOMETRIAS:")
    try:
        resumo = con.execute(f"SELECT ST_GeometryType(geom) as tipo, count(*) as total FROM ST_Read('{pbf}') GROUP BY 1").df()
        print(resumo)
    except Exception as e:
        # Se a coluna de geometria não se chamar 'geom', o DuckDB Spatial geralmente usa 'wkb_geometry'
        print("Tentando com nome de geometria alternativo...")
        try:
            resumo = con.execute(f"SELECT count(*) as total FROM ST_Read('{pbf}')").df()
            print(f"Total de registros no arquivo: {resumo.iloc[0,0]}")
        except:
            print("Não foi possível contar as geometrias.")

if __name__ == "__main__":
    investigar()
