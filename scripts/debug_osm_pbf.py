from pyrosm import OSM
import pandas as pd
import pathlib

# Configuração de caminhos
pasta_base = pathlib.Path(r"C:\Users\Lucas Pereira\Downloads\vscode_malha")
pbf_path = pasta_base / "sudeste-260417.osm.pbf"

def processar_e_descobrir():
    print(f"🚀 Iniciando a análise do OSM: {pbf_path.name}")
    
    # 1. Inicializa o leitor
    osm = OSM(str(pbf_path))
    
    # --- CAMADA 1: PONTOS DE INTERESSE (POIs) ---
    print("\n🔹 Extraindo Pontos de Interesse (Hospitais, Bares, Delegacias)...")
    pois = osm.get_pois()
    if pois is not None:
        # Convertendo geometria para texto para salvar no Parquet padrão
        pois["wkt_geometry"] = pois.geometry.apply(lambda x: x.wkt if x else None)
        df_pois = pd.DataFrame(pois.drop(columns=['geometry']))
        
        # Salvando
        df_pois.to_parquet(pasta_base / "osm_pois_silver.parquet", compression='snappy')
        print(f"✅ POIs salvos! Colunas encontradas: {list(df_pois.columns[:10])}")
        print(f"🔍 Dicionário de Amostra (Top 5 tipos):\n{df_pois['amenity'].value_counts().head(5)}")

    # --- CAMADA 2: USO DO SOLO (LANDUSE) ---
    # Essencial para saber se a área é Industrial, Residencial ou Parque
    print("\n🔹 Extraindo Uso do Solo (Áreas Verdes, Industriais)...")
    landuse = osm.get_landuse()
    if landuse is not None:
        landuse["wkt_geometry"] = landuse.geometry.apply(lambda x: x.wkt if x else None)
        df_landuse = pd.DataFrame(landuse.drop(columns=['geometry']))
        df_landuse.to_parquet(pasta_base / "osm_landuse_silver.parquet", compression='snappy')
        print(f"✅ Uso do Solo salvo! Tipos detectados:\n{df_landuse['landuse'].value_counts().head(5)}")

    # --- CAMADA 3: MALHA VIÁRIA (ROAD NETWORK) ---
    # Para saber o tipo de via (Rodovia vs Rua de Bairro)
    print("\n🔹 Extraindo Malha Viária (Hierarquia de Ruas)...")
    roads = osm.get_network(network_type="driving")
    if roads is not None:
        roads["wkt_geometry"] = roads.geometry.apply(lambda x: x.wkt if x else None)
        df_roads = pd.DataFrame(roads.drop(columns=['geometry']))
        df_roads.to_parquet(pasta_base / "osm_roads_silver.parquet", compression='snappy')
        print(f"✅ Rodovias salvas! Hierarquias:\n{df_roads['highway'].value_counts().head(5)}")

if __name__ == "__main__":
    processar_e_descobrir()
