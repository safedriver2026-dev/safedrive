import polars as pl
import folium
import h3
import json
import os
from branca.colormap import LinearColormap

def gerar_mapa_calor():
    # 1. Carregar dados da Camada Ouro (ou diretamente da View do BigQuery)
    # Aqui vamos simular pegando do arquivo parquet mais recente na Prata/Ouro
    path_ouro = "safedriver/datalake/prata/ssp_consolidada_2026.parquet"
    df = pl.read_parquet(path_ouro).to_pandas()

    # 2. Configurar o mapa centralizado em São Paulo
    mapa = folium.Map(location=[-23.5505, -46.6333], zoom_start=11, tiles='cartodbpositron')

    # 3. Criar escala de cores (Verde -> Amarelo -> Vermelho)
    colormap = LinearColormap(['green', 'yellow', 'red'], vmin=0, vmax=100)

    # 4. Função para converter H3 em Polígono GeoJSON
    def h3_to_geojson(h3_index):
        geo_boundary = h3.cell_to_boundary(h3_index)
        return [[lat, lng] for lat, lng in geo_boundary]

    # 5. Adicionar Hexágonos ao Mapa
    for _, row in df.iterrows():
        try:
            poligono = h3_to_geojson(row['H3_INDEX'])
            folium.Polygon(
                locations=poligono,
                fill=True,
                fill_color=colormap(row['TOTAL_CRIMES_MOTORISTA'] * 10), # Escala de risco
                color='black',
                weight=1,
                fill_opacity=0.6,
                tooltip=f"H3: {row['H3_INDEX']}<br>Crimes: {row['TOTAL_CRIMES_MOTORISTA']}"
            ).add_to(mapa)
        except: continue

    # 6. Salvar o arquivo final
    os.makedirs('docs', exist_ok=True)
    mapa.save('docs/index.html')
    print("✅ Mapa gerado com sucesso em docs/index.html")

if __name__ == "__main__":
    gerar_mapa_calor()
