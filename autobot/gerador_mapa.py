import polars as pl
import folium
from folium.plugins import Search, Fullscreen, LocateControl
import h3
import io
import os
import boto3
from botocore.config import Config
from branca.colormap import LinearColormap

class GeradorMapaRisco:
    """
    Motor de Visualização Geospacial do SafeDriver.
    Transforma predições analíticas em uma interface SIG (Sistema de Informação Geográfica) 
    interativa hospedada no GitHub Pages.
    """
    def __init__(self):
        # Configuração de acesso ao Data Lake (R2)
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )

    def _obter_dados_ouro(self):
        """Busca o arquivo de predição mais recente na Camada Prata/Ouro."""
        ano_atual = 2026 # Contexto do projeto
        path = f"safedriver/datalake/prata/ssp_consolidada_{ano_atual}.parquet"
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path)
            return pl.read_parquet(io.BytesIO(resp['Body'].read())).to_pandas()
        except Exception as e:
            print(f"Erro ao carregar dados: {e}")
            return None

    def h3_to_geojson(self, h3_index):
        """Converte índice H3 para coordenadas de polígono."""
        boundary = h3.cell_to_boundary(h3_index)
        return [[lat, lng] for lat, lng in boundary]

    def construir(self):
        df = self._obter_dados_ouro()
        if df is None: return

        # 1. Base do Mapa com Estilo Dark (Melhor contraste para Heatmaps)
        mapa = folium.Map(
            location=[-23.5505, -46.6333], 
            zoom_start=11, 
            tiles='cartodbdark_matter', # Estilo profissional 'Dark'
            control_scale=True
        )

        # 2. Paleta de Cores Semântica (Seguro -> Crítico)
        colormap = LinearColormap(
            ['#00ff00', '#ffff00', '#ff0000'], # Verde, Amarelo, Vermelho
            vmin=0, vmax=10,
            caption='Intensidade de Risco SafeDriver'
        )
        colormap.add_to(mapa)

        # 3. Definição de Camadas por Persona
        # Justificativa Técnica: Permite que o usuário final filtre o risco por perfil.
        fg_motorista = folium.FeatureGroup(name="🚗 Risco: Motoristas", show=True)
        fg_pedestre = folium.FeatureGroup(name="🚶 Risco: Pedestres", show=False)
        fg_moto = folium.FeatureGroup(name="🏍️ Risco: Motociclistas", show=False)

        for _, row in df.iterrows():
            try:
                poligono = self.h3_to_geojson(row['H3_INDEX'])
                
                # Tooltip Rico (HTML/CSS)
                tooltip_html = f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4 style="margin-bottom:5px;">Hexágono {row['H3_INDEX']}</h4>
                    <hr>
                    <b>🚗 Motorista:</b> {row['TOTAL_CRIMES_MOTORISTA']:.1f}<br>
                    <b>🚶 Pedestre:</b> {row['TOTAL_CRIMES_PEDESTRE']:.1f}<br>
                    <b>🏍️ Motociclista:</b> {row['TOTAL_CRIMES_MOTOCICLISTA']:.1f}<br>
                    <hr>
                    <b>📈 Delta (Tendência):</b> {row.get('DELTA_MOTORISTA', 0):+d}
                </div>
                """

                # Adicionando polígonos às respectivas camadas
                folium.Polygon(
                    locations=poligono,
                    fill=True,
                    fill_color=colormap(row['TOTAL_CRIMES_MOTORISTA']),
                    color='white', weight=0.5, fill_opacity=0.5,
                    tooltip=tooltip_html
                ).add_to(fg_motorista)

                folium.Polygon(
                    locations=poligono,
                    fill=True,
                    fill_color=colormap(row['TOTAL_CRIMES_PEDESTRE']),
                    color='white', weight=0.5, fill_opacity=0.5,
                    tooltip=tooltip_html
                ).add_to(fg_pedestre)

            except: continue

        # 4. Plugins e Interatividade Avançada
        fg_motorista.add_to(mapa)
        fg_pedestre.add_to(mapa)
        fg_moto.add_to(mapa)
        
        folium.LayerControl(collapsed=False).add_to(mapa) # Seletor de camadas
        Fullscreen().add_to(mapa) # Botão de tela cheia
        LocateControl().add_to(mapa) # Geolocalização do usuário

        # 5. Exportação para Documentação GitHub Pages
        os.makedirs('docs', exist_ok=True)
        mapa.save('docs/index.html')
        print("🚀 Dashboard Geográfico atualizado com sucesso!")

if __name__ == "__main__":
    GeradorMapaRisco().construir()
