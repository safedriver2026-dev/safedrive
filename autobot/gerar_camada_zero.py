import os
import h3
import requests
import polars as pl
import boto3
from botocore.config import Config
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeradorCamadaZero:
    def __init__(self):
        self.bucket = os.getenv("R2_BUCKET_NAME")
        self.r2 = boto3.client(
            "s3",
            endpoint_url=os.getenv("R2_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            config=Config(region_name="auto")
        )

    def executar(self):
        logger.info("🗺️ Obtendo fronteiras de SP via API do IBGE...")
        url = "https://servicodados.ibge.gov.br/api/v3/malhas/estados/35?formato=application/vnd.geo+json"
        geo_sp = requests.get(url).json()
        

        features = geo_sp['features'][0]['geometry']['coordinates']
        
     
        poly_coords = features[0][0] if isinstance(features[0][0][0], list) else features[0]
        polygon_h3 = {"type": "Polygon", "coordinates": [[ [c[1], c[0]] for c in poly_coords ]]}

        logger.info("⬢ Gerando malha H9 (Isto pode demorar alguns segundos)...")
        hexs_h9 = list(h3.polyfill(polygon_h3, 9))
        
        logger.info(f"✅ {len(hexs_h9)} hexágonos gerados. Criando mapeamento hierárquico...")
        
        df_mestre = pl.DataFrame({"H3_INDEX_H9": hexs_h9}).with_columns(
            pl.col("H3_INDEX_H9").map_elements(lambda x: h3.cell_to_parent(x, 8)).alias("H3_INDEX_H8")
        )

        # Upload direto para o R2
        buffer = io.BytesIO()
        df_mestre.write_parquet(buffer)
        buffer.seek(0)
        
        logger.info("🚀 Enviando Camada Zero para o R2...")
        self.r2.upload_fileobj(buffer, self.bucket, "referencia/grid_sao_paulo_mestre.parquet")
        logger.info("✨ Processo concluído com sucesso!")

if __name__ == "__main__":
    GeradorCamadaZero().executar()
