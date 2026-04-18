import os
import json
import boto3
import polars as pl
import h3
import io
import logging
from google.cloud import bigquery
from google.oauth2 import service_account
from botocore.config import Config

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class GeradorReferencia:
    def __init__(self):
        # Carregar credenciais a partir do Secret JSON
        self.bq_json = json.loads(os.getenv("BQ_SERVICE_ACCOUNT_JSON"))
        self.creds = service_account.Credentials.from_service_account_info(self.bq_json)
        self.project_id = os.getenv("BQ_PROJECT_ID")
        self.bucket_name = os.getenv("R2_BUCKET_NAME")
        
        # Cliente BigQuery forçado para São Paulo (South America)
        self.bq_client = bigquery.Client(
            credentials=self.creds,
            project=self.project_id,
            location="southamerica-east1"
        )
        
        # Cliente Cloudflare R2
        self.r2_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("R2_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            config=Config(region_name="auto")
        )

    def obter_sql_mestre(self):
        """Retorna a query otimizada com filtros de Bounding Box para SP."""
        return """
        WITH osm_rodovias AS (
            -- Extração de rodovias principais do OSM filtradas por SP
            SELECT geometry 
            FROM `bigquery-public-data.geo_openstreetmap.planet_features`
            WHERE feature_type = 'lines'
              AND EXISTS(SELECT 1 FROM UNNEST(all_tags) t WHERE t.key = 'highway' AND t.value IN ('motorway', 'trunk', 'primary'))
              AND ST_INTERSECTS(geometry, ST_GEOGFROMTEXT('POLYGON((-53.1 -25.3, -44.1 -25.3, -44.1 -19.7, -53.1 -19.7, -53.1 -25.3))'))
        ),
        censo_rais AS (
            -- União do Censo 2022 (População) com RAIS 2021 (Empregos)
            SELECT 
                s.id_municipio, s.id_setor_censitario, s.pessoas, s.area, s.domicilios,
                ST_CENTROID(s.geometria) as centroide,
                r.quantidade_vinculos_ativos as empregos
            FROM `basedosdados.br_ibge_censo_2022.setor_censitario` s
            LEFT JOIN `basedosdados.br_me_rais.estabelecimento` r 
                ON s.id_municipio = r.id_municipio AND r.ano = 2021
            WHERE s.id_uf = '35'
        )
        SELECT 
            c.id_municipio, c.pessoas, c.area, c.domicilios, c.empregos,
            ST_X(c.centroide) as lon, ST_Y(c.centroide) as lat,
            -- Cálculo da distância para a rodovia mais próxima
            MIN(ST_DISTANCE(c.centroide, o.geometry)) as dist_rodovia
        FROM censo_rais c
        CROSS JOIN osm_rodovias o
        GROUP BY 1, 2, 3, 4, 5, 6, 7
        """

    def executar_geracao(self):
        logger.info("🚀 Iniciando extração da malha mestra (BigQuery SP)...")
        
        try:
            # 1. Extração
            query_job = self.bq_client.query(self.obter_sql_mestre())
            df_pandas = query_job.to_dataframe()
            logger.info(f"✅ Dados extraídos: {len(df_pandas)} setores processados.")

            # 2. Indexação H3 (Resolução 9)
            logger.info("⬢ Gerando índices hexagonais H3...")
            df_pandas['H3_INDEX'] = df_pandas.apply(
                lambda row: h3.latlng_to_cell(row['lat'], row['lon'], 9), axis=1
            )

            # 3. Equalização e Agregação com Polars
            df_pl = pl.from_pandas(df_pandas)
            
            malha_mestra = df_pl.group_by("H3_INDEX").agg([
                pl.col("id_municipio").first().alias("id_municipio"),
                pl.col("pessoas").sum().alias("populacao_fixa"),
                pl.col("empregos").sum().alias("vagas_trabalho_diurno"),
                pl.col("dist_rodovia").mean().alias("distancia_media_rodovia"),
                (pl.col("pessoas").sum() / pl.col("area").sum()).alias("densidade_demografica"),
                (pl.col("domicilios").sum() / pl.col("area").sum()).alias("densidade_habitacional")
            ]).fill_null(0)

            # 4. Upload para o Cloudflare R2
            logger.info("📤 Enviando Parquet de referência para o R2...")
            
            buffer = io.BytesIO()
            malha_mestra.write_parquet(buffer)
            buffer.seek(0)

            self.r2_client.upload_fileobj(
                buffer, 
                self.bucket_name, 
                "referencia/malha_mestra_sp_rica.parquet"
            )

            logger.info("✨ Processo concluído! Malha guardada em referencia/malha_mestra_sp_rica.parquet")

        except Exception as e:
            logger.error(f"❌ Erro durante o processamento: {str(e)}")
            raise

if __name__ == "__main__":
    GeradorReferencia().executar_geracao()
