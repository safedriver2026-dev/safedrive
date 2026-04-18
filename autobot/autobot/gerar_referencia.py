import os
import json
import boto3
import polars as pl
import h3
from google.cloud import bigquery
from google.oauth2 import service_account
from botocore.config import Config
import io

class GeradorReferencia:
    def __init__(self):
        # Carrega credenciais dos Secrets do GitHub (via variáveis de ambiente)
        info = json.loads(os.getenv("BQ_SERVICE_ACCOUNT_JSON"))
        self.creds = service_account.Credentials.from_service_account_info(info)
        self.projeto_id = os.getenv("BQ_PROJECT_ID")
        
        # Configuração R2
        self.r2_client = boto3.client(
            "s3",
            endpoint_url=os.getenv("R2_ENDPOINT_URL"),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"),
            config=Config(region_name="auto")
        )
        self.bucket = os.getenv("R2_BUCKET_NAME")

    def extrair_dados_sp(self):
        # Forçamos South America para liberar RAIS e Censo SP
        client = bigquery.Client(
            credentials=self.creds, 
            project=self.projeto_id, 
            location="southamerica-east1"
        )

        sql = """
        WITH osm_rodovias AS (
            SELECT geometry FROM `bigquery-public-data.geo_openstreetmap.planet_features`
            WHERE feature_type = 'lines'
              AND EXISTS(SELECT 1 FROM UNNEST(all_tags) t WHERE t.key = 'highway' AND t.value IN ('motorway', 'trunk', 'primary'))
              -- Bounding Box São Paulo: Filtra o scan para não ler o mundo todo
              AND ST_INTERSECTS(geometry, ST_GEOGFROMTEXT('POLYGON((-53.1 -25.3, -44.1 -25.3, -44.1 -19.7, -53.1 -19.7, -53.1 -25.3))'))
        ),
        censo_rais AS (
            SELECT 
                s.id_municipio, s.pessoas, s.area, s.domicilios,
                ST_CENTROID(s.geometria) as centroide,
                r.quantidade_vinculos_ativos as empregos
            FROM `basedosdados.br_ibge_censo_2022.setor_censitario` s
            LEFT JOIN `basedosdados.br_me_rais.estabelecimento` r 
                ON s.id_municipio = r.id_municipio AND r.ano = 2021
            WHERE s.id_uf = '35'
        )
        SELECT 
            c.pessoas, c.area, c.domicilios, c.empregos,
            ST_X(c.centroide) as lon, ST_Y(c.centroide) as lat,
            MIN(ST_DISTANCE(c.centroide, o.geometry)) as dist_rodovia
        FROM censo_rais c
        CROSS JOIN osm_rodovias o
        GROUP BY 1, 2, 3, 4, 5, 6
        """
        
        print("📥 Consultando BigQuery em South America...")
        return client.query(sql).to_dataframe()

    def processar_e_subir(self):
        df_raw = self.extrair_dados_sp()
        
        print("⬢ Gerando IDs H3 (Resolução 9)...")
        df_raw['H3_INDEX'] = df_raw.apply(
            lambda r: h3.latlng_to_cell(r['lat'], r['lon'], 9), axis=1
        )
        
        # Equalização por Hexágono
        malha_mestra = pl.from_pandas(df_raw).group_by("H3_INDEX").agg([
            pl.col("pessoas").sum().alias("populacao_fixa"),
            pl.col("empregos").sum().alias("vagas_trabalho"),
            pl.col("dist_rodovia").mean().alias("distancia_rodovia_media"),
            (pl.col("pessoas").sum() / pl.col("area").sum()).alias("densidade_demografica")
        ]).fill_null(0)

        # Salva em Buffer para subir direto pro R2
        buffer = io.BytesIO()
        malha_mestra.write_parquet(buffer)
        buffer.seek(0)

        print("🚀 Subindo Malha Mestra para o R2...")
        self.r2_client.upload_fileobj(
            buffer, 
            self.bucket, 
            "referencia/malha_mestra_sp_rica.parquet"
        )
        print("✅ Processo concluído com sucesso!")

if __name__ == "__main__":
    GeradorReferencia().processar_e_subir()
