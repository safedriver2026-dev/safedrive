import os
import json
import boto3
import duckdb
import polars as pl
import h3
import unicodedata
import geopandas as gpd
import zipfile
import charset_normalizer
from datetime import datetime
from google.cloud import bigquery

# ==========================================
# CONFIGURAÇÕES DE ARQUITETURA SAFEDRIVER
# ==========================================
H3_RES = 9 
BRONZE_DIR = "data_raw"
PRATA_DIR = "data_prata"

def normalizar_string(valor):
    if valor is None or valor == "" or str(valor).upper() in ["NULL", "NAN", ".", "N/A"]: 
        return "NAO INFORMADO"
    texto = str(valor)
    texto = "".join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn')
    return texto.upper().strip()

def descobrir_encoding(caminho_arquivo):
    try:
        with open(caminho_arquivo, 'rb') as f:
            amostra_bytes = f.read(100000)
        resultado = charset_normalizer.detect(amostra_bytes)
        encoding_detectado = resultado['encoding'] or 'utf-8'
        print(f"   [Sensor] Encoding '{encoding_detectado}' detectado para {os.path.basename(caminho_arquivo)}")
        return encoding_detectado
    except:
        return 'utf-8'

class ArquitetoSafeDriver:
    def __init__(self):
        self.audit_log = {
            "DATA_EXECUCAO": datetime.now().isoformat(), 
            "AUDITORIA_QUALIDADE": {"ERROS_MUNICIPAIS_CORRIGIDOS": 0}, 
            "CAMADAS": {}
        }
        
        os.makedirs(PRATA_DIR, exist_ok=True)
        os.makedirs(BRONZE_DIR, exist_ok=True)
        
        # 🔐 SEGURANÇA E AMBIENTE
        self.project_id = os.getenv('BQ_PROJECT_ID')
        self.dataset_id = os.getenv('BQ_DATASET_ID')
        self.bq_client = bigquery.Client(project=self.project_id)
        
        self.lgpd_pepper = os.getenv('LGPD_PEPPER')
        self.lgpd_salt = os.getenv('LGPD_SALT')
        
        os.environ["OGR_INTERLEAVED_READING"] = "YES"
        self.gerar_configuracao_osm()
        
        self.s3 = boto3.client('s3',
            endpoint_url=os.getenv('R2_ENDPOINT_URL', '').strip(),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID', '').strip(),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY', '').strip()
        )
        self.bucket = os.getenv('R2_BUCKET_NAME', '').strip()
        self.con = duckdb.connect()
        self.con.execute("INSTALL spatial; LOAD spatial;")

    def gerar_configuracao_osm(self):
        ini_content = "[points]\nosm_id=yes\nattributes=name\nother_tags=yes\n[lines]\nosm_id=yes\nattributes=name\nother_tags=yes\n[multipolygons]\nosm_id=yes\nattributes=name\nother_tags=yes"
        with open("osmconf.ini", "w", encoding="utf-8") as f: 
            f.write(ini_content)
        os.environ["OSM_CONFIG_FILE"] = os.path.abspath("osmconf.ini")

    def download_bronze(self):
        print("📥 DESCARREGANDO ATIVOS DA CAMADA BRONZE (R2)...")
        ficheiros = ["Agregados_por_setores_basico_BR_20250417.csv", "SP_Faces_2022.zip", 
                     "sp-latest.osm.pbf", "SP_Municipios_2022.shp", "SP_Municipios_2022.dbf", 
                     "SP_Municipios_2022.shx", "SP_Municipios_2022.prj"]
        for f in ficheiros:
            local = os.path.join(BRONZE_DIR, f)
            if not os.path.exists(local): 
                self.s3.download_file(self.bucket, f"datalake/bronze/malha_raw/{f}", local)

    def normalizar_e_validar_espacial(self, df_pl):
        mun_path = f"{BRONZE_DIR}/SP_Municipios_2022.shp"
        dbf_path = f"{BRONZE_DIR}/SP_Municipios_2022.dbf"
        encoding_dinamico = descobrir_encoding(dbf_path)
        municipios = gpd.read_file(mun_path, encoding=encoding_dinamico).to_crs("EPSG:4326")
        municipios["NM_MUN"] = municipios["NM_MUN"].apply(normalizar_string)
        pdf = df_pl.to_pandas()
        gdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf['LON'], pdf['LAT']), crs="EPSG:4326")
        joined = gpd.sjoin(gdf, municipios[['NM_MUN', 'geometry']], how="left", predicate="within")
        return pl.from_pandas(joined.drop(columns=["geometry", "index_right"]))

    def normalizar_viaria(self):
        print("📍 PROCESSANDO MALHA VIÁRIA...")
        faces_zip = f"{BRONZE_DIR}/SP_Faces_2022.zip"
        json_paths = []
        with zipfile.ZipFile(faces_zip, 'r') as z:
            for f in z.namelist():
                if f.endswith('.json'): 
                    z.extract(f, BRONZE_DIR)
                    json_paths.append(os.path.join(BRONZE_DIR, f))
        lista_df = []
        for path in json_paths:
            df = self.con.execute(f"SELECT CD_SETOR, trim(NM_TIP_LOG || ' ' || NM_LOG) as RUA, ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON FROM ST_Read('{path}') WHERE NM_LOG IS NOT NULL").pl()
            lista_df.append(df)
        df_final = self.normalizar_e_validar_espacial(pl.concat(lista_df))
        df_final = df_final.with_columns([
            pl.col("RUA").map_elements(normalizar_string, return_dtype=pl.Utf8),
            pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], H3_RES) for x in s])).alias("H3_INDEX")
        ]).unique(subset=["H3_INDEX", "RUA"])
        df_final.write_parquet(f"{PRATA_DIR}/MALHA_VIARIA_INFRA.parquet")
        for path in json_paths: os.remove(path)

    def normalizar_social(self):
        print("👥 A NORMALIZAR MALHA SOCIAL...")
        csv_path = f"{BRONZE_DIR}/Agregados_por_setores_basico_BR_20250417.csv"
        enc = descobrir_encoding(csv_path)
        
        # 🛡️ BLINDAGEM RESTAURADA: null_values=["."] resolve o erro do CD_SIT
        df = pl.read_csv(
            csv_path, 
            separator=";", 
            encoding=enc, 
            schema_overrides={"CD_SETOR": pl.Utf8}, 
            infer_schema_length=10000,
            null_values=["."] 
        )
        
        df_final = df.filter(pl.col("CD_SETOR").str.starts_with("35")).select([
            pl.col("CD_SETOR"),
            pl.col("NM_MUN").map_elements(normalizar_string, return_dtype=pl.Utf8).alias("MUNICIPIO"),
            pl.col("v0001").cast(pl.Int32).fill_null(0).alias("POPULACAO")
        ])
        df_final.write_parquet(f"{PRATA_DIR}/MALHA_SOCIAL.parquet")

    def normalizar_comercial(self):
        print("🛍️ EXECUTAR DATA FUSION (BQ + OSM)...")
        osm_path = f"{BRONZE_DIR}/sp-latest.osm.pbf"
        query_bq = """
            SELECT e.nome_fantasia as NOME_BRUTO,
            CASE 
                WHEN SUBSTR(e.cnae_fiscal_principal, 1, 5) = '84248' THEN 'SEGURANCA_DELEGACIA'
                WHEN SUBSTR(e.cnae_fiscal_principal, 1, 2) IN ('47') THEN 'VAREJO_COMERCIO'
                WHEN SUBSTR(e.cnae_fiscal_principal, 1, 1) IN ('1', '2', '3') THEN 'INDUSTRIA_FABRICA'
                ELSE 'OUTROS'
            END as CATEGORIA,
            ST_Y(c.centroide) as LAT, ST_X(c.centroide) as LON
            FROM `basedosdados.br_me_cnpj.estabelecimentos` e
            INNER JOIN `basedosdados.br_bd_diretorios_brasil.cep` c ON e.cep = c.cep
            WHERE e.sigla_uf = 'SP' AND c.centroide IS NOT NULL
        """
        df_bq = pl.from_pandas(self.bq_client.query(query_bq).to_dataframe())
        
        query_osm = f"SELECT 'POSTO_OSM' as NOME_BRUTO, 'SEGURANCA_DELEGACIA' as CATEGORIA, ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON FROM ST_Read('{osm_path}', layer='points') WHERE other_tags LIKE '%\"amenity\"=>\"police\"%'"
        df_osm = self.con.execute(query_osm).pl()

        df_unido = pl.concat([df_bq, df_osm]).with_columns([
            pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], H3_RES) for x in s])).alias("H3_INDEX"),
            pl.concat_str([
                pl.col("NOME_BRUTO").fill_null("SEM_NOME"),
                pl.lit(self.lgpd_salt),
                pl.lit(self.lgpd_pepper)
            ]).hash().cast(pl.Utf8).alias("HASH_PSEUDONIMIZADO")
        ]).drop(["NOME_BRUTO", "LAT", "LON"])
        
        df_final = df_unido.group_by(["H3_INDEX", "CATEGORIA"]).agg(pl.len().alias("QTD")).sort("H3_INDEX")
        df_final.write_parquet(f"{PRATA_DIR}/MALHA_ESTABELECIMENTOS.parquet")

    def upload_final(self):
        print("🚀 SUBINDO PARA R2...")
        for root, _, files in os.walk(PRATA_DIR):
            for f in files:
                self.s3.upload_file(os.path.join(root, f), self.bucket, f"datalake/prata/malha_geo_infra_social/{f}")

    def executar(self):
        self.download_bronze()
        self.normalizar_viaria()
        self.normalizar_social()
        self.normalizar_comercial()
        self.upload_final()
        print("✅ PIPELINE FINALIZADO COM SUCESSO.")

if __name__ == "__main__":
    ArquitetoSafeDriver().executar()
