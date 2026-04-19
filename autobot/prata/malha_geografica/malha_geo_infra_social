import os
import json
import boto3
import duckdb
import polars as pl
import h3
import unicodedata
import geopandas as gpd
from datetime import datetime

# ==========================================
# CONFIGURACOES DE ARQUITETURA SAFEDRIVER
# ==========================================
H3_RES = 9 
BRONZE_DIR = "data_raw"
PRATA_DIR = "datalake/prata/malha_geo_infra_social"
AUDIT_DIR = "datalake/prata/auditoria"

def limpar_texto(valor):
    if valor is None or valor == "" or str(valor).upper() in ["NULL", "NAN", ".", "N/A"]: 
        return "NAO INFORMADO"
    # REMOVE ACENTOS E CONVERTE PARA MAIUSCULAS
    texto = str(valor)
    texto = "".join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn')
    return texto.upper().strip()

class ArquitetoSafeDriver:
    def __init__(self):
        self.audit_log = {"DATA_EXECUCAO": datetime.now().isoformat(), "AUDITORIA_QUALIDADE": {}, "CAMADAS": {}}
        os.makedirs(PRATA_DIR, exist_ok=True)
        os.makedirs(AUDIT_DIR, exist_ok=True)
        
        # CONFIGURACAO R2 (BOTO3)
        self.s3 = boto3.client('s3',
            endpoint_url=os.getenv('R2_ENDPOINT_URL', '').strip(),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID', '').strip(),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY', '').strip()
        )
        self.bucket = os.getenv('R2_BUCKET_NAME', '').strip()
        
        # MOTOR ESPACIAL DUCKDB
        self.con = duckdb.connect()
        self.con.execute("INSTALL spatial; LOAD spatial;")

    def download_bronze(self):
        print("📥 BAIXANDO ATIVOS DA CAMADA BRONZE (R2)...")
        os.makedirs(BRONZE_DIR, exist_ok=True)
        files = [
            "Agregados_por_setores_basico_BR_20250417.csv",
            "SP_Faces_2022.zip",
            "sp-latest.osm.pbf",
            "SP_Municipios_2022.shp", "SP_Municipios_2022.dbf", "SP_Municipios_2022.shx"
        ]
        for f in files:
            local_path = os.path.join(BRONZE_DIR, f)
            if not os.path.exists(local_path):
                self.s3.download_file(self.bucket, f"datalake/bronze/malha_raw/{f}", local_path)

    # ==========================================
    # 🧠 MOTOR DE INTELIGENCIA: VALIDACAO E RECUPERACAO
    # ==========================================
    def validar_e_recuperar(self, df_pl):
        print("🛡️ EXECUTANDO MOTOR DE INTELIGENCIA E AUDITORIA ESPACIAL...")
        
        # 1. CROSS-CHECK DE MUNICIPIOS VIA SHAPEFILE
        mun_path = f"{BRONZE_DIR}/SP_Municipios_2022.shp"
        municipios = gpd.read_file(mun_path).to_crs("EPSG:4326")
        municipios["NM_MUN"] = municipios["NM_MUN"].apply(limpar_texto)

        pdf = df_pl.to_pandas()
        gdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf['LON'], pdf['LAT']), crs="EPSG:4326")
        
        # DETERMINA O MUNICIPIO REAL PELA GEOMETRIA
        joined = gpd.sjoin(gdf, municipios[['NM_MUN', 'geometry']], how="left", predicate="within")
        joined["MUNICIPIO_REAL"] = joined["NM_MUN_right"].fillna("FORA DA AREA")
        
        # 2. RECUPERACAO DE LOGRADOURO VIA OSM (SE NAO INFORMADO)
        # NESTE EXEMPLO, O LOGRADOURO JA VEM TRATADO DA QUERY DUCKDB QUE UNE AS FONTES
        
        return pl.from_pandas(joined.drop(columns=["geometry", "index_right", "NM_MUN_right"]))

    # ==========================================
    # 1. MALHA VIARIA (GEOGRAFIA + INFRAESTRUTURA)
    # ==========================================
    def processar_viaria(self):
        print("📍 PROCESSANDO MALHA VIARIA (GEO + INFRA)...")
        faces_zip = f"{BRONZE_DIR}/SP_Faces_2022.zip"
        osm_path = f"{BRONZE_DIR}/sp-latest.osm.pbf"
        
        # DUCKDB UNE AS INFORMACOES DAS FACES E INFRA DO OSM VIA PROXIMIDADE
        query = f"""
            SELECT 
                f.CD_SETOR AS ID_SETOR,
                trim(f.NM_TIP_LOG || ' ' || f.NM_LOG) as LOGRADOURO,
                o.highway as TIPO_VIA,
                o.maxspeed as VELOCIDADE_MAXIMA,
                o.surface as PAVIMENTO,
                o.lit as ILUMINACAO,
                ST_Y(ST_Centroid(f.geom)) as LAT,
                ST_X(ST_Centroid(f.geom)) as LON
            FROM ST_Read('/vsizip/{faces_zip}') f
            LEFT JOIN ST_Read('{osm_path}', layer='lines') o 
                ON ST_DWithin(f.geom, o.geom, 0.0001)
            WHERE f.NM_LOG IS NOT NULL
        """
        df = self.con.execute(query).pl()

        # APLICAR MOTOR DE INTELIGENCIA
        df_validado = self.validar_e_recuperar(df)
        
        df_final = df_validado.with_columns([
            pl.col("LOGRADOURO").map_elements(limpar_texto, return_dtype=pl.Utf8),
            pl.col("TIPO_VIA").map_elements(limpar_texto, return_dtype=pl.Utf8),
            pl.col("PAVIMENTO").map_elements(limpar_texto, return_dtype=pl.Utf8),
            pl.col("ILUMINACAO").map_elements(limpar_texto, return_dtype=pl.Utf8),
            pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], H3_RES) for x in s])).alias("H3_INDEX")
        ]).select([
            "H3_INDEX", "LOGRADOURO", "TIPO_VIA", "VELOCIDADE_MAXIMA", "PAVIMENTO", "ILUMINACAO", 
            pl.col("MUNICIPIO_REAL").alias("MUNICIPIO"), "ID_SETOR"
        ]).unique(subset=["H3_INDEX", "LOGRADOURO"]).sort("H3_INDEX")

        caminho = f"{PRATA_DIR}/MALHA_VIARIA_INFRA.parquet"
        df_final.write_parquet(caminho, compression="zstd", statistics=True)
        self.audit_log["CAMADAS"]["VIARIA"] = {"REGISTROS": df_final.height}

    # ==========================================
    # 2. MALHA SOCIAL (DEMOGRAFIA)
    # ==========================================
    def processar_social(self):
        print("👥 PROCESSANDO MALHA SOCIAL...")
        csv_path = f"{BRONZE_DIR}/Agregados_por_setores_basico_BR_20250417.csv"
        df = pl.read_csv(csv_path, separator=";", encoding="latin1", dtypes={"CD_SETOR": pl.Utf8})
        
        df_final = (
            df.filter(pl.col("CD_SETOR").str.starts_with("35"))
            .select([
                pl.col("CD_SETOR").alias("ID_SETOR"),
                pl.col("NM_MUN").map_elements(limpar_texto, return_dtype=pl.Utf8).alias("MUNICIPIO"),
                pl.col("NM_BAIRRO").map_elements(limpar_texto, return_dtype=pl.Utf8).alias("BAIRRO"),
                pl.col("v0001").cast(pl.Int32).fill_null(0).alias("POPULACAO")
            ])
            .unique(subset=["ID_SETOR"])
            .sort("ID_SETOR")
        )
        caminho = f"{PRATA_DIR}/MALHA_SOCIAL.parquet"
        df_final.write_parquet(caminho, compression="zstd", statistics=True)
        self.audit_log["CAMADAS"]["SOCIAL"] = {"REGISTROS": df_final.height}

    # ==========================================
    # 3. MALHA ESTABELECIMENTOS (COMERCIO)
    # ==========================================
    def processar_estabelecimentos(self):
        print("🛍️ PROCESSANDO MALHA ESTABELECIMENTOS...")
        osm_path = f"{BRONZE_DIR}/sp-latest.osm.pbf"
        
        query = f"""
            SELECT 
                COALESCE(regexp_extract(other_tags, '"shop"=>"([^"]+)"', 1), 
                         regexp_extract(other_tags, '"amenity"=>"([^"]+)"', 1)) as CATEGORIA,
                name as NOME, lat as LAT, lon as LON
            FROM ST_Read('{osm_path}', layer='points')
            WHERE CATEGORIA != ''
        """
        df = self.con.execute(query).pl()
        
        df_final = df.with_columns([
            pl.col("CATEGORIA").map_elements(limpar_texto, return_dtype=pl.Utf8),
            pl.col("NOME").map_elements(limpar_texto, return_dtype=pl.Utf8),
            pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], H3_RES) for x in s])).alias("H3_INDEX")
        ]).group_by(["H3_INDEX", "CATEGORIA"]).agg(pl.len().alias("QTD")).sort("H3_INDEX")

        caminho = f"{PRATA_DIR}/MALHA_ESTABELECIMENTOS.parquet"
        df_final.write_parquet(caminho, compression="zstd", statistics=True)
        self.audit_log["CAMADAS"]["ESTABELECIMENTOS"] = {"REGISTROS": df_final.height}

    def finalizar(self):
        # SALVAR AUDITORIA
        with open(f"{AUDIT_DIR}/AUDITORIA_PRATA.json", "w", encoding="utf-8") as f:
            json.dump(self.audit_log, f, indent=4, ensure_ascii=False)
        
        # UPLOAD PARA R2
        for root, dirs, files in os.walk("datalake/prata"):
            for file in files:
                local_path = os.path.join(root, file)
                r2_key = local_path.replace("\\", "/") # GARANTE FORMATO UNIX PARA O R2
                print(f"🚀 UPLOAD: {r2_key}")
                self.s3.upload_file(local_path, self.bucket, r2_key)

    def executar(self):
        self.download_bronze()
        self.processar_viaria()
        self.processar_social()
        self.processar_estabelecimentos()
        self.finalizar()

if __name__ == "__main__":
    ArquitetoSafeDriver().executar()
