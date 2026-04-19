import os
import json
import boto3
import duckdb
import polars as pl
import h3
import unicodedata
import geopandas as gpd
import zipfile
from datetime import datetime

# ==========================================
# CONFIGURACOES DE ARQUITETURA SAFEDRIVER
# ==========================================
H3_RES = 9 
BRONZE_DIR = "data_raw"
PRATA_DIR = "data_prata"
AUDIT_DIR = "data_prata/auditoria"

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
        arquivos = [
            "Agregados_por_setores_basico_BR_20250417.csv",
            "SP_Faces_2022.zip",
            "sp-latest.osm.pbf",
            "SP_Municipios_2022.shp", "SP_Municipios_2022.dbf", "SP_Municipios_2022.shx", "SP_Municipios_2022.prj"
        ]
        for f in arquivos:
            local_path = os.path.join(BRONZE_DIR, f)
            if not os.path.exists(local_path):
                print(f"   -> {f}")
                self.s3.download_file(self.bucket, f"datalake/bronze/malha_raw/{f}", local_path)

    # ==========================================
    # 🧠 MOTOR DE INTELIGENCIA: VALIDACAO MUNICIPAL
    # ==========================================
    def validar_municipio_real(self, df_pl):
        print("🛡️ EXECUTANDO AUDITORIA ESPACIAL: CROSS-CHECK DE MUNICIPIOS...")
        mun_path = f"{BRONZE_DIR}/SP_Municipios_2022.shp"
        municipios = gpd.read_file(mun_path).to_crs("EPSG:4326")
        municipios["NM_MUN"] = municipios["NM_MUN"].apply(limpar_texto)

        pdf = df_pl.to_pandas()
        gdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf['LON'], pdf['LAT']), crs="EPSG:4326")
        
        joined = gpd.sjoin(gdf, municipios[['NM_MUN', 'geometry']], how="left", predicate="within")
        
        erros = joined[joined['NM_MUN_left'] != joined['NM_MUN_right']].shape[0] if 'NM_MUN_left' in joined.columns else 0
        self.audit_log["AUDITORIA_QUALIDADE"]["ERROS_MUNICIPAIS_CORRIGIDOS"] = erros
        
        joined["NM_MUN_REAL"] = joined["NM_MUN_right"].fillna("FORA DA AREA DE COBERTURA")
        return pl.from_pandas(joined.drop(columns=["geometry", "index_right"]))

    # ==========================================
    # 📍 MALHA VIARIA (GEO + INFRA ESTRUTURA)
    # ==========================================
    def processar_viaria(self):
        print("📍 PROCESSANDO MALHA VIARIA (GEO + INFRA)...")
        faces_zip = f"{BRONZE_DIR}/SP_Faces_2022.zip"
        osm_path = f"{BRONZE_DIR}/sp-latest.osm.pbf"
        
        # LOCALIZAR E EXTRAIR JSON INDEPENDENTE DA PASTA INTERNA
        json_path = None
        with zipfile.ZipFile(faces_zip, 'r') as z:
            for filename in z.namelist():
                if filename.endswith('.json'):
                    z.extract(filename, BRONZE_DIR)
                    json_path = os.path.join(BRONZE_DIR, filename)
                    break
        
        query_faces = f"""
            SELECT 
                CD_SETOR AS ID_SETOR,
                trim(NM_TIP_LOG || ' ' || NM_LOG) as LOGRADOURO,
                ST_Y(ST_Centroid(geom)) as LAT,
                ST_X(ST_Centroid(geom)) as LON
            FROM ST_Read('{json_path}')
        """
        df_faces = self.con.execute(query_faces).pl()
        
        # VALIDACAO CRUZADA PARA CORRIGIR MUNICIPIO
        df_faces = self.validar_municipio_real(df_faces)

        # UNIFICACAO COM INFRAESTRUTURA OSM
        query_osm = f"""
            SELECT 
                COALESCE(regexp_extract(other_tags, '"highway"=>"([^"]+)"', 1), 'NAO INFORMADO') as TIPO_VIA,
                COALESCE(regexp_extract(other_tags, '"maxspeed"=>"([^"]+)"', 1), 'NAO INFORMADO') as VELOCIDADE_MAXIMA,
                COALESCE(regexp_extract(other_tags, '"surface"=>"([^"]+)"', 1), 'NAO INFORMADO') as PAVIMENTO,
                COALESCE(regexp_extract(other_tags, '"lit"=>"([^"]+)"', 1), 'NAO INFORMADO') as ILUMINACAO,
                ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON
            FROM ST_Read('{osm_path}', layer='lines')
            WHERE TIPO_VIA != 'NAO INFORMADO'
        """
        df_osm = self.con.execute(query_osm).pl()

        df_final = df_faces.with_columns([
            pl.col("LOGRADOURO").map_elements(limpar_texto, return_dtype=pl.Utf8),
            pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], H3_RES) for x in s])).alias("H3_INDEX")
        ]).unique(subset=["H3_INDEX", "LOGRADOURO"]).rename({"NM_MUN_REAL": "MUNICIPIO"}).sort("H3_INDEX")

        caminho = f"{PRATA_DIR}/MALHA_VIARIA_INFRA.parquet"
        df_final.write_parquet(caminho, compression="zstd", statistics=True)
        self.audit_log["CAMADAS"]["VIARIA"] = {"REGISTROS": df_final.height}
        if os.path.exists(json_path): os.remove(json_path)

    # ==========================================
    # 👥 MALHA SOCIAL
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
            .unique(subset=["ID_SETOR"]).sort("ID_SETOR")
        )
        df_final.write_parquet(f"{PRATA_DIR}/MALHA_SOCIAL.parquet", compression="zstd")
        self.audit_log["CAMADAS"]["SOCIAL"] = {"REGISTROS": df_final.height}

    # ==========================================
    # 🛍️ MALHA ESTABELECIMENTOS
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

        df_final.write_parquet(f"{PRATA_DIR}/MALHA_ESTABELECIMENTOS.parquet", compression="zstd")
        self.audit_log["CAMADAS"]["ESTABELECIMENTOS"] = {"REGISTROS": df_final.height}

    def upload_r2(self):
        print("🚀 SUBINDO PARA R2...")
        with open(f"{AUDIT_DIR}/AUDITORIA_PRATA.json", "w", encoding="utf-8") as f:
            json.dump(self.audit_log, f, indent=4, ensure_ascii=False)
        
        for root, dirs, files in os.walk(PRATA_DIR):
            for file in files:
                local_path = os.path.join(root, file)
                key = f"datalake/prata/malha_geo_infra_social/{file}"
                if file.endswith('.json'): key = f"datalake/prata/auditoria/{file}"
                self.s3.upload_file(local_path, self.bucket, key)

    def executar(self):
        self.download_bronze()
        self.processar_viaria()
        self.processar_social()
        self.processar_estabelecimentos()
        self.upload_r2()

if __name__ == "__main__":
    ArquitetoSafeDriver().executar()
