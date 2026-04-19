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
# A variável AUDIT_DIR foi removida para unificar a saída na mesma pasta

def normalizar_string(valor):
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
        os.makedirs(BRONZE_DIR, exist_ok=True)
        
        # CONFIGURACAO DE AMBIENTE PARA OSM
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
        ini_content = "[points]\nosm_id=yes\nattributes=name\nother_tags=yes\n[lines]\nosm_id=yes\nattributes=name\nother_tags=yes"
        with open("osmconf.ini", "w") as f: f.write(ini_content)
        os.environ["OSM_CONFIG_FILE"] = os.path.abspath("osmconf.ini")

    def download_bronze(self):
        print("📥 BAIXANDO ATIVOS DA CAMADA BRONZE (R2)...")
        arquivos = ["Agregados_por_setores_basico_BR_20250417.csv", "SP_Faces_2022.zip", "sp-latest.osm.pbf", 
                    "SP_Municipios_2022.shp", "SP_Municipios_2022.dbf", "SP_Municipios_2022.shx", "SP_Municipios_2022.prj"]
        for f in arquivos:
            local = os.path.join(BRONZE_DIR, f)
            if not os.path.exists(local): self.s3.download_file(self.bucket, f"datalake/bronze/malha_raw/{f}", local)

    # ==========================================
    # 🧠 MOTOR DE NORMALIZACAO E AUDITORIA
    # ==========================================
    def normalizar_e_validar_espacial(self, df_pl):
        print("🛡️ NORMALIZANDO E VALIDANDO INTEGRIDADE ESPACIAL...")
        mun_path = f"{BRONZE_DIR}/SP_Municipios_2022.shp"
        municipios = gpd.read_file(mun_path).to_crs("EPSG:4326")
        municipios["NM_MUN"] = municipios["NM_MUN"].apply(normalizar_string)

        pdf = df_pl.to_pandas()
        gdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf['LON'], pdf['LAT']), crs="EPSG:4326")
        joined = gpd.sjoin(gdf, municipios[['NM_MUN', 'geometry']], how="left", predicate="within")
        
        coluna_mun = 'NM_MUN_right' if 'NM_MUN_right' in joined.columns else 'NM_MUN'
        
        # REGISTRA AUDITORIA DE ERROS MUNICIPAIS
        erros = joined[joined['NM_MUN_left'] != joined[coluna_mun]].shape[0] if 'NM_MUN_left' in joined.columns else 0
        self.audit_log["AUDITORIA_QUALIDADE"]["ERROS_MUNICIPAIS_CORRIGIDOS"] = erros
        
        joined["MUNICIPIO_VALIDADO"] = joined[coluna_mun].fillna("FORA DA AREA")
        
        cols_remover = ["geometry", "index_right", "NM_MUN", "NM_MUN_left", "NM_MUN_right"]
        cols_existentes = [c for c in cols_remover if c in joined.columns]
        return pl.from_pandas(joined.drop(columns=cols_existentes))

    # ==========================================
    # 📍 ETAPA 1: NORMALIZACAO DA MALHA VIARIA
    # ==========================================
    def normalizar_viaria(self):
        print("📍 NORMALIZANDO MALHA VIARIA ESTADUAL EM LOTE...")
        faces_zip = f"{BRONZE_DIR}/SP_Faces_2022.zip"
        osm_path = f"{BRONZE_DIR}/sp-latest.osm.pbf"
        
        json_paths = []
        with zipfile.ZipFile(faces_zip, 'r') as z:
            for f in z.namelist():
                if f.endswith('.json'): 
                    z.extract(f, BRONZE_DIR)
                    json_paths.append(os.path.join(BRONZE_DIR, f))

        print(f"   -> {len(json_paths)} ficheiros municipais de ruas encontrados. Iniciando processamento espacial...")

        lista_df_faces = []
        for path in json_paths:
            q_faces = f"SELECT CD_SETOR, trim(NM_TIP_LOG || ' ' || NM_LOG) as RUA, ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON FROM ST_Read('{path}') WHERE NM_LOG IS NOT NULL"
            try:
                df = self.con.execute(q_faces).pl()
                lista_df_faces.append(df)
            except Exception as e:
                print(f"   -> Aviso: Erro ao ler a geometria de {path}: {e}")

        df_faces = pl.concat(lista_df_faces)
        lista_df_faces.clear()

        df_faces = self.normalizar_e_validar_espacial(df_faces)

        print("   -> Extraindo infraestrutura viária de todo o OSM...")
        q_osm = f"""
            SELECT * FROM (
                SELECT 
                    COALESCE(regexp_extract(other_tags, '"highway"=>"([^"]+)"', 1), 'NAO INFORMADO') as TIPO_VIA, 
                    ST_Y(ST_Centroid(geom)) as LAT, 
                    ST_X(ST_Centroid(geom)) as LON 
                FROM ST_Read('{osm_path}', layer='lines')
            ) WHERE TIPO_VIA != 'NAO INFORMADO'
        """
        df_osm = self.con.execute(q_osm).pl()

        print("   -> Indexando por H3 e removendo duplicados estaduais...")
        df_final = df_faces.with_columns([
            pl.col("RUA").map_elements(normalizar_string, return_dtype=pl.Utf8),
            pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], H3_RES) for x in s])).alias("H3_INDEX")
        ]).unique(subset=["H3_INDEX", "RUA"]).sort("H3_INDEX")

        df_final.write_parquet(f"{PRATA_DIR}/MALHA_VIARIA_INFRA.parquet", compression="zstd")
        self.audit_log["CAMADAS"]["VIARIA"] = {"REGISTROS": df_final.height}
        
        for path in json_paths:
            if os.path.exists(path):
                os.remove(path)

    # ==========================================
    # 👥 ETAPA 2: NORMALIZACAO DA MALHA SOCIAL
    # ==========================================
    def normalizar_social(self):
        print("👥 NORMALIZANDO MALHA SOCIAL...")
        csv_path = f"{BRONZE_DIR}/Agregados_por_setores_basico_BR_20250417.csv"
        df = pl.read_csv(csv_path, separator=";", encoding="latin1", schema_overrides={"CD_SETOR": pl.Utf8}, null_values=["."], infer_schema_length=10000)
        
        df_final = df.filter(pl.col("CD_SETOR").str.starts_with("35")).select([
            pl.col("CD_SETOR"),
            pl.col("NM_MUN").map_elements(normalizar_string, return_dtype=pl.Utf8).alias("MUNICIPIO"),
            pl.col("NM_BAIRRO").map_elements(normalizar_string, return_dtype=pl.Utf8).alias("BAIRRO"),
            pl.col("v0001").cast(pl.Int32).fill_null(0).alias("POPULACAO")
        ]).unique(subset=["CD_SETOR"]).sort("CD_SETOR")
        
        df_final.write_parquet(f"{PRATA_DIR}/MALHA_SOCIAL.parquet", compression="zstd")
        self.audit_log["CAMADAS"]["SOCIAL"] = {"REGISTROS": df_final.height}

    # ==========================================
    # 🛍️ ETAPA 3: NORMALIZACAO DA MALHA COMERCIAL
    # ==========================================
    def normalizar_comercial(self):
        print("🛍️ NORMALIZANDO MALHA COMERCIAL...")
        osm_path = f"{BRONZE_DIR}/sp-latest.osm.pbf"
        
        q = f"""
            SELECT * FROM (
                SELECT 
                    COALESCE(regexp_extract(other_tags, '"shop"=>"([^"]+)"', 1), 
                             regexp_extract(other_tags, '"amenity"=>"([^"]+)"', 1)) as CAT, 
                    name, 
                    ST_Y(geom) as LAT, 
                    ST_X(geom) as LON 
                FROM ST_Read('{osm_path}', layer='points') 
            ) WHERE CAT IS NOT NULL AND CAT != ''
        """
        df = self.con.execute(q).pl()
        
        df_final = df.with_columns([
            pl.col("CAT").map_elements(normalizar_string, return_dtype=pl.Utf8).alias("CATEGORIA"),
            pl.col("name").map_elements(normalizar_string, return_dtype=pl.Utf8).alias("NOME"),
            pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], H3_RES) for x in s])).alias("H3_INDEX")
        ]).group_by(["H3_INDEX", "CATEGORIA"]).agg(pl.len().alias("QTD")).sort("H3_INDEX")
        
        df_final.write_parquet(f"{PRATA_DIR}/MALHA_ESTABELECIMENTOS.parquet", compression="zstd")
        self.audit_log["CAMADAS"]["ESTABELECIMENTOS"] = {"REGISTROS": df_final.height}

    def upload_final(self):
        print("🚀 GERANDO AUDITORIA E SUBINDO PARA R2...")
        
        # 📄 CRIA O ARQUIVO FISICO DA AUDITORIA DIRETAMENTE NA PRATA_DIR
        audit_file = f"{PRATA_DIR}/AUDITORIA_PRATA.json"
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(self.audit_log, f, indent=4, ensure_ascii=False)
            
        # UPLOAD SIMPLIFICADO: TUDO VAI PARA A MESMA PASTA NO DATALAKE
        for root, dirs, files in os.walk(PRATA_DIR):
            for f in files:
                local = os.path.join(root, f)
                key = f"datalake/prata/malha_geo_infra_social/{f}"
                self.s3.upload_file(local, self.bucket, key)

    def executar(self):
        self.download_bronze()
        self.normalizar_viaria()
        self.normalizar_social()
        self.normalizar_comercial()
        self.upload_final()
        print("✅ TODAS AS ENTIDADES FORAM NORMALIZADAS, AUDITADAS E ESTÃO PRONTAS.")

if __name__ == "__main__":
    ArquitetoSafeDriver().executar()
