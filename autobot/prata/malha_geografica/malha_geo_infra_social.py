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
    """Sensor inteligente para inferir codificação e evitar corrupção de strings."""
    try:
        with open(caminho_arquivo, 'rb') as f:
            amostra_bytes = f.read(100000)
        resultado = charset_normalizer.detect(amostra_bytes)
        encoding_detectado = resultado['encoding']
        
        if not encoding_detectado:
            return 'utf-8'
            
        print(f"   [Sensor] Encoding '{encoding_detectado}' detetado para {os.path.basename(caminho_arquivo)}")
        return encoding_detectado
    except Exception as e:
        print(f"   [Aviso] Falha ao detetar encoding de {caminho_arquivo}: {e}. A assumir utf-8.")
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
        
        # 🔐 SEGURANÇA E AMBIENTE (BigQuery + LGPD Pepper)
        project_id = os.getenv('BQ_PROJECT_ID', 'safe-driver-fc3a9')
        self.bq_client = bigquery.Client(project=project_id)
        
        # O Pepper global secreto injetado pelo GitHub Actions
        self.lgpd_pepper = os.getenv('LGPD_PEPPER', 'safedriver_pepper_default_123')
        
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
        print("📥 A DESCARREGAR ATIVOS DA CAMADA BRONZE (R2)...")
        ficheiros = [
            "Agregados_por_setores_basico_BR_20250417.csv", "SP_Faces_2022.zip", 
            "sp-latest.osm.pbf", "SP_Municipios_2022.shp", "SP_Municipios_2022.dbf", 
            "SP_Municipios_2022.shx", "SP_Municipios_2022.prj"
        ]
        for f in ficheiros:
            local = os.path.join(BRONZE_DIR, f)
            if not os.path.exists(local): 
                self.s3.download_file(self.bucket, f"datalake/bronze/malha_raw/{f}", local)

    def normalizar_e_validar_espacial(self, df_pl):
        print("🛡️ A NORMALIZAR E VALIDAR INTEGRIDADE ESPACIAL...")
        mun_path = f"{BRONZE_DIR}/SP_Municipios_2022.shp"
        dbf_path = f"{BRONZE_DIR}/SP_Municipios_2022.dbf"
        
        encoding_dinamico = descobrir_encoding(dbf_path)
        
        municipios = gpd.read_file(mun_path, encoding=encoding_dinamico).to_crs("EPSG:4326")
        municipios["NM_MUN"] = municipios["NM_MUN"].apply(normalizar_string)

        pdf = df_pl.to_pandas()
        gdf = gpd.GeoDataFrame(pdf, geometry=gpd.points_from_xy(pdf['LON'], pdf['LAT']), crs="EPSG:4326")
        joined = gpd.sjoin(gdf, municipios[['NM_MUN', 'geometry']], how="left", predicate="within")
        
        coluna_mun = 'NM_MUN_right' if 'NM_MUN_right' in joined.columns else 'NM_MUN'
        
        if 'NM_MUN_left' in joined.columns:
            erros = joined[joined['NM_MUN_left'] != joined[coluna_mun]].shape[0]
            self.audit_log["AUDITORIA_QUALIDADE"]["ERROS_MUNICIPAIS_CORRIGIDOS"] += erros
        
        joined["MUNICIPIO_VALIDADO"] = joined[coluna_mun].fillna("FORA DA AREA")
        
        cols_remover = ["geometry", "index_right", "NM_MUN", "NM_MUN_left", "NM_MUN_right"]
        cols_existentes = [c for c in cols_remover if c in joined.columns]
        return pl.from_pandas(joined.drop(columns=cols_existentes))

    def normalizar_viaria(self):
        print("📍 A PROCESSAR MALHA VIÁRIA ESTADUAL EM LOTE...")
        faces_zip = f"{BRONZE_DIR}/SP_Faces_2022.zip"
        
        json_paths = []
        with zipfile.ZipFile(faces_zip, 'r') as z:
            for f in z.namelist():
                if f.endswith('.json'): 
                    z.extract(f, BRONZE_DIR)
                    json_paths.append(os.path.join(BRONZE_DIR, f))

        lista_df_faces = []
        for path in json_paths:
            q_faces = f"""
                SELECT CD_SETOR, trim(NM_TIP_LOG || ' ' || NM_LOG) as RUA, 
                ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON 
                FROM ST_Read('{path}') WHERE NM_LOG IS NOT NULL
            """
            try:
                df = self.con.execute(q_faces).pl()
                lista_df_faces.append(df)
            except Exception as e:
                print(f"⚠️ Erro ao ler {path}: {e}")

        df_faces = pl.concat(lista_df_faces)
        lista_df_faces.clear()

        df_faces = self.normalizar_e_validar_espacial(df_faces)

        df_final = df_faces.with_columns([
            pl.col("RUA").map_elements(normalizar_string, return_dtype=pl.Utf8),
            pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([
                h3.latlng_to_cell(x["LAT"], x["LON"], H3_RES) for x in s
            ])).alias("H3_INDEX")
        ]).unique(subset=["H3_INDEX", "RUA"]).sort("H3_INDEX")

        df_final.write_parquet(f"{PRATA_DIR}/MALHA_VIARIA_INFRA.parquet", compression="zstd")
        self.audit_log["CAMADAS"]["VIARIA"] = {"REGISTROS": df_final.height}
        
        for path in json_paths:
            if os.path.exists(path): os.remove(path)

    def normalizar_social(self):
        print("👥 A NORMALIZAR MALHA SOCIAL...")
        csv_path = f"{BRONZE_DIR}/Agregados_por_setores_basico_BR_20250417.csv"
        
        encoding_dinamico = descobrir_encoding(csv_path)
        
        df = pl.read_csv(
            csv_path, separator=";", encoding=encoding_dinamico, 
            schema_overrides={"CD_SETOR": pl.Utf8}, 
            null_values=["."], infer_schema_length=10000
        )
        
        df_final = df.filter(pl.col("CD_SETOR").str.starts_with("35")).select([
            pl.col("CD_SETOR"),
            pl.col("NM_MUN").map_elements(normalizar_string, return_dtype=pl.Utf8).alias("MUNICIPIO"),
            pl.col("NM_BAIRRO").map_elements(normalizar_string, return_dtype=pl.Utf8).alias("BAIRRO"),
            pl.col("v0001").cast(pl.Int32).fill_null(0).alias("POPULACAO")
        ]).unique(subset=["CD_SETOR"]).sort("CD_SETOR")
        
        df_final.write_parquet(f"{PRATA_DIR}/MALHA_SOCIAL.parquet", compression="zstd")
        self.audit_log["CAMADAS"]["SOCIAL"] = {"REGISTROS": df_final.height}

    def normalizar_comercial(self):
        print("🛍️ EXECUTAR DATA FUSION (BQ INNER JOIN CEP + OSM)...")
        osm_path = f"{BRONZE_DIR}/sp-latest.osm.pbf"
        
        # O SUPER JOIN corrigido: Extrai ST_Y e ST_X da coluna 'centroide'
        query_bq = """
            SELECT 
                e.nome_fantasia as NOME_BRUTO,
                CASE 
                    WHEN SUBSTR(e.cnae_fiscal_principal, 1, 5) = '84248' THEN 'SEGURANCA_DELEGACIA'
                    WHEN SUBSTR(e.cnae_fiscal_principal, 1, 2) IN ('47') THEN 'VAREJO_COMERCIO'
                    WHEN SUBSTR(e.cnae_fiscal_principal, 1, 2) IN ('56') THEN 'ALIMENTACAO_BAR'
                    WHEN SUBSTR(e.cnae_fiscal_principal, 1, 2) IN ('64') THEN 'FINANCEIRO_BANCO'
                    WHEN SUBSTR(e.cnae_fiscal_principal, 1, 2) IN ('85') THEN 'EDUCACAO'
                    WHEN SUBSTR(e.cnae_fiscal_principal, 1, 1) IN ('1', '2', '3') THEN 'INDUSTRIA_FABRICA'
                    WHEN SUBSTR(e.cnae_fiscal_principal, 1, 2) IN ('46') THEN 'ATACADO_GALPAO'
                    WHEN SUBSTR(e.cnae_fiscal_principal, 1, 2) IN ('52') THEN 'LOGISTICA_ARMAZEM'
                    WHEN SUBSTR(e.cnae_fiscal_principal, 1, 2) IN ('69', '70', '82') THEN 'ESCRITORIO_CORPORATIVO'
                    ELSE 'OUTROS'
                END as CATEGORIA,
                ST_Y(c.centroide) as LAT,
                ST_X(c.centroide) as LON
            FROM `basedosdados.br_me_cnpj.estabelecimentos` e
            INNER JOIN `basedosdados.br_bd_diretorios_brasil.cep` c
                ON e.cep = c.cep
            WHERE e.sigla_uf = 'SP' 
              AND c.centroide IS NOT NULL
              AND (
                  SUBSTR(e.cnae_fiscal_principal, 1, 5) = '84248'
                  OR SUBSTR(e.cnae_fiscal_principal, 1, 2) IN ('47', '56', '64', '85', '46', '52', '69', '70', '82')
                  OR SUBSTR(e.cnae_fiscal_principal, 1, 1) IN ('1', '2', '3')
              )
        """
        
        query_osm = f"""
            SELECT 'POSTO_OSM' as NOME_BRUTO, 'SEGURANCA_DELEGACIA' as CATEGORIA, ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON 
            FROM ST_Read('{osm_path}', layer='points') WHERE other_tags LIKE '%"amenity"=>"police"%'
            UNION ALL
            SELECT 'POSTO_OSM' as NOME_BRUTO, 'SEGURANCA_DELEGACIA' as CATEGORIA, ST_Y(ST_Centroid(geom)) as LAT, ST_X(ST_Centroid(geom)) as LON 
            FROM ST_Read('{osm_path}', layer='multipolygons') WHERE other_tags LIKE '%"amenity"=>"police"%'
        """
        
        try:
            print("   -> 📡 A cruzar CNPJs com a malha de CEPs (GEOGRAPHY) no BigQuery...")
            df_bq = pl.from_pandas(self.bq_client.query(query_bq).to_dataframe())
            
            print("   -> 🗺️ A ler postos comunitários OSM...")
            df_osm = self.con.execute(query_osm).pl()
            
            print("   -> 🧬 A gerar H3 e Criptografia (Geo-Salt + Pepper)...")
            
            # 1. H3 Indexing (O nosso Geo-Salt)
            df_h3 = pl.concat([df_bq, df_osm]).with_columns([
                pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([
                    h3.latlng_to_cell(x["LAT"], x["LON"], H3_RES) for x in s
                ])).alias("H3_INDEX")
            ])
            
            # 2. Pseudonimização LGPD
            df_unido = df_h3.with_columns([
                pl.concat_str([
                    pl.col("NOME_BRUTO").fill_null("SEM_NOME"),
                    pl.col("H3_INDEX"),
                    pl.lit(self.lgpd_pepper)
                ]).hash().cast(pl.Utf8).alias("HASH_PSEUDONIMIZADO")
            ])
            
            df_unido = df_unido.drop(["NOME_BRUTO", "LAT", "LON"])
            
            print("   -> 🧹 A aplicar desduplicação espacial de segurança...")
            df_seguranca = df_unido.filter(pl.col("CATEGORIA") == "SEGURANCA_DELEGACIA").unique(subset=["H3_INDEX"])
            df_outros = df_unido.filter(pl.col("CATEGORIA") != "SEGURANCA_DELEGACIA")
            
            # 3. Agregação Final Feature Store
            df_final = pl.concat([df_seguranca, df_outros]).group_by(["H3_INDEX", "CATEGORIA"]).agg([
                pl.len().alias("QTD_TOTAL")
            ]).sort("H3_INDEX")
            
            df_final.write_parquet(f"{PRATA_DIR}/MALHA_ESTABELECIMENTOS.parquet", compression="zstd")
            self.audit_log["CAMADAS"]["ESTABELECIMENTOS"] = {
                "REGISTROS": df_final.height, 
                "STATUS": "DATA FUSION CONCLUÍDO (SALT+PEPPER LGPD)"
            }
            
        except Exception as e:
            print(f"❌ Erro no processo de Data Fusion: {e}")
            self.audit_log["CAMADAS"]["ESTABELECIMENTOS"] = {"REGISTROS": 0, "STATUS": f"ERRO: {e}"}

    def upload_final(self):
        print("🚀 A GERAR AUDITORIA UNIFICADA E A SUBIR PARA R2...")
        
        audit_file = f"{PRATA_DIR}/AUDITORIA_PRATA.json"
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(self.audit_log, f, indent=4, ensure_ascii=False)
            
        for root, dirs, ficheiros in os.walk(PRATA_DIR):
            for f in ficheiros:
                local = os.path.join(root, f)
                key = f"datalake/prata/malha_geo_infra_social/{f}"
                self.s3.upload_file(local, self.bucket, key)

    def executar(self):
        self.download_bronze()
        self.normalizar_viaria()
        self.normalizar_social()
        self.normalizar_comercial()
        self.upload_final()
        print("✅ PIPELINE FINALIZADO COM SUCESSO. A CAMADA PRATA ESTÁ BLINDADA E OTIMIZADA.")

if __name__ == "__main__":
    ArquitetoSafeDriver().executar()
