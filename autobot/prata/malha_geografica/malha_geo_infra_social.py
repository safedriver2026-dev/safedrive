import os
import duckdb
import polars as pl
import h3
import unicodedata
import geopandas as gpd
import zipfile
import charset_normalizer
from datetime import datetime

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

class ArquitetoSafeDriverPrata:
    def __init__(self):
        os.makedirs(PRATA_DIR, exist_ok=True)
        
        # Inicia DuckDB com suporte Espacial
        self.con = duckdb.connect()
        self.con.execute("INSTALL spatial; LOAD spatial;")
        
        # Configuração para o DuckDB ler arquivos do OpenStreetMap corretamente
        self.gerar_configuracao_osm()

    def gerar_configuracao_osm(self):
        ini_content = "[points]\nosm_id=yes\nattributes=name\nother_tags=yes\n[lines]\nosm_id=yes\nattributes=name\nother_tags=yes\n[multipolygons]\nosm_id=yes\nattributes=name\nother_tags=yes"
        with open("osmconf.ini", "w", encoding="utf-8") as f: 
            f.write(ini_content)
        os.environ["OSM_CONFIG_FILE"] = os.path.abspath("osmconf.ini")
        os.environ["OGR_INTERLEAVED_READING"] = "YES"

    # ==========================================
    # 1. MALHA GEOGRÁFICA (VIÁRIA)
    # ==========================================
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
        print("📍 [PRATA] PROCESSANDO MALHA VIÁRIA...")
        faces_zip = f"{BRONZE_DIR}/SP_Faces_2022.zip"
        
        if not os.path.exists(faces_zip):
            print(f"⚠️ Arquivo {faces_zip} não encontrado. Pulando malha viária.")
            return

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
        print("✅ Malha Viária finalizada.")

    # ==========================================
    # 2. MALHA SOCIAL (DEMOGRÁFICA IBGE)
    # ==========================================
    def normalizar_social(self):
        print("👥 [PRATA] PROCESSANDO MALHA DEMOGRÁFICA...")
        csv_path = f"{BRONZE_DIR}/Agregados_por_setores_basico_BR_20250417.csv"
        
        if not os.path.exists(csv_path):
            print(f"⚠️ Arquivo {csv_path} não encontrado. Pulando malha social.")
            return

        enc = descobrir_encoding(csv_path)
        df = pl.read_csv(
            csv_path, separator=";", encoding=enc, 
            schema_overrides={"CD_SETOR": pl.Utf8}, 
            infer_schema_length=10000, null_values=["."] 
        )
        
        df_final = df.filter(pl.col("CD_SETOR").str.starts_with("35")).select([
            pl.col("CD_SETOR"),
            pl.col("NM_MUN").map_elements(normalizar_string, return_dtype=pl.Utf8).alias("MUNICIPIO"),
            pl.col("v0001").cast(pl.Int32).fill_null(0).alias("POPULACAO")
        ])
        
        df_final.write_parquet(f"{PRATA_DIR}/MALHA_SOCIAL.parquet")
        print("✅ Malha Demográfica finalizada.")

    # ==========================================
    # 3. MALHA COMERCIAL (VITALIDADE E DESERTOS)
    # ==========================================
    def classificar_impacto(self, cnae_col):
        prefixo = cnae_col.str.slice(0, 2)
        return (
            pl.when(prefixo.is_in(["47", "56", "96"])).then(pl.lit("VAREJO_VITALIDADE"))
            .when(prefixo.is_in(["49", "52", "10", "11"])).then(pl.lit("INDUSTRIA_GALPAO"))
            .when(prefixo == "84").then(pl.lit("SEGURANCA_DELEGACIA"))
            .otherwise(pl.lit("OUTROS_SERVICOS"))
        )

    def normalizar_comercial(self):
        print("🛍️ [PRATA] PROCESSANDO MALHA COMERCIAL (HISTÓRICA)...")
        path_bronze = f"{BRONZE_DIR}/CNPJ_SP_HISTORICO.parquet"
        
        if not os.path.exists(path_bronze):
            print(f"⚠️ Arquivo {path_bronze} não encontrado. Pulando malha comercial.")
            return

        lf = pl.scan_parquet(path_bronze).drop_nulls(subset=["lat", "lon"])

        lf = lf.with_columns([
            self.classificar_impacto(pl.col("cnae_principal")).alias("CATEGORIA"),
            pl.struct(["lat", "lon"]).map_batches(lambda s: pl.Series([
                h3.latlng_to_cell(x["lat"], x["lon"], H3_RES) for x in s
            ])).alias("H3_INDEX")
        ])

        filtro_2022 = (pl.col("data_inicio_atividade") <= datetime(2022, 12, 31)) & \
                      ((pl.col("data_situacao").is_null()) | (pl.col("data_situacao") > datetime(2022, 12, 31)))
        
        filtro_atual = (pl.col("situacao_cadastral") == "02")

        agg_2022 = (lf.filter(filtro_2022).group_by(["H3_INDEX", "CATEGORIA"]).agg(pl.len().alias("QTD_2022")))
        agg_atual = (lf.filter(filtro_atual).group_by(["H3_INDEX", "CATEGORIA"]).agg(pl.len().alias("QTD_ATUAL")))

        df_prata = agg_2022.join(agg_atual, on=["H3_INDEX", "CATEGORIA"], how="outer").collect()
        
        df_prata = df_prata.with_columns([
            pl.col("QTD_2022").fill_null(0),
            pl.col("QTD_ATUAL").fill_null(0)
        ]).drop_nulls(subset=["H3_INDEX"])

        df_prata.write_parquet(f"{PRATA_DIR}/MALHA_COMERCIAL_LIMPA.parquet")
        print("✅ Malha Comercial finalizada.")

    def executar(self):
        print(f"⚙️ INICIANDO ARQUITETURA PRATA (TRIPLA MALHA) - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        self.normalizar_viaria()
        self.normalizar_social()
        self.normalizar_comercial()
        print("✅ PROCESSAMENTO TOTAL DA PRATA FINALIZADO COM SUCESSO.")

if __name__ == "__main__":
    ArquitetoSafeDriverPrata().executar()
