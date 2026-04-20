import os
import duckdb
import polars as pl
import h3
import unicodedata
import zipfile
import glob
import charset_normalizer
from datetime import date, datetime

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
        return resultado['encoding'] or 'utf-8'
    except:
        return 'utf-8'

class ArquitetoSafeDriverPrata:
    def __init__(self):
        os.makedirs(PRATA_DIR, exist_ok=True)
        # DuckDB em memória para Joins Espaciais de alta performance
        self.con = duckdb.connect(database=':memory:')
        self.con.execute("INSTALL spatial; LOAD spatial;")
        self.gerar_configuracao_osm()

    def gerar_configuracao_osm(self):
        ini_content = "[points]\nosm_id=yes\nattributes=name\nother_tags=yes\n[lines]\nosm_id=yes\nattributes=name\nother_tags=yes"
        with open("osmconf.ini", "w", encoding="utf-8") as f: 
            f.write(ini_content)
        os.environ["OSM_CONFIG_FILE"] = os.path.abspath("osmconf.ini")

    # ==========================================
    # 1. MALHA VIÁRIA (BASAL ANUAL)
    # ==========================================
    def normalizar_viaria(self):
        print("📍 [PRATA] PROCESSANDO MALHA VIÁRIA ANUAL...")
        faces_zip = f"{BRONZE_DIR}/SP_Faces_2022.zip"
        mun_shp = f"{BRONZE_DIR}/SP_Municipios_2022.shp"
        
        if not os.path.exists(faces_zip) or not os.path.exists(mun_shp):
            print("⚠️ Arquivos viários não encontrados. Pulando malha basal.")
            return

        json_paths = []
        with zipfile.ZipFile(faces_zip, 'r') as z:
            for f in z.namelist():
                if f.endswith('.json'): 
                    z.extract(f, BRONZE_DIR)
                    json_paths.append(os.path.join(BRONZE_DIR, f))

        self.con.execute(f"CREATE TABLE municipios AS SELECT NM_MUN, geom FROM ST_Read('{mun_shp}')")
        
        lista_dfs = []
        for path in json_paths:
            query = f"""
                SELECT m.NM_MUN as MUNICIPIO, trim(r.NM_TIP_LOG || ' ' || r.NM_LOG) as RUA, 
                       ST_Y(ST_Centroid(r.geom)) as LAT, ST_X(ST_Centroid(r.geom)) as LON 
                FROM ST_Read('{path}') r 
                LEFT JOIN municipios m ON ST_Within(ST_Centroid(r.geom), m.geom) 
                WHERE r.NM_LOG IS NOT NULL
            """
            lista_dfs.append(self.con.execute(query).pl())
            os.remove(path)

        df_final = pl.concat(lista_dfs).with_columns([
            pl.col("MUNICIPIO").map_elements(normalizar_string, return_dtype=pl.Utf8),
            pl.col("RUA").map_elements(normalizar_string, return_dtype=pl.Utf8),
            pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], H3_RES) for x in s])).alias("H3_INDEX")
        ]).unique(subset=["H3_INDEX", "RUA"])
        
        df_final.write_parquet(f"{PRATA_DIR}/MALHA_VIARIA_INFRA.parquet")
        print("✅ Malha Viária Basal concluída.")

    # ==========================================
    # 2. MALHA SOCIAL (IBGE - SNAPSHOT ANUAL)
    # ==========================================
    def normalizar_social(self):
        print("👥 [PRATA] PROCESSANDO MALHA DEMOGRÁFICA ANUAL...")
        csv_path = f"{BRONZE_DIR}/Agregados_por_setores_basico_BR_20250417.csv"
        if not os.path.exists(csv_path): return

        enc = descobrir_encoding(csv_path)
        # Scan para garantir que não exploda a RAM se o IBGE vier pesado
        df = pl.read_csv(csv_path, separator=";", encoding=enc, 
                         schema_overrides={"CD_SETOR": pl.Utf8}, 
                         infer_schema_length=0, null_values=["."])
        
        df.filter(pl.col("CD_SETOR").str.starts_with("35")).select([
            pl.col("CD_SETOR"),
            pl.col("NM_MUN").map_elements(normalizar_string, return_dtype=pl.Utf8).alias("MUNICIPIO"),
            pl.col("v0001").cast(pl.Int32, strict=False).fill_null(0).alias("POPULACAO")
        ]).write_parquet(f"{PRATA_DIR}/MALHA_SOCIAL.parquet")
        print("✅ Malha Demográfica concluída.")

    # ==========================================
    # 3. MALHA COMERCIAL (O MONSTRO DE 171M)
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
        print(f"🛍️ [PRATA] PROCESSANDO 171M DE LINHAS (ESTRATÉGIA ANUAL)...")
        pattern = os.path.join(BRONZE_DIR, "CNPJ_SP_HISTORICO_*.parquet")
        arquivos = glob.glob(pattern)
        
        if not arquivos:
            print(f"⚠️ Shards não encontrados. Verifique a ingestão via GCS.")
            return

        # Lazy Scan: O cérebro do processamento
        lf = pl.scan_parquet(pattern)

        # 1. Tipagem Forte (Bronze Agnóstica -> Prata Tipada)
        lf = lf.with_columns([
            pl.col("lat").cast(pl.Float64, strict=False),
            pl.col("lon").cast(pl.Float64, strict=False),
            pl.col("data_inicio_atividade").str.to_date("%Y-%m-%d", strict=False),
            pl.col("data_situacao_cadastral").str.to_date("%Y-%m-%d", strict=False)
        ]).drop_nulls(subset=["lat", "lon"])

        # 2. Geo-Enriquecimento
        lf = lf.with_columns([
            self.classificar_impacto(pl.col("cnae_fiscal_principal")).alias("CATEGORIA"),
            pl.struct(["lat", "lon"]).map_batches(lambda s: pl.Series([
                h3.latlng_to_cell(x["lat"], x["lon"], H3_RES) for x in s
            ])).alias("H3_INDEX")
        ])

        # 3. Lógica de Snapshot Anual
        data_corte = date(2022, 12, 31)
        f_2022 = (pl.col("data_inicio_atividade") <= data_corte) & \
                 ((pl.col("data_situacao_cadastral").is_null()) | (pl.col("data_situacao_cadastral") > data_corte))
        f_atual = (pl.col("situacao_cadastral") == "02")

        agg_2022 = (lf.filter(f_2022).group_by(["H3_INDEX", "CATEGORIA"]).agg(pl.len().alias("QTD_2022")))
        agg_atual = (lf.filter(f_atual).group_by(["H3_INDEX", "CATEGORIA"]).agg(pl.len().alias("QTD_ATUAL")))

        # 4. Join e Streaming (Onde a mágica acontece)
        print("   -> Consolidando 3.447 fragmentos em streaming...")
        df_prata = agg_2022.join(agg_atual, on=["H3_INDEX", "CATEGORIA"], how="outer").collect(streaming=True)
        
        df_prata = df_prata.with_columns([
            pl.col("QTD_2022").fill_null(0).cast(pl.Int32),
            pl.col("QTD_ATUAL").fill_null(0).cast(pl.Int32)
        ]).drop_nulls(subset=["H3_INDEX"])

        df_prata.write_parquet(f"{PRATA_DIR}/MALHA_COMERCIAL_LIMPA.parquet")
        print(f"✅ Malha Comercial Anual finalizada: {len(df_prata)} hexágonos.")

    def executar(self):
        start_time = datetime.now()
        print(f"--- ⚙️ SAFEDRIVER PRATA: SNAPSHOT ANUAL INICIADO ---")
        print(f"Data/Hora: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.normalizar_viaria()
        self.normalizar_social()
        self.normalizar_comercial()
        
        print(f"✅ PROCESSAMENTO ANUAL CONCLUÍDO COM SUCESSO.")
        print(f"Duração Total: {datetime.now() - start_time}")
        print(f"---------------------------------------------------")

if __name__ == "__main__":
    ArquitetoSafeDriverPrata().executar()
