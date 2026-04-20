import os
import duckdb
import polars as pl
import h3
import unicodedata
import zipfile
import glob
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

class ArquitetoSafeDriverPrata:
    def __init__(self):
        os.makedirs(PRATA_DIR, exist_ok=True)
        self.con = duckdb.connect(database=':memory:')
        self.con.execute("INSTALL spatial; LOAD spatial;")
        self.gerar_configuracao_osm()

    def gerar_configuracao_osm(self):
        ini_content = "[points]\nosm_id=yes\nattributes=name\nother_tags=yes\n[lines]\nosm_id=yes\nattributes=name\nother_tags=yes"
        with open("osmconf.ini", "w", encoding="utf-8") as f: 
            f.write(ini_content)
        os.environ["OSM_CONFIG_FILE"] = os.path.abspath("osmconf.ini")

    def normalizar_viaria(self):
        print("📍 [PRATA] PROCESSANDO MALHA VIÁRIA (BASAL)...")
        faces_zip = f"{BRONZE_DIR}/SP_Faces_2022.zip"
        mun_shp = f"{BRONZE_DIR}/SP_Municipios_2022.shp"
        
        if not os.path.exists(faces_zip): return

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

        pl.concat(lista_dfs).with_columns([
            pl.col("MUNICIPIO").map_elements(normalizar_string, return_dtype=pl.Utf8),
            pl.col("RUA").map_elements(normalizar_string, return_dtype=pl.Utf8),
            pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([h3.latlng_to_cell(x["LAT"], x["LON"], H3_RES) for x in s])).alias("H3_INDEX")
        ]).unique(subset=["H3_INDEX", "RUA"]).write_parquet(f"{PRATA_DIR}/MALHA_VIARIA_INFRA.parquet")

    def normalizar_social(self):
        print("👥 [PRATA] PROCESSANDO MALHA SOCIAL ANUAL...")
        csv_path = f"{BRONZE_DIR}/Agregados_por_setores_basico_BR_20250417.csv"
        if not os.path.exists(csv_path): return

        pl.read_csv(csv_path, separator=";", schema_overrides={"CD_SETOR": pl.Utf8}, 
                    infer_schema_length=0, null_values=["."]).filter(
            pl.col("CD_SETOR").str.starts_with("35")
        ).select([
            pl.col("CD_SETOR"),
            pl.col("NM_MUN").map_elements(normalizar_string, return_dtype=pl.Utf8).alias("MUNICIPIO"),
            pl.col("v0001").cast(pl.Int32, strict=False).fill_null(0).alias("POPULACAO")
        ]).write_parquet(f"{PRATA_DIR}/MALHA_SOCIAL.parquet")

    def normalizar_comercial(self):
        print(f"🛍️ [PRATA] PROCESSANDO 171M DE LINHAS (STREAMING ATIVO)...")
        pattern = os.path.join(BRONZE_DIR, "CNPJ_SP_HISTORICO_*.parquet")
        
        # O scan_parquet lê os 3.447 shards de uma vez só como um LazyFrame
        lf = pl.scan_parquet(pattern)

        # 1. Classificação e Tipagem
        lf = lf.with_columns([
            pl.col("lat").cast(pl.Float64, strict=False),
            pl.col("lon").cast(pl.Float64, strict=False),
            pl.col("data_inicio_atividade").str.to_date("%Y-%m-%d", strict=False),
            pl.col("data_situacao_cadastral").str.to_date("%Y-%m-%d", strict=False)
        ]).drop_nulls(subset=["lat", "lon"])

        lf = lf.with_columns([
            pl.struct(["lat", "lon"]).map_batches(lambda s: pl.Series([
                h3.latlng_to_cell(x["lat"], x["lon"], H3_RES) for x in s
            ])).alias("H3_INDEX")
        ])

        # 2. Agregação por Snapshots (Snapshot 2022 vs Atual)
        data_corte = date(2022, 12, 31)
        f_2022 = (pl.col("data_inicio_atividade") <= data_corte) & \
                 ((pl.col("data_situacao_cadastral").is_null()) | (pl.col("data_situacao_cadastral") > data_corte))
        f_atual = (pl.col("situacao_cadastral") == "02")

        agg_2022 = (lf.filter(f_2022).group_by("H3_INDEX").agg(pl.len().alias("QTD_2022")))
        agg_atual = (lf.filter(f_atual).group_by("H3_INDEX").agg(pl.len().alias("QTD_ATUAL")))

        # 3. Join Final em Streaming (Protege a RAM do GitHub)
        df_prata = agg_2022.join(agg_atual, on="H3_INDEX", how="outer").collect(streaming=True)
        
        df_prata.with_columns([
            pl.col("QTD_2022").fill_null(0).cast(pl.Int32),
            pl.col("QTD_ATUAL").fill_null(0).cast(pl.Int32)
        ]).write_parquet(f"{PRATA_DIR}/MALHA_COMERCIAL_LIMPA.parquet")
        print(f"✅ Malha Comercial processada.")

    def executar(self):
        start = datetime.now()
        self.normalizar_viaria()
        self.normalizar_social()
        self.normalizar_comercial()
        print(f"🚀 [PRATA] Tempo Total: {datetime.now() - start}")

if __name__ == "__main__":
    ArquitetoSafeDriverPrata().executar()
