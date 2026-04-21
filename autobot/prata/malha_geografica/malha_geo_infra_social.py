import os
import duckdb
import polars as pl
import h3
import unicodedata
import zipfile
import json
import glob
from datetime import datetime

# ==========================================
# ARQUITETURA SAFEDRIVER: CAMADA PRATA
# ==========================================
class ArquitetoSafeDriverPrata:
    def __init__(self):
        # Configurações Essenciais
        self.H3_RES = 9 
        self.bronze_dir = "./data_raw"
        self.prata_dir = "./data_prata"
        os.makedirs(self.prata_dir, exist_ok=True)
        
        # Engine Espacial (DuckDB)
        self.con = duckdb.connect(database=':memory:')
        self.con.execute("INSTALL spatial; LOAD spatial;")
        
        # O Dicionário de Auditoria para o Data Quality
        self.auditoria = {
            "DATA_EXECUCAO": str(datetime.now()),
            "QUALIDADE_GERAL": "PROCESSANDO",
            "CAMADAS": {}
        }

    def _normalizar_string(self, valor):
        """Limpa textos: remove acentos, coloca em maiúsculas e trata nulos."""
        if valor is None or valor == "" or str(valor).upper() in ["NULL", "NAN", ".", "N/A", "NONE"]: 
            return "NAO INFORMADO"
        texto = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        return texto.upper().strip()

    # ---------------------------------------------------------
    # 1. MALHA GEOGRÁFICA (Esqueleto Viário)
    # ---------------------------------------------------------
    def processar_malha_geografica(self):
        print("🗺️ [PRATA] A construir Malha Geográfica (Vias)...")
        zip_path = os.path.join(self.bronze_dir, "SP_Faces_2022.zip")
        
        if not os.path.exists(zip_path):
            print("⚠️ Ficheiro de faces não encontrado no diretório local. A saltar.")
            return

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(self.bronze_dir)

        # Extração Espacial com DuckDB
        query = f"""
            SELECT 
                CD_FACE,
                trim(NM_TIP_LOG || ' ' || NM_LOG) as RUA, 
                ST_Y(ST_Centroid(geom)) as LAT, 
                ST_X(ST_Centroid(geom)) as LON
            FROM ST_Read('{self.bronze_dir}/*.json')
            WHERE NM_LOG IS NOT NULL
        """
        df_geo = self.con.execute(query).pl()
        
        # Transformação e Tipagem Defensiva (Polars)
        df_geo = df_geo.with_columns([
            pl.col("RUA").map_elements(self._normalizar_string, return_dtype=pl.Utf8).cast(pl.Categorical),
            pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([
                h3.latlng_to_cell(x["LAT"], x["LON"], self.H3_RES) for x in s
            ])).alias("H3_INDEX").cast(pl.Categorical),
            pl.col("LAT").cast(pl.Float32, strict=False),
            pl.col("LON").cast(pl.Float32, strict=False)
        ]).unique(subset=["CD_FACE"])
        
        caminho_saida = os.path.join(self.prata_dir, "PRATA_MALHA_GEOGRAFICA_VIAS.parquet")
        df_geo.write_parquet(caminho_saida, compression="zstd", compression_level=9)
        
        # Registo de Auditoria
        self.auditoria["CAMADAS"]["GEOGRAFICA"] = {
            "REGISTROS_TOTAIS": len(df_geo),
            "HEXAGONOS_H3_UNICOS": df_geo["H3_INDEX"].n_unique(),
            "VALORES_NULOS_LAT_LON": df_geo.select(pl.col("LAT").is_null().sum()).item()
        }

    # ---------------------------------------------------------
    # 2. MALHA SOCIAL (Demografia IBGE)
    # ---------------------------------------------------------
    def processar_malha_social(self):
        print("👥 [PRATA] A construir Malha Social/Demográfica...")
        csv_path = glob.glob(os.path.join(self.bronze_dir, "Agregados_por_setores_basico*.csv"))
        
        if not csv_path:
            print("⚠️ CSV de Setores Censitários não encontrado. A saltar.")
            return
            
        # Leitura Inteligente e Proteção da Chave Primária (CD_SETOR)
        df_social = pl.read_csv(
            csv_path[0], separator=";", null_values=["."],
            infer_schema_length=10000, 
            schema_overrides={"CD_SETOR": pl.Utf8, "CD_MUN": pl.Utf8, "CD_BAIRRO": pl.Utf8}
        ).filter(pl.col("CD_SETOR").str.starts_with("35")) # Filtra SP
        
        colunas_alvo = ["CD_SETOR", "CD_MUN", "NM_MUN", "CD_BAIRRO", "NM_BAIRRO", "AREA_KM2", "SITUACAO", "CD_SITUACAO", "CD_TIPO", "v0001", "v0002"]
        df_social = df_social.select([c for c in colunas_alvo if c in df_social.columns])

        # Micro-Otimização e Tratamento de Decimais
        df_social = df_social.with_columns([
            pl.col("AREA_KM2").str.replace(",", ".").cast(pl.Float32, strict=False),
            pl.col("NM_MUN").map_elements(self._normalizar_string, return_dtype=pl.Utf8).cast(pl.Categorical),
            pl.col("NM_BAIRRO").map_elements(self._normalizar_string, return_dtype=pl.Utf8).cast(pl.Categorical),
            pl.col("SITUACAO").map_elements(self._normalizar_string, return_dtype=pl.Utf8).cast(pl.Categorical),
            pl.col("CD_SITUACAO").cast(pl.UInt8, strict=False),
            pl.col("CD_TIPO").cast(pl.UInt8, strict=False),
            pl.col("v0001").cast(pl.UInt32, strict=False).fill_null(0).alias("TOTAL_PESSOAS"),
            pl.col("v0002").cast(pl.UInt32, strict=False).fill_null(0).alias("TOTAL_DOMICILIOS")
        ]).drop(["v0001", "v0002"])

        caminho_saida = os.path.join(self.prata_dir, "PRATA_MALHA_SOCIAL_SETORES.parquet")
        df_social.write_parquet(caminho_saida, compression="zstd", compression_level=9)
        
        # Registo de Auditoria
        self.auditoria["CAMADAS"]["SOCIAL"] = {
            "SETORES_TOTAIS": len(df_social),
            "POPULACAO_MAPEADA": df_social["TOTAL_PESSOAS"].sum(),
            "BAIRROS_MAPEADOS": df_social["NM_BAIRRO"].n_unique()
        }

    # ---------------------------------------------------------
    # 3. MALHA DE INFRAESTRUTURA (Comércio / CNPJ)
    # ---------------------------------------------------------
    def processar_malha_infraestrutura(self):
        print("🏗️ [PRATA] A construir Malha de Infraestrutura (Comércio)...")
        pattern = os.path.join(self.bronze_dir, "CNPJ_SP_HISTORICO_LOTE_*.parquet")
        arquivos = glob.glob(pattern)
        
        if not arquivos:
            print("⚠️ Nenhum lote de CNPJ encontrado. A saltar.")
            return

        # Lê todos os lotes de uma vez para a RAM
        df_infra = pl.read_parquet(pattern)

        # Filtro Espacial e Indexação H3
        df_infra = df_infra.drop_nulls(subset=["lat", "lon"]).with_columns([
            pl.struct(["lat", "lon"]).map_batches(lambda s: pl.Series([
                h3.latlng_to_cell(x["lat"], x["lon"], self.H3_RES) for x in s
            ])).alias("H3_INDEX")
        ])

        # Otimização de Memória Extremamente Agressiva
        colunas_categoria = ["H3_INDEX", "cnae_fiscal_principal", "cep", "municipio", "bairro", "situacao_cadastral"]
        df_infra = df_infra.with_columns(
            [pl.col(c).cast(pl.Categorical) for c in colunas_categoria if c in df_infra.columns] + 
            [
                pl.col("lat").cast(pl.Float32, strict=False),
                pl.col("lon").cast(pl.Float32, strict=False)
            ]
        )
        
        if "data_inicio_atividade" in df_infra.columns:
            df_infra = df_infra.with_columns(pl.col("data_inicio_atividade").cast(pl.Date, strict=False))
            
        caminho_saida = os.path.join(self.prata_dir, "PRATA_MALHA_INFRA_COMERCIAL.parquet")
        df_infra.write_parquet(caminho_saida, compression="zstd", compression_level=9)
        
        # Registo de Auditoria
        self.auditoria["CAMADAS"]["INFRAESTRUTURA"] = {
            "EMPRESAS_TOTAIS": len(df_infra),
            "HEXAGONOS_H3_COM_COMERCIO": df_infra["H3_INDEX"].n_unique(),
            "DIVERSIDADE_CNAE": df_infra["cnae_fiscal_principal"].n_unique() if "cnae_fiscal_principal" in df_infra.columns else 0
        }

    # ---------------------------------------------------------
    # 4. AUDITORIA E FINALIZAÇÃO
    # ---------------------------------------------------------
    def finalizar_auditoria(self):
        self.auditoria["QUALIDADE_GERAL"] = "SUCESSO_TOTAL"
        
        caminho_audit = os.path.join(self.prata_dir, "AUDITORIA_PRATA_MALHAS.json")
        with open(caminho_audit, "w", encoding="utf-8") as f:
            json.dump(self.auditoria, f, indent=4, ensure_ascii=False)
            
        print("\n" + "="*60)
        print("📊 AUDITORIA FINALIZADA (Copie o JSON abaixo e envie no chat):")
        print("="*60)
        print(json.dumps(self.auditoria, indent=4, ensure_ascii=False))
        print("="*60 + "\n")

    def executar(self):
        start = datetime.now()
        print("🚀 INICIANDO PROCESSAMENTO DE PRODUÇÃO (CAMADA PRATA)")
        
        try:
            self.processar_malha_geografica()
            self.processar_malha_social()
            self.processar_malha_infraestrutura()
            self.finalizar_auditoria()
        except Exception as e:
            self.auditoria["QUALIDADE_GERAL"] = f"FALHA_CRITICA: {str(e)}"
            print(f"\n❌ ERRO CRÍTICO NO PIPELINE: {e}")
            self.finalizar_auditoria()
            
        print(f"🏁 [TEMPO TOTAL]: {datetime.now() - start}")

if __name__ == "__main__":
    ArquitetoSafeDriverPrata().executar()
