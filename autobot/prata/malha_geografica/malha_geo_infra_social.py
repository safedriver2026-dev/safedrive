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

class ArquitetoSafeDriver:
    def __init__(self):
        os.makedirs(PRATA_DIR, exist_ok=True)
        os.makedirs(BRONZE_DIR, exist_ok=True)
        
        # 🔐 SEGURANÇA E AMBIENTE
        self.lgpd_pepper = os.getenv('LGPD_PEPPER', 'default_pepper')
        self.lgpd_salt = os.getenv('LGPD_SALT', 'default_salt')
        
        self.con = duckdb.connect()
        self.con.execute("INSTALL spatial; LOAD spatial;")

    def classificar_impacto_urbano(self, cnae_col):
        """Categorização inteligente para identificar Desertos vs Vitalidade"""
        prefixo = cnae_col.str.slice(0, 2)
        return (
            pl.when(prefixo.is_in(["47", "56", "96"]))
            .then(pl.lit("VAREJO_VITALIDADE"))
            .when(prefixo.is_in(["49", "52", "10", "11"]))
            .then(pl.lit("INDUSTRIA_GALPAO_DESERTO"))
            .when(prefixo == "84")
            .then(pl.lit("SEGURANCA_DELEGACIA"))
            .otherwise(pl.lit("OUTROS_SERVICOS"))
        )

    def normalizar_comercial(self):
        print("🛍️ EXECUTANDO DATA FUSION TEMPORAL (BRONZE HISTÓRICA)...")
        
        path_historico = f"{BRONZE_DIR}/CNPJ_SP_HISTORICO.parquet"
        
        if not os.path.exists(path_historico):
            print(f"❌ Erro: Arquivo {path_historico} não encontrado na Bronze.")
            return

        # 1. Carregamento Otimizado (Lazy)
        lf_historico = pl.scan_parquet(path_historico)

        # 2. Definição das Réguas Temporais
        # Panorama 2022: Criado até 2022 e (Ainda Ativo OU fechou depois de 2022)
        filtro_2022 = (pl.col("data_inicio_atividade") <= datetime(2022, 12, 31)) & \
                      ((pl.col("data_situacao").is_null()) | (pl.col("data_situacao") > datetime(2022, 12, 31)))

        # Panorama Atual: Situação Cadastral '02' (Ativa)
        filtro_atual = (pl.col("situacao_cadastral") == "02")

        # 3. Processamento de H3 e Categorias
        print("   -> Calculando índices H3 e Classificando Impacto...")
        lf_processado = lf_historico.with_columns([
            self.classificar_impacto_urbano(pl.col("cnae_principal")).alias("CATEGORIA_IMPACTO"),
            pl.struct(["LAT", "LON"]).map_batches(lambda s: pl.Series([
                h3.latlng_to_cell(x["LAT"], x["LON"], H3_RES) for x in s
            ])).alias("H3_INDEX")
        ])

        # 4. Agregações por Ano (Snapshots)
        print("   -> Gerando Snapshots Temporais...")
        
        agg_2022 = (
            lf_processado.filter(filtro_2022)
            .group_by(["H3_INDEX", "CATEGORIA_IMPACTO"])
            .agg(pl.len().alias("QTD_2022"))
        )

        agg_atual = (
            lf_processado.filter(filtro_atual)
            .group_by(["H3_INDEX", "CATEGORIA_IMPACTO"])
            .agg(pl.len().alias("QTD_ATUAL"))
        )

        # 5. Join e Cálculo de Evolução (Delta)
        df_final = agg_2022.join(agg_atual, on=["H3_INDEX", "CATEGORIA_IMPACTO"], how="outer").collect()
        
        df_final = df_final.with_columns([
            pl.col("QTD_2022").fill_null(0),
            pl.col("QTD_ATUAL").fill_null(0)
        ]).with_columns([
            (pl.col("QTD_ATUAL") - pl.col("QTD_2022")).alias("DELTA_CRESCIMENTO")
        ])

        # 6. Identificação de Risco de Deserto Urbano
        # Se perdeu varejo e ganhou galpão = Alerta de Risco
        df_final.write_parquet(f"{PRATA_DIR}/MALHA_EVOLUCAO_COMERCIAL.parquet")
        print(f"✅ Malha Comercial Evolutiva salva em Prata. Total de Hexágonos: {len(df_final)}")

    def executar(self):
        print(f"⚙️ INICIANDO ARQUITETURA PRATA - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        # self.normalizar_viaria() # Mantém se necessário
        # self.normalizar_social() # Mantém se necessário
        self.normalizar_comercial()
        print("✅ PROCESSAMENTO FINALIZADO.")

if __name__ == "__main__":
    ArquitetoSafeDriver().executar()
