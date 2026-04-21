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
# ARQUITETURA SAFEDRIVER: CAMADA PRATA (PRODUÇÃO)
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
        
        # Dicionário de Auditoria Blindado
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
        try:
            # Busca recursiva no diretório
            zip_paths = glob.glob(os.path.join(self.bronze_dir, "**", "SP_Faces_2022.zip"), recursive=True)
            
            if not zip_paths:
                raise FileNotFoundError("Arquivo SP_Faces_2022.zip não encontrado no diretório local.")

            with zipfile.ZipFile(zip_paths[0], 'r') as z:
                z.extractall(self.bronze_dir)

            # Extração Espacial com DuckDB
            query = f"""
                SELECT 
                    CD_FACE,
                    trim(NM_TIP_LOG || ' ' || NM_LOG) as RUA, 
                    ST_Y(ST_Centroid(geom)) as LAT, 
                    ST_X(ST_Centroid(geom)) as LON
                FROM ST_Read('{self.bronze_dir}/**/*.json')
                WHERE NM_LOG IS NOT NULL
            """
            df_geo = self.con.execute(query).pl()
            
            # Transformação e Tipagem Defensiva
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
            
            # Registo de Auditoria Seguro (.item() converte de Polars para Python nativo)
            self.auditoria["CAMADAS"]["GEOGRAFICA"] = {
                "STATUS": "SUCESSO",
                "REGISTROS_TOTAIS": df_geo.height,
                "HEXAGONOS_H3_UNICOS": df_geo.select(pl.col("H3_INDEX").n_unique()).item(),
                "VALORES_NULOS_LAT_LON": df_geo.select(pl.col("LAT").is_null().sum()).item()
            }
            
        except Exception as e:
            print(f"❌ Erro na Malha Geográfica: {e}")
            self.auditoria["CAMADAS"]["GEOGRAFICA"] = {"STATUS": "ERRO", "MENSAGEM": str(e)}

    # ---------------------------------------------------------
    # 2. MALHA SOCIAL (Demografia IBGE)
    # ---------------------------------------------------------
    def processar_malha_social(self):
        print("👥 [PRATA] A construir Malha Social/Demográfica...")
        try:
            csv_paths = glob.glob(os.path.join(self.bronze_dir, "**", "Agregados_por_setores_basico*.csv"), recursive=True)
            
            if not csv_paths:
                raise FileNotFoundError("CSV de Setores Censitários não encontrado.")
                
            # Leitura Inteligente e Proteção da Chave Primária
            df_social = pl.read_csv(
                csv_paths[0], separator=";", null_values=["."],
                infer_schema_length=10000, 
                schema_overrides={"CD_SETOR": pl.Utf8, "CD_MUN": pl.Utf8, "CD_BAIRRO": pl.Utf8}
            ).filter(pl.col("CD_SETOR").str.starts_with("35")) # Filtra SP
            
            colunas_alvo = ["CD_SETOR", "CD_MUN", "NM_MUN", "CD_BAIRRO", "NM_BAIRRO", "AREA_KM2", "SITUACAO", "CD_SITUACAO", "CD_TIPO", "v0001", "v0002"]
            df_social = df_social.select([c for c in colunas_alvo if c in df_social.columns])

            # Tratamento da área (replace da vírgula brasileira para ponto)
            area_expr = pl.col("AREA_KM2").str.replace_all(",", ".").cast(pl.Float32, strict=False) if df_social.schema["AREA_KM2"] == pl.Utf8 else pl.col("AREA_KM2").cast(pl.Float32, strict=False)

            df_social = df_social.with_columns([
                area_expr,
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
            
            # Registo de Auditoria Seguro
            self.auditoria["CAMADAS"]["SOCIAL"] = {
                "STATUS": "SUCESSO",
                "SETORES_TOTAIS": df_social.height,
                "POPULACAO_MAPEADA": int(df_social.select(pl.col("TOTAL_PESSOAS").sum()).item()),
                "BAIRROS_MAPEADOS": df_social.select(pl.col("NM_BAIRRO").n_unique()).item()
            }
            
        except Exception as e:
            print(f"❌ Erro na Malha Social: {e}")
            self.auditoria["CAMADAS"]["SOCIAL"] = {"STATUS": "ERRO", "MENSAGEM": str(e)}

    # ---------------------------------------------------------
    # 3. MALHA DE INFRAESTRUTURA (Comércio / CNPJ)
    # ---------------------------------------------------------
    def processar_malha_infraestrutura(self):
        print("🏗️ [PRATA] A construir Malha de Infraestrutura (Comércio)...")
        try:
            pattern = os.path.join(self.bronze_dir, "**", "CNPJ_SP_HISTORICO_LOTE_*.parquet")
            arquivos = glob.glob(pattern, recursive=True)
            
            if not arquivos:
                raise FileNotFoundError("Nenhum lote de CNPJ encontrado.")

            df_infra = pl.read_parquet(arquivos)

            df_infra = df_infra.drop_nulls(subset=["lat", "lon"]).with_columns([
                pl.struct(["lat", "lon"]).map_batches(lambda s: pl.Series([
                    h3.latlng_to_cell(x["lat"], x["lon"], self.H3_RES) for x in s
                ])).alias("H3_INDEX")
            ])

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
            
            # Registo de Auditoria Seguro
            diversidade_cnae = df_infra.select(pl.col("cnae_fiscal_principal").n_unique()).item() if "cnae_fiscal_principal" in df_infra.columns else 0
            
            self.auditoria["CAMADAS"]["INFRAESTRUTURA"] = {
                "STATUS": "SUCESSO",
                "EMPRESAS_TOTAIS": df_infra.height,
                "HEXAGONOS_H3_COM_COMERCIO": df_infra.select(pl.col("H3_INDEX").n_unique()).item(),
                "DIVERSIDADE_CNAE": diversidade_cnae
            }
            
        except Exception as e:
            print(f"❌ Erro na Malha de Infraestrutura: {e}")
            self.auditoria["CAMADAS"]["INFRAESTRUTURA"] = {"STATUS": "ERRO", "MENSAGEM": str(e)}

    # ---------------------------------------------------------
    # 4. AUDITORIA E FINALIZAÇÃO
    # ---------------------------------------------------------
    def finalizar_auditoria(self):
        # Verifica se alguma camada teve erro para definir a qualidade geral
        erros = [v for k, v in self.auditoria["CAMADAS"].items() if v.get("STATUS") == "ERRO"]
        if erros:
            self.auditoria["QUALIDADE_GERAL"] = "SUCESSO_PARCIAL" if len(erros) < 3 else "FALHA_CRITICA"
        else:
            self.auditoria["QUALIDADE_GERAL"] = "SUCESSO_TOTAL"
            
        caminho_audit = os.path.join(self.prata_dir, "AUDITORIA_PRATA_MALHAS.json")
        with open(caminho_audit, "w", encoding="utf-8") as f:
            # O ensure_ascii=False garante que os acentos fiquem legíveis no JSON
            json.dump(self.auditoria, f, indent=4, ensure_ascii=False)
            
        print("\n" + "="*60)
        print("📊 AUDITORIA FINALIZADA (Copie o JSON abaixo e envie no chat):")
        print("="*60)
        print(json.dumps(self.auditoria, indent=4, ensure_ascii=False))
        print("="*60 + "\n")

    def executar(self):
        start = datetime.now()
        print("🚀 INICIANDO PROCESSAMENTO DE PRODUÇÃO (CAMADA PRATA)")
        
        self.processar_malha_geografica()
        self.processar_malha_social()
        self.processar_malha_infraestrutura()
        self.finalizar_auditoria()
            
        print(f"🏁 [TEMPO TOTAL]: {datetime.now() - start}")

if __name__ == "__main__":
    ArquitetoSafeDriverPrata().executar()
