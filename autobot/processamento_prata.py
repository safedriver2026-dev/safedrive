import polars as pl
import pandas as pd
import h3
import boto3
from botocore.config import Config
import io
import os
import json
import logging
from datetime import datetime

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}, max_pool_connections=50))
        
        self.base_path = self._localizar_datalake_real()
        self.tracker_path = f"{self.base_path}/prata/tracker_estado_bronze.json"
        self.malha_path = f"{self.base_path}/base_geografica/safedriver_geo_base_sp_h3_9.parquet"
        
        self._inicializar_dependencias()

    def _localizar_datalake_real(self):
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, MaxKeys=100):
                for obj in page.get('Contents', []):
                    if "datalake/bronze/trusted/" in obj['Key']:
                        return obj['Key'].split("datalake/")[0] + "datalake"
            return "datalake"
        except: return "datalake"

    def _limpar_texto_extremo(self, coluna):
        return (
            pl.col(coluna)
            .cast(pl.String)
            .str.to_uppercase()
            .str.replace_all(r"[ÁÀÂÃÄ]", "A")
            .str.replace_all(r"[ÉÈÊË]", "E")
            .str.replace_all(r"[ÍÌÎÏ]", "I")
            .str.replace_all(r"[ÓÒÔÕÖ]", "O")
            .str.replace_all(r"[ÚÙÛÜ]", "U")
            .str.replace_all(r"[Ç]", "C")
            .str.replace_all(r"[Ñ]", "N")
            .str.strip_chars()
            .fill_null("INDEFINIDO")
        )

    def _inicializar_dependencias(self):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            
            # 🛡️ DATA CONTRACT: A CURA DA EXPLOSÃO CARTESIANA (DEDUPLICAÇÃO H3)
            # Downcasting na malha: Float64 para Float32 (50% menos RAM)
            self.df_malha = pl.read_parquet(io.BytesIO(resp['Body'].read())).unique(subset=["H3_INDEX"]).with_columns([
                pl.col("DENSIDADE_AJUSTADA").cast(pl.Float32),
                pl.col("TAXA_VACANCIA").cast(pl.Float32)
            ])
            self.df_malha_lazy = self.df_malha.lazy().with_columns([
                self._limpar_texto_extremo("NM_MUN").alias("NM_MUN"),
                self._limpar_texto_extremo("NM_BAIRRO").alias("NM_BAIRRO")
            ])
            logger.info("PRATA: Malha geográfica base carregada, deduplicada e comprimida.")
        except Exception as e:
            logger.error(f"PRATA: Falha crítica na malha: {e}")
            self.df_malha_lazy = None

    def processar_ano_com_delta(self, ano, estado, force=False):
        path_trusted = f"{self.base_path}/bronze/trusted/ssp_trusted_{ano}.parquet"
        path_prata = f"{self.base_path}/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            tamanho_atual = meta['ContentLength']
            if not force and estado.get(str(ano)) == tamanho_atual: return None

            resp = self.s3.get_object(Bucket=self.bucket, Key=path_trusted)
            lf = pl.read_parquet(io.BytesIO(resp['Body'].read())).lazy()

            # 1. Alias Machine (Busca dinâmica de colunas para 2022-2026)
            cols = lf.collect_schema().names()
            rename_map = {}
            drop_cols = []

            mun_col = next((c for c in cols if c in ["CIDADE", "MUNICIPIO", "NOME_MUNICIPIO"]), None)
            if mun_col: rename_map[mun_col] = "NM_MUN_ORIGINAL"

            data_col = next((c for c in cols if c in ["DATA_OCORRENCIA_BO", "DATA_OCORRENCIA"]), None)
            if data_col: rename_map[data_col] = "DATA_BRUTA"

            if "BAIRRO" in cols: rename_map["BAIRRO"] = "NM_BAIRRO_ORIGINAL"
            
            hora_col = next((c for c in cols if c in ["HORA_OCORRENCIA_BO", "HORA"]), None)
            if hora_col: rename_map[hora_col] = "HORA"

            if "DESCR_SUBTIPOLOCAL" in cols: 
                rename_map["DESCR_SUBTIPOLOCAL"] = "TIPO_LOCAL"
                if "DESCR_TIPOLOCAL" in cols: drop_cols.append("DESCR_TIPOLOCAL")
            elif "DESCR_TIPOLOCAL" in cols: 
                rename_map["DESCR_TIPOLOCAL"] = "TIPO_LOCAL"

            lf = lf.rename(rename_map).filter(pl.col("H3_INDEX").is_not_null())
            if drop_cols: lf = lf.drop(drop_cols)

            # 2. Engenharia de Contexto
            lf = lf.with_columns([
                pl.col("DATA_BRUTA").str.to_date(format="%d/%m/%Y", strict=False).alias("DATA"),
                pl.col("HORA").str.split(":").list.first().cast(pl.Int8, strict=False).alias("HORA_INT")
            ]).with_columns([
                pl.col("DATA").dt.month().cast(pl.Int8).alias("MES_OCORRENCIA"),
                pl.col("DATA").dt.weekday().cast(pl.Int8).alias("DIA_SEMANA"),
                pl.when(pl.col("HORA_INT") < 6).then(pl.lit("MADRUGADA"))
                  .when(pl.col("HORA_INT") < 12).then(pl.lit("MANHA"))
                  .when(pl.col("HORA_INT") < 18).then(pl.lit("TARDE"))
                  .otherwise(pl.lit("NOITE")).alias("PERIODO_DIA"),
                pl.when(pl.col("RUBRICA").str.contains("(?i)VEICULO|AUTO|MOTO")).then(pl.lit("MOTORISTA"))
                  .otherwise(pl.lit("PEDESTRE")).alias("PERFIL_ALVO"),
                # Tags temporais históricas para a IA aprender:
                pl.when(pl.col("DATA").dt.day().is_between(5, 10)).then(1).otherwise(0).cast(pl.Int8).alias("IS_PAGAMENTO"),
                pl.when(pl.col("DATA").dt.weekday() >= 6).then(1).otherwise(0).cast(pl.Int8).alias("IS_FDS"),
                
                # ⚖️ A MATRIZ DE GRAVIDADE (CRIME WEIGHTING)
                pl.when(pl.col("RUBRICA").str.contains("(?i)HOMICIDIO|LATROCINIO|ESTUPRO|SEQUESTRO|MORTE"))
                  .then(pl.lit(10.0)) # Risco Máximo à Vida
                  .when(pl.col("RUBRICA").str.contains("(?i)ROUBO"))
                  .then(pl.lit(5.0))  # Risco com Violência/Ameaça
                  .when(pl.col("RUBRICA").str.contains("(?i)VEICULO|AUTO|MOTO"))
                  .then(pl.lit(3.0))  # Risco Patrimonial Elevado
                  .otherwise(pl.lit(1.0)) # Furto simples e outras ocorrências menores
                  .alias("PESO_CRIME")
            ])

            # 3. Agregação com Foco no Dano (Gravidade)
            lf_agg = lf.group_by([
                "H3_INDEX", "NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL", 
                "MES_OCORRENCIA", "DIA_SEMANA", "PERIODO_DIA", "PERFIL_ALVO", "TIPO_LOCAL", 
                "IS_PAGAMENTO", "IS_FDS"
            ]).agg([
                pl.len().cast(pl.Int32).alias("TOTAL_CRIMES"),
                pl.col("PESO_CRIME").sum().cast(pl.Float32).alias("INDICE_GRAVIDADE") # O Novo Target da IA
            ])

            # 4. Enriquecimento Geográfico
            lf_enriquecido = lf_agg.join(self.df_malha_lazy, on="H3_INDEX", how="left").with_columns([
                pl.coalesce([pl.col("NM_MUN"), pl.col("NM_MUN_ORIGINAL")]).alias("NM_MUN"),
                pl.coalesce([pl.col("NM_BAIRRO"), pl.col("NM_BAIRRO_ORIGINAL")]).alias("NM_BAIRRO"),
                pl.col("DENSIDADE_AJUSTADA").cast(pl.Float32).alias("DENSIDADE"),
                pl.lit(ano).cast(pl.Int16).alias("ANO_REF") # Prevenção de Data Leakage
            ])

            # 5. Cálculo Final de Exposição ao Risco (Baseado na Gravidade)
            df_final = lf_enriquecido.with_columns([
                # O índice de exposição agora mede o impacto real (Gravidade) vs Densidade
                (pl.col("INDICE_GRAVIDADE") / (pl.col("DENSIDADE") + 1)).cast(pl.Float32).alias("INDICE_EXPOSICAO"),
                (pl.col("INDICE_GRAVIDADE").rank().over("PERIODO_DIA") / pl.col("INDICE_GRAVIDADE").count().over("PERIODO_DIA")).cast(pl.Float32).alias("RANKING_RISCO_LOCAL")
            ]).drop(["NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL"])

            # 6. Gravação Otimizada
            buffer = io.BytesIO()
            df_final.collect().write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            estado[str(ano)] = tamanho_atual 
            logger.info(f"PRATA: [{ano}] Consolidado com sucesso. Peso processado para Matriz de Gravidade.")
            return True
        except Exception as e:
            logger.error(f"PRATA: Erro no ano {ano}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def executar_todos_os_anos(self, force=False):
        logger.info(f"PRATA: Iniciando consolidação Heavy Silver (Force={force}).")
        estado = self._carregar_tracker()
        for ano in range(2022, datetime.now().year + 1):
            if self.processar_ano_com_delta(ano, estado, force):
                self._salvar_tracker(estado)
        return {"status": "Prata Completa"}

    def _carregar_tracker(self):
        try:
            return json.loads(self.s3.get_object(Bucket=self.bucket, Key=self.tracker_path)['Body'].read())
        except: return {}

    def _salvar_tracker(self, estado):
        self.s3.put_object(Bucket=self.bucket, Key=self.tracker_path, Body=json.dumps(estado))

if __name__ == "__main__":
    prata = ProcessamentoPrata()
    prata.executar_todos_os_anos(force=True) # Force=True para reprocessar a Matriz de Gravidade
