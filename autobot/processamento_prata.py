import polars as pl
import h3
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import io
import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        
        self.tracker_path = "datalake/prata/tracker_estado_bronze.json"
        self.malha_path = "datalake/base_geografica/safedriver_geo_base_sp_h3_9.parquet"
        self._inicializar_dependencias()

    def _inicializar_dependencias(self):
        """Carrega a malha e prepara métricas de densidade para fallback."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            self.df_malha = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            
            # Garantia de coluna de Bairro para a IA
            if "NM_BAIRRO" not in self.df_malha.columns:
                self.df_malha = self.df_malha.with_columns(pl.lit("DESCONHECIDO").alias("NM_BAIRRO"))
            
            # Chaves de agrupamento dinâmicas
            chaves_distrito = ["NM_MUN", "NM_DIST"] if "NM_DIST" in self.df_malha.columns else ["NM_MUN"]
            col_densidade = "DENSIDADE_AJUSTADA" if "DENSIDADE_AJUSTADA" in self.df_malha.columns else "DENSIDADE_DEMOGRAFICA"

            self.dens_distrito = self.df_malha.group_by(chaves_distrito).agg(
                pl.col(col_densidade).mean().alias("DENS_DIST")
            )
            self.dens_cidade = self.df_malha.group_by(["NM_MUN"]).agg(
                pl.col(col_densidade).mean().alias("DENS_CID")
            )
            logger.info("PRATA: Infraestrutura geográfica pronta para processamento colunar.")
        except Exception as e:
            logger.error(f"PRATA: Falha crítica ao carregar malha no R2: {e}")
            self.df_malha = None

    def _canonizar_texto(self, coluna):
        """Limpeza e padronização de strings para junção espacial."""
        return (
            pl.when(coluna.is_not_null())
            .then(
                coluna.cast(pl.String).str.to_uppercase()
                .str.replace(r"^(RUA|R|AVENIDA|AV|ALAMEDA|AL|PRACA|PRC|ESTRADA|EST|VIELA|VL)\.?\s+", "")
                .str.replace_all(r"[ÁÀÂÃÄ]", "A").str.replace_all(r"[ÉÈÊË]", "E")
                .str.replace_all(r"[ÍÌÎÏ]", "I").str.replace_all(r"[ÓÒÔÕÖ]", "O")
                .str.replace_all(r"[ÚÙÛÜ]", "U").str.replace_all(r"[Ç]", "C")
                .str.replace_all(r"[^A-Z0-9 ]", "")
                .str.replace_all(r"\s+", " ")
                .str.strip_chars()
            )
            .otherwise(pl.lit("DESCONHECIDO"))
        )

    def _motor_recuperacao_cruzada(self, df_bronze_trusted):
        """Motor de geocodificação interna e enriquecimento de malha."""
        # 1. Padronização
        df_ssp = df_bronze_trusted.with_columns([
            pl.col("LATITUDE").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("LONGITUDE").cast(pl.Float64, strict=False).fill_null(0.0),
            self._canonizar_texto(pl.col("LOGRADOURO")).alias("LOG_CANONICO"),
            self._canonizar_texto(pl.col("MUNICIPIO")).alias("MUN_CANONICO")
        ])

        # 2. Divisão para geocodificação
        df_gps = df_ssp.filter((pl.col("LATITUDE") != 0.0) & (pl.col("LONGITUDE") != 0.0)).with_columns(
            pl.struct(["LATITUDE", "LONGITUDE"])
            .map_elements(lambda x: h3.latlng_to_cell(x["LATITUDE"], x["LONGITUDE"], 9), return_dtype=pl.String)
            .alias("H3_INDEX")
        )
        df_cegos = df_ssp.filter((pl.col("LATITUDE") == 0.0) | (pl.col("LONGITUDE") == 0.0))

        # 3. Resgate de BOs sem GPS via Logradouro na Malha
        malha_lookup = self.df_malha.with_columns([
            self._canonizar_texto(pl.col("LOGRADOURO")).alias("LOG_CANONICO"),
            self._canonizar_texto(pl.col("NM_MUN")).alias("MUN_CANONICO")
        ]).unique(subset=["MUN_CANONICO", "LOG_CANONICO"]).select(["MUN_CANONICO", "LOG_CANONICO", "H3_INDEX"])

        df_resgatados = df_cegos.join(malha_lookup, on=["MUN_CANONICO", "LOG_CANONICO"], how="inner")
        df_unificado = pl.concat([df_gps, df_resgatados], how="diagonal")

        # 4. Cruzamento com Features Demográficas
        cols_geo = ["H3_INDEX", "DENSIDADE_AJUSTADA", "DENSIDADE_DEMOGRAFICA", "TAXA_VACANCIA", "NM_BAIRRO"]
        malha_feat = self.df_malha.select([c for c in cols_geo if c in self.df_malha.columns]).unique(subset=["H3_INDEX"])
        
        df_final = df_unificado.join(malha_feat, on="H3_INDEX", how="left")

        return df_final, df_resgatados.height

    def processar_ano_com_delta(self, ano, estado, force=False):
        path_trusted = f"datalake/bronze/trusted/ssp_trusted_{ano}.parquet"
        path_prata = f"datalake/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            tamanho = meta['ContentLength']
        except: return None

        if not force and estado.get(str(ano)) == tamanho: 
            logger.info(f"PRATA: Ano {ano} ja consolidado. Pulando.")
            return None

        try:
            logger.info(f"PRATA: Processando {ano} via motor Parquet (Trusted)...")
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_trusted)
            df_trusted = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            
            total_raw = df_trusted.height
            df_curado, resgate_h3 = self._motor_recuperacao_cruzada(df_trusted)

            # Agregação final por Hexágono
            df_agg = df_curado.group_by(["H3_INDEX"]).agg([
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("ROUBO|FURTO")).count().alias("TOTAL_CRIMES"),
                pl.col("NM_BAIRRO").first(),
                pl.col("MUN_CANONICO").first().alias("NM_MUN"),
                pl.col("DENSIDADE_AJUSTADA").first().alias("DENSIDADE") if "DENSIDADE_AJUSTADA" in df_curado.columns else pl.col("DENSIDADE_DEMOGRAFICA").first().alias("DENSIDADE"),
                pl.col("TAXA_VACANCIA").first().alias("TAXA_VACANCIA") if "TAXA_VACANCIA" in df_curado.columns else pl.lit(0.0).alias("TAXA_VACANCIA")
            ]).with_columns(pl.lit(ano).cast(pl.Int32).alias("ANO_REFERENCIA"))

            # Persistência
            buffer = io.BytesIO()
            df_agg.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            estado[str(ano)] = tamanho
            return {"ano": ano, "total_raw": total_raw, "bo_recuperados": resgate_h3, "h3_unicos": df_agg.height}
        except Exception as e:
            logger.error(f"PRATA: Erro ao consolidar ano {ano}: {e}")
            return None

    def executar_todos_os_anos(self, force=False):
        if self.df_malha is None:
            raise RuntimeError("Execução abortada: Malha geografica inacessivel.")

        estado_atual = self._carregar_tracker()
        relatorio = []
        
        for ano in range(2022, datetime.now().year + 1):
            res = self.processar_ano_com_delta(ano, estado_atual, force)
            if res:
                self._salvar_tracker(estado_atual)
                relatorio.append(res)
        
        return relatorio

    def _carregar_tracker(self):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.tracker_path)
            return json.loads(resp['Body'].read())
        except ClientError: return {}

    def _salvar_tracker(self, estado):
        self.s3.put_object(Bucket=self.bucket, Key=self.tracker_path, Body=json.dumps(estado))
