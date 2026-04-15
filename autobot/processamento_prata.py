import polars as pl
import boto3
from botocore.config import Config
import io
import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
        # Credenciais Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        client_config = Config(
            signature_version='s3v4', 
            s3={'addressing_style': 'path'},
            max_pool_connections=50 
        )
        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=client_config)
        
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

    def _inicializar_dependencias(self):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            self.df_malha_lazy = pl.read_parquet(io.BytesIO(resp['Body'].read())).lazy()
            logger.info("PRATA: Malha geográfica H3-9 carregada.")
        except Exception as e:
            logger.error(f"PRATA: Erro ao carregar malha: {e}")
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

            # --- 1. CONVERSÃO E LIMPEZA INICIAL ---
            lf = lf.with_columns([
                pl.col("HORA").cast(pl.String).str.split(":").list.first().cast(pl.Int32, strict=False).alias("HORA_INT"),
                pl.col("MUNICIPIO").str.to_uppercase().str.strip_chars(),
                pl.col("BAIRRO").str.to_uppercase().str.strip_chars(),
                pl.col("PERIODO_TEXTO").cast(pl.String).str.to_uppercase().fill_null("")
            ])

            # --- 2. RECUPERAÇÃO CRUZADA (CURA) ---
            lf_enriquecido = lf.join(self.df_malha_lazy, on="H3_INDEX", how="left", suffix="_grid")

            lf_enriquecido = lf_enriquecido.with_columns([
                pl.coalesce([pl.col("NM_MUN"), pl.col("MUNICIPIO")]).alias("NM_MUN_FINAL"),
                pl.coalesce([pl.col("NM_BAIRRO"), pl.col("BAIRRO")]).alias("NM_BAIRRO_FINAL")
            ])

            # --- 3. FILTRAGEM DE QUALIDADE (EXCLUSÃO DE VAZIOS) ---
            # Aqui removemos as linhas que não puderam ser recuperadas
            count_antes = 0 # Placeholder para log se desejar
            lf_enriquecido = lf_enriquecido.filter(
                (pl.col("H3_INDEX").is_not_null()) & 
                (pl.col("NM_MUN_FINAL").is_not_null()) &
                (pl.col("NM_MUN_FINAL") != "") &
                (pl.col("NM_BAIRRO_FINAL").is_not_null()) &
                (pl.col("NM_BAIRRO_FINAL") != "")
            )

            # --- 4. LÓGICA DE NEGÓCIO ---
            lf_enriquecido = lf_enriquecido.with_columns([
                pl.when(pl.col("CONDUTA").str.contains("TRANSEUNTE|PEDESTRE")).then(pl.lit("PEDESTRE"))
                  .when(pl.col("RUBRICA").str.contains("VEICULO|AUTO|MOTO")).then(pl.lit("MOTORISTA"))
                  .otherwise(pl.lit("PEDESTRE")).alias("PERFIL_ALVO"),
                
                pl.when((pl.col("HORA_INT") > 0) & (pl.col("HORA_INT") < 6)).then(pl.lit("MADRUGADA"))
                  .when((pl.col("HORA_INT") >= 6) & (pl.col("HORA_INT") < 12)).then(pl.lit("MANHA"))
                  .when((pl.col("HORA_INT") >= 12) & (pl.col("HORA_INT") < 18)).then(pl.lit("TARDE"))
                  .when((pl.col("HORA_INT") >= 18) & (pl.col("HORA_INT") <= 23)).then(pl.lit("NOITE"))
                  .when(pl.col("PERIODO_TEXTO").str.contains(r"MADRUGADA|EM MADRUGADA")).then(pl.lit("MADRUGADA"))
                  .when(pl.col("PERIODO_TEXTO").str.contains(r"MANHA|MANHÃ")).then(pl.lit("MANHA"))
                  .when(pl.col("PERIODO_TEXTO").str.contains(r"TARDE")).then(pl.lit("TARDE"))
                  .when(pl.col("PERIODO_TEXTO").str.contains(r"NOITE")).then(pl.lit("NOITE"))
                  .otherwise(pl.lit("MADRUGADA")).alias("PERIODO_DIA")
            ])

            # --- 5. AGREGAÇÃO ---
            lf_agg = lf_enriquecido.group_by(["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO", "TIPO_LOCAL"]).agg([
                pl.when(pl.col("RUBRICA").str.contains("ROUBO")).then(3).otherwise(1).sum().alias("TOTAL_CRIMES"),
                pl.col("NM_MUN_FINAL").first().alias("NM_MUN"),
                pl.col("NM_BAIRRO_FINAL").first().alias("NM_BAIRRO"),
                pl.col("DENSIDADE_AJUSTADA").cast(pl.Float64).first().alias("DENSIDADE"),
                pl.col("TAXA_VACANCIA").cast(pl.Float64).first().alias("TAXA_VACANCIA")
            ])

            lf_final = lf_agg.with_columns([
                (pl.col("TOTAL_CRIMES").rank().over("PERIODO_DIA") / pl.col("TOTAL_CRIMES").count().over("PERIODO_DIA")).alias("RANKING_RISCO_LOCAL"),
                (pl.col("TOTAL_CRIMES") / (pl.col("DENSIDADE").fill_null(0) + 1)).alias("INDICE_EXPOSICAO"),
                pl.lit(ano).alias("ANO_REFERENCIA")
            ])

            # Execução e Salvamento
            df_final = lf_final.collect(engine="streaming")
            buffer = io.BytesIO()
            df_final.write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            estado[str(ano)] = tamanho_atual
            logger.info(f"PRATA: [{ano}] Sucesso. Linhas purificadas enviadas para o R2.")
            return True
        except Exception as e:
            logger.error(f"PRATA: Erro no ano {ano}: {e}")
            return False

    def executar_todos_os_anos(self, force=False):
        if self.df_malha_lazy is None: return
        estado = self._carregar_tracker()
        for ano in range(2022, datetime.now().year + 1):
            if self.processar_ano_com_delta(ano, estado, force):
                self._salvar_tracker(estado)

    def _carregar_tracker(self):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.tracker_path)
            return json.loads(resp['Body'].read())
        except: return {}

    def _salvar_tracker(self, estado):
        self.s3.put_object(Bucket=self.bucket, Key=self.tracker_path, Body=json.dumps(estado))

if __name__ == "__main__":
    prata = ProcessamentoPrata()
    prata.executar_todos_os_anos(force=True)
