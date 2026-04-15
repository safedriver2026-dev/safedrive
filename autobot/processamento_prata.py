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

        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}))
        
        # Localização dinâmica do Data Lake (Resolve aninhamentos safedriver/safedriver)
        self.base_path = self._localizar_datalake_real()
        logger.info(f"PRATA: Data Lake mestre detectado em: '{self.base_path}'")

        # Caminhos Dinâmicos
        self.tracker_path = f"{self.base_path}/prata/tracker_estado_bronze.json".replace("//", "/")
        self.malha_path = f"{self.base_path}/base_geografica/safedriver_geo_base_sp_h3_9.parquet".replace("//", "/")
        
        self._inicializar_dependencias()

    def _localizar_datalake_real(self):
        """Busca o prefixo correto da pasta datalake, ignorando aninhamentos."""
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, MaxKeys=100):
                for obj in page.get('Contents', []):
                    if "datalake/bronze/trusted/" in obj['Key']:
                        return obj['Key'].split("datalake/")[0] + "datalake"
            return "datalake"
        except: return "datalake"

    def _inicializar_dependencias(self):
        """Carrega a malha geográfica H3 (Censo)."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            self.df_malha = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            logger.info("PRATA: Malha geográfica carregada.")
        except Exception as e:
            logger.error(f"PRATA: Erro ao carregar malha: {e}")
            self.df_malha = None

    def processar_ano_com_delta(self, ano, estado, force=False):
        path_trusted = f"{self.base_path}/bronze/trusted/ssp_trusted_{ano}.parquet".replace("//", "/")
        path_prata = f"{self.base_path}/prata/ssp_consolidada_{ano}.parquet".replace("//", "/")
        
        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            tamanho_atual = meta['ContentLength']
            if not force and estado.get(str(ano)) == tamanho_atual: return None

            resp = self.s3.get_object(Bucket=self.bucket, Key=path_trusted)
            df = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            
            # --- 1. MAPEAMENTO DINÂMICO DE COLUNAS (Evita Erro NoneType) ---
            cols = df.columns
            mapeamento = {
                "CIDADE": "MUNICIPIO", "NOME_MUNICIPIO": "MUNICIPIO",
                "DESCR_PERIODO": "PERIODO_TEXTO", "DESC_PERIODO": "PERIODO_TEXTO",
                "HORA_OCORRENCIA_BO": "HORA", "DESCR_CONDUTA": "CONDUTA",
                "DESCR_TIPOLOCAL": "TIPO_LOCAL"
            }
            rename_dict = {old: new for old, new in mapeamento.items() if old in cols}
            if rename_dict:
                df = df.rename(rename_dict)

            # --- 2. ENGENHARIA DE FEATURES (100% NATIVA POLARS - ULTRAVELOZ) ---
            # Prepara as colunas para evitar erros caso o ano não as possua
            conduta_col = pl.col("CONDUTA").cast(pl.String).fill_null("") if "CONDUTA" in df.columns else pl.lit("")
            rubrica_col = pl.col("RUBRICA").cast(pl.String).fill_null("")
            hora_col = pl.col("HORA").cast(pl.String).fill_null("") if "HORA" in df.columns else pl.lit("")
            periodo_txt_col = pl.col("PERIODO_TEXTO").cast(pl.String).fill_null("").str.to_uppercase() if "PERIODO_TEXTO" in df.columns else pl.lit("")

            # Tenta extrair a hora como número inteiro nativamente
            hora_int = hora_col.str.split(":").list.first().cast(pl.Int32, strict=False)

            df = df.with_columns([
                # Inteligência: Perfil do Alvo (Vetorizado)
                pl.when(conduta_col.str.to_uppercase().str.contains("TRANSEUNTE|PEDESTRE|PASSAGEIRO"))
                  .then(pl.lit("PEDESTRE"))
                  .when(rubrica_col.str.to_uppercase().str.contains("VEICULO|CARGA|AUTO|MOTO|CAMINHAO|CONDUZIR"))
                  .then(pl.lit("MOTORISTA"))
                  .otherwise(pl.lit("PEDESTRE"))
                  .alias("PERFIL_ALVO"),
                
                # Inteligência: Período do Dia (Vetorizado)
                pl.when((hora_int >= 0) & (hora_int < 6)).then(pl.lit("MADRUGADA"))
                  .when((hora_int >= 6) & (hora_int < 12)).then(pl.lit("MANHA"))
                  .when((hora_int >= 12) & (hora_int < 18)).then(pl.lit("TARDE"))
                  .when(hora_int >= 18).then(pl.lit("NOITE"))
                  # Fallback para a coluna de texto caso a hora seja nula
                  .when(periodo_txt_col.str.contains("MADRUGADA")).then(pl.lit("MADRUGADA"))
                  .when(periodo_txt_col.str.contains("MANHÃ|MANHA")).then(pl.lit("MANHA"))
                  .when(periodo_txt_col.str.contains("TARDE")).then(pl.lit("TARDE"))
                  .when(periodo_txt_col.str.contains("NOITE")).then(pl.lit("NOITE"))
                  .otherwise(pl.lit("INDEFINIDO"))
                  .alias("PERIODO_DIA"),
                
                # Inteligência: Tipo de Local
                pl.col("TIPO_LOCAL").fill_null("VIA PUBLICA").alias("TIPO_LOCAL") if "TIPO_LOCAL" in df.columns else pl.lit("VIA PUBLICA").alias("TIPO_LOCAL")
            ])

            # --- 3. JOIN COM MALHA (H3 Geográfico) ---
            df_enriquecido = df.join(self.df_malha, on="H3_INDEX", how="left")

            # --- 4. AGREGAÇÃO PONDERADA (Matriz de Severidade: Roubo=3, Furto=1) ---
            df_agg = df_enriquecido.group_by(["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO", "TIPO_LOCAL"]).agg([
                pl.when(pl.col("RUBRICA").str.contains("ROUBO")).then(3).otherwise(1).sum().alias("TOTAL_CRIMES"),
                pl.col("MUNICIPIO").first().alias("NM_MUN"),
                pl.col("NM_BAIRRO").first(),
                pl.col("DENSIDADE_DEMOGRAFICA").first().alias("DENSIDADE"),
                pl.col("TAXA_VACANCIA").first().alias("TAXA_VACANCIA")
            ])

            # --- 5. ANALYTICS (Feature Store para a IA e Ouro) ---
            df_final = df_agg.with_columns([
                (pl.col("TOTAL_CRIMES").rank(descending=False).over("PERIODO_DIA") / pl.col("TOTAL_CRIMES").count().over("PERIODO_DIA")).alias("RANKING_RISCO_LOCAL"),
                (pl.col("TOTAL_CRIMES") / (pl.col("DENSIDADE").fill_null(0) + 1)).alias("INDICE_EXPOSICAO"),
                pl.lit(ano).alias("ANO_REFERENCIA")
            ])

            # --- 6. SALVAMENTO ---
            buffer = io.BytesIO()
            df_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            estado[str(ano)] = tamanho_atual
            logger.info(f"PRATA: [{ano}] Sucesso - {df_final.height} registros (Processamento 100% Nativo).")
            return True
        except Exception as e:
            logger.error(f"PRATA: Falha no ano {ano}: {e}")
            return False

    def executar_todos_os_anos(self, force=False):
        if self.df_malha is None: return
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
    Prata = ProcessamentoPrata()
    Prata.executar_todos_os_anos(force=True)
