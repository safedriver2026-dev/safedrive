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
        # Configurações do R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}))
        
        # --- LÓGICA DE LOCALIZAÇÃO AUTOMÁTICA (Igual à Bronze) ---
        self.base_path = self._descobrir_prefixo_datalake()
        logger.info(f"PRATA: Raiz do Data Lake detectada em: '{self.base_path}'")

        # Caminhos Dinâmicos
        self.tracker_path = f"{self.base_path}/prata/tracker_estado_bronze.json".replace("//", "/")
        self.malha_path = f"{self.base_path}/base_geografica/safedriver_geo_base_sp_h3_9.parquet".replace("//", "/")
        
        self._inicializar_dependencias()

    def _descobrir_prefixo_datalake(self):
        """Procura dinamicamente se a pasta 'datalake' está na raiz ou dentro de 'safedriver/'."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, MaxKeys=15)
            if 'Contents' in response:
                keys = [obj['Key'] for obj in response['Contents']]
                for key in keys:
                    if "safedriver/datalake" in key: return "safedriver/datalake"
                    if "datalake" in key: return "datalake"
            return "datalake"
        except:
            return "datalake"

    def _inicializar_dependencias(self):
        """Carrega a malha geográfica e prepara métricas de densidade."""
        try:
            logger.info(f"PRATA: Carregando malha geográfica de {self.malha_path}...")
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            self.df_malha = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            
            # Garantia de Bairro e Densidade
            if "NM_BAIRRO" not in self.df_malha.columns:
                self.df_malha = self.df_malha.with_columns(pl.lit("DESCONHECIDO").alias("NM_BAIRRO"))
            
            logger.info("PRATA: Malha geográfica carregada e pronta.")
        except Exception as e:
            logger.error(f"PRATA: Erro ao carregar malha no R2: {e}")
            self.df_malha = None

    def _motor_h3(self, lat, lng):
        """Converte coordenadas para H3 Resolução 9."""
        try:
            if lat == 0.0 or lng == 0.0: return None
            return h3.latlng_to_cell(float(lat), float(lng), 9)
        except:
            return None

    def processar_ano_com_delta(self, ano, estado, force=False):
        """Processa um ano específico se houver mudança no Trusted."""
        path_trusted = f"{self.base_path}/bronze/trusted/ssp_trusted_{ano}.parquet".replace("//", "/")
        path_prata = f"{self.base_path}/prata/ssp_consolidada_{ano}.parquet".replace("//", "/")
        
        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            tamanho_atual = meta['ContentLength']
        except:
            logger.warning(f"PRATA: Arquivo Trusted de {ano} não encontrado. Pulando.")
            return None

        # Verificação de Delta (Idempotência)
        if not force and estado.get(str(ano)) == tamanho_atual:
            return None

        logger.info(f"PRATA: Consolidando {ano} (H3 + Georeferenciamento)...")
        
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_trusted)
            df = pl.read_parquet(io.BytesIO(resp['Body'].read()))

            # 1. Limpeza e H3
            df = df.with_columns([
                pl.col("LATITUDE").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("LONGITUDE").cast(pl.Float64, strict=False).fill_null(0.0)
            ]).with_columns(
                pl.struct(["LATITUDE", "LONGITUDE"])
                .map_elements(lambda x: self._motor_h3(x["LATITUDE"], x["LONGITUDE"]), return_dtype=pl.String)
                .alias("H3_INDEX")
            ).filter(pl.col("H3_INDEX").is_not_null())

            # 2. Enriquecimento com Malha (Bairro, Densidade, Vacância)
            cols_malha = ["H3_INDEX", "DENSIDADE_DEMOGRAFICA", "TAXA_VACANCIA", "NM_BAIRRO"]
            df_enriquecido = df.join(
                self.df_malha.select([c for c in cols_malha if c in self.df_malha.columns]),
                on="H3_INDEX", 
                how="left"
            )

            # 3. Agregação por Hexágono (Foco em Roubo e Furto)
            df_final = df_enriquecido.group_by(["H3_INDEX"]).agg([
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("ROUBO|FURTO")).count().alias("TOTAL_CRIMES"),
                pl.col("NM_BAIRRO").first(),
                pl.col("MUNICIPIO").first().alias("NM_MUN"),
                pl.col("DENSIDADE_DEMOGRAFICA").first().alias("DENSIDADE"),
                pl.col("TAXA_VACANCIA").first().alias("TAXA_VACANCIA")
            ]).with_columns(pl.lit(ano).alias("ANO_REFERENCIA"))

            # 4. Salvamento
            buffer = io.BytesIO()
            df_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            estado[str(ano)] = tamanho_atual
            return {"ano": ano, "rows": df_final.height}

        except Exception as e:
            logger.error(f"PRATA: Falha no ano {ano}: {e}")
            return None

    def executar_todos_os_anos(self, force=False):
        if self.df_malha is None: return []
        
        estado = self._carregar_tracker()
        relatorio = []
        
        for ano in range(2022, datetime.now().year + 1):
            res = self.processar_ano_com_delta(ano, estado, force)
            if res:
                relatorio.append(res)
                self._salvar_tracker(estado)
        
        return relatorio

    def _carregar_tracker(self):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.tracker_path)
            return json.loads(resp['Body'].read())
        except: return {}

    def _salvar_tracker(self, estado):
        self.s3.put_object(Bucket=self.bucket, Key=self.tracker_path, Body=json.dumps(estado))
