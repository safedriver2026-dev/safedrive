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
        # Configurações Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}))
        
        # Localização dinâmica do Data Lake (Resolve o problema de aninhamento)
        self.base_path = self._localizar_datalake_real()
        logger.info(f"PRATA: Data Lake mestre detectado em: '{self.base_path}'")

        # Caminhos Dinâmicos
        self.tracker_path = f"{self.base_path}/prata/tracker_estado_bronze.json".replace("//", "/")
        self.malha_path = f"{self.base_path}/base_geografica/safedriver_geo_base_sp_h3_9.parquet".replace("//", "/")
        
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

    def _classificar_perfil_alvo(self, rubrica, conduta):
        """Diferencia MOTORISTA de PEDESTRE com base na rubrica e conduta (2022-2026)."""
        rub = str(rubrica).upper()
        con = str(conduta).upper() if conduta else ""
        if "TRANSEUNTE" in con or "PEDESTRE" in con or "PASSAGEIRO" in con:
            return "PEDESTRE"
        if any(x in rub for x in ["VEICULO", "CARGA", "AUTO", "MOTO", "CAMINHAO", "CONDUZIR"]):
            return "MOTORISTA"
        return "PEDESTRE"

    def _definir_periodo(self, hora, desc_periodo):
        """Normaliza o período lidando com variações de colunas (2022-2026)."""
        if hora and str(hora) != 'nan' and ":" in str(hora):
            try:
                h = int(str(hora).split(':')[0])
                if 0 <= h < 6: return "MADRUGADA"
                if 6 <= h < 12: return "MANHA"
                if 12 <= h < 18: return "TARDE"
                return "NOITE"
            except: pass
        
        desc = str(desc_periodo).upper() if desc_periodo else ""
        if "MADRUGADA" in desc: return "MADRUGADA"
        if "MANHÃ" in desc or "MANHA" in desc: return "MANHA"
        if "TARDE" in desc: return "TARDE"
        if "NOITE" in desc: return "NOITE"
        return "INDEFINIDO"

    def _inicializar_dependencias(self):
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
            
            # Normalização Dinâmica de Colunas (Resiliência 2022-2026)
            cols = df.columns
            df = df.rename({
                "CIDADE": "MUNICIPIO" if "CIDADE" in cols else None,
                "NOME_MUNICIPIO": "MUNICIPIO" if "NOME_MUNICIPIO" in cols else None,
                "DESCR_PERIODO": "PERIODO_TEXTO" if "DESCR_PERIODO" in cols else None,
                "DESC_PERIODO": "PERIODO_TEXTO" if "DESC_PERIODO" in cols else None,
                "HORA_OCORRENCIA_BO": "HORA" if "HORA_OCORRENCIA_BO" in cols else None,
                "DESCR_CONDUTA": "CONDUTA" if "DESCR_CONDUTA" in cols else None,
                "DESCR_TIPOLOCAL": "TIPO_LOCAL" if "DESCR_TIPOLOCAL" in cols else None
            })

            # 1. ENGENHARIA DE FEATURES (PERFIL, PERÍODO E TIPO DE LOCAL)
            df = df.with_columns([
                pl.struct(["RUBRICA", "CONDUTA" if "CONDUTA" in df.columns else "RUBRICA"])
                  .map_elements(lambda x: self._classificar_perfil_alvo(x["RUBRICA"], x.get("CONDUTA")), return_dtype=pl.String)
                  .alias("PERFIL_ALVO"),
                
                pl.struct(["HORA" if "HORA" in df.columns else "RUBRICA", 
                           "PERIODO_TEXTO" if "PERIODO_TEXTO" in df.columns else "RUBRICA"])
                  .map_elements(lambda x: self._definir_periodo(x.get("HORA"), x.get("PERIODO_TEXTO")), return_dtype=pl.String)
                  .alias("PERIODO_DIA"),
                
                pl.col("TIPO_LOCAL").fill_null("VIA PUBLICA").alias("TIPO_LOCAL") if "TIPO_LOCAL" in df.columns else pl.lit("VIA PUBLICA").alias("TIPO_LOCAL")
            ])

            # 2. ENRIQUECIMENTO (Join com Malha Demográfica)
            df_enriquecido = df.join(self.df_malha, on="H3_INDEX", how="left")

            # 3. AGREGAÇÃO PONDERADA (Severidade: Roubo=3, Furto=1)
            df_agg = df_enriquecido.group_by(["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO", "TIPO_LOCAL"]).agg([
                pl.when(pl.col("RUBRICA").str.contains("ROUBO")).then(3).otherwise(1).sum().alias("TOTAL_CRIMES"),
                pl.col("MUNICIPIO").first().alias("NM_MUN"),
                pl.col("NM_BAIRRO").first(),
                pl.col("DENSIDADE_DEMOGRAFICA").first().alias("DENSIDADE"),
                pl.col("TAXA_VACANCIA").first().alias("TAXA_VACANCIA")
            ])

            # 4. FEATURE STORE (PREPARAÇÃO PARA IA E OURO)
            # Ranking de Risco Local (Percentil por Período) e Índice de Exposição
            df_final = df_agg.with_columns([
                (pl.col("TOTAL_CRIMES").rank(descending=False).over("PERIODO_DIA") / pl.col("TOTAL_CRIMES").count().over("PERIODO_DIA")).alias("RANKING_RISCO_LOCAL"),
                (pl.col("TOTAL_CRIMES") / (pl.col("DENSIDADE").fill_null(0) + 1)).alias("INDICE_EXPOSICAO"),
                pl.lit(ano).alias("ANO_REFERENCIA")
            ])

            # 5. SALVAMENTO
            buffer = io.BytesIO()
            df_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            estado[str(ano)] = tamanho_atual
            logger.info(f"PRATA: [{ano}] Consolidado com Analytics: {df_final.height} linhas.")
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
