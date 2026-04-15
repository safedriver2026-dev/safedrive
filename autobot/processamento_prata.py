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
        # Configurações de conexão R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}))
        
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

    def _limpar_texto_standard(self, coluna):
        """Aplica Uppercase e remove acentos de forma performática no Polars."""
        return (
            pl.col(coluna)
            .cast(pl.String)
            .str.to_uppercase()
            .str.replace_all(r"[ÁÀÂÃ]", "A")
            .str.replace_all(r"[ÉÈÊ]", "E")
            .str.replace_all(r"[ÍÌÎ]", "I")
            .str.replace_all(r"[ÓÒÔÕ]", "O")
            .str.replace_all(r"[ÚÙÛ]", "U")
            .str.replace_all(r"[Ç]", "C")
            .str.strip_chars()
            .fill_null("INDEFINIDO")
        )

    def _inicializar_dependencias(self):
        """Carrega a malha geográfica e já normaliza as colunas de cruzamento."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            # NORMALIZAÇÃO DA MALHA (O lado fixo do Join)
            self.df_malha_lazy = (
                pl.read_parquet(io.BytesIO(resp['Body'].read()))
                .lazy()
                .with_columns([
                    self._limpar_texto_standard("NM_MUN").alias("NM_MUN"),
                    self._limpar_texto_standard("NM_BAIRRO").alias("NM_BAIRRO"),
                    self._limpar_texto_standard("LOGRADOURO").alias("LOGRADOURO_GRID")
                ])
            )
            logger.info("PRATA: Malha geográfica normalizada e pronta para cruzamento.")
        except Exception as e:
            logger.error(f"PRATA: Erro ao carregar malha: {e}")
            self.df_malha_lazy = None

    def processar_ano_com_delta(self, ano, estado, force=False):
        path_trusted = f"{self.base_path}/bronze/trusted/ssp_trusted_{ano}.parquet"
        path_prata = f"{self.base_path}/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            if not force and estado.get(str(ano)) == meta['ContentLength']: return None

            resp = self.s3.get_object(Bucket=self.bucket, Key=path_trusted)
            lf = pl.read_parquet(io.BytesIO(resp['Body'].read())).lazy()

            # --- 1. MAPEAMENTO DE SINÔNIMOS (Schema Drift) ---
            cols_atuais = lf.collect_schema().names()
            mapeamento = {
                "CIDADE": "MUNICIPIO", "NOME_MUNICIPIO": "MUNICIPIO",
                "HORA_OCORRENCIA_BO": "HORA",
                "DESC_PERIODO": "PERIODO_TEXTO", "DESCR_PERIODO": "PERIODO_TEXTO",
                "DESCR_TIPOLOCAL": "TIPO_LOCAL", "DESCR_SUBTIPOLOCAL": "TIPO_LOCAL",
                "DESCR_CONDUTA": "CONDUTA"
            }
            rename_dict = {old: new for old, new in mapeamento.items() if old in cols_atuais}
            if rename_dict: lf = lf.rename(rename_dict)

            total_entrada = lf.select(pl.len()).collect().item()

            # --- 2. NORMALIZAÇÃO DO B.O. (O lado variável do Join) ---
            # Aplicamos a mesma régua de limpeza da Malha
            colunas_para_limpar = ["MUNICIPIO", "BAIRRO", "LOGRADOURO", "RUBRICA", "CONDUTA", "TIPO_LOCAL", "PERIODO_TEXTO"]
            lf = lf.with_columns([self._limpar_texto_standard(c) for c in colunas_para_limpar if c in lf.collect_schema().names()])

            # --- 3. CRUZAMENTO E CURA ---
            # O Join agora acontece entre colunas que foram limpas da mesma forma
            lf_enriquecido = lf.join(self.df_malha_lazy, on="H3_INDEX", how="left")
            
            lf_enriquecido = lf_enriquecido.with_columns([
                pl.coalesce([pl.col("NM_MUN"), pl.col("MUNICIPIO")]).alias("NM_MUN_FINAL"),
                pl.coalesce([pl.col("NM_BAIRRO"), pl.col("BAIRRO")]).alias("NM_BAIRRO_FINAL")
            ])

            # HIGIENE: Excluir se não tiver localização mínima
            lf_enriquecido = lf_enriquecido.filter(
                (pl.col("H3_INDEX").is_not_null()) & 
                (pl.col("NM_MUN_FINAL") != "INDEFINIDO") & 
                (pl.col("NM_BAIRRO_FINAL") != "INDEFINIDO")
            )

            # --- 4. LÓGICA DE NEGÓCIO (HORAS E PERÍODOS) ---
            lf_enriquecido = lf_enriquecido.with_columns([
                pl.col("HORA").str.split(":").list.first().cast(pl.Int32, strict=False).alias("HORA_INT"),
                pl.when(pl.col("CONDUTA").str.contains("TRANSEUNTE|PEDESTRE")).then(pl.lit("PEDESTRE"))
                  .when(pl.col("RUBRICA").str.contains("VEICULO|AUTO|MOTO")).then(pl.lit("MOTORISTA"))
                  .otherwise(pl.lit("PEDESTRE")).alias("PERFIL_ALVO")
            ]).with_columns([
                pl.when((pl.col("HORA_INT") > 0) & (pl.col("HORA_INT") < 6)).then(pl.lit("MADRUGADA"))
                  .when((pl.col("HORA_INT") >= 6) & (pl.col("HORA_INT") < 12)).then(pl.lit("MANHA"))
                  .when((pl.col("HORA_INT") >= 12) & (pl.col("HORA_INT") < 18)).then(pl.lit("TARDE"))
                  .when((pl.col("HORA_INT") >= 18) & (pl.col("HORA_INT") <= 23)).then(pl.lit("NOITE"))
                  .when(pl.col("PERIODO_TEXTO").str.contains("MADRUGADA")).then(pl.lit("MADRUGADA"))
                  .when(pl.col("PERIODO_TEXTO").str.contains("MANHA")).then(pl.lit("MANHA"))
                  .when(pl.col("PERIODO_TEXTO").str.contains("TARDE")).then(pl.lit("TARDE"))
                  .when(pl.col("PERIODO_TEXTO").str.contains("NOITE")).then(pl.lit("NOITE"))
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

            df_final = lf_final.collect(engine="streaming")
            
            buffer = io.BytesIO()
            df_final.write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            estado[str(ano)] = meta['ContentLength']
            return {"linhas_in": total_entrada, "linhas_out": df_final.height}

        except Exception as e:
            logger.error(f"PRATA: Erro {ano}: {e}")
            return None

    def executar_todos_os_anos(self, force=False):
        stats = {"linhas_in": 0, "linhas_out": 0}
        estado = self._carregar_tracker()
        for ano in range(2022, datetime.now().year + 1):
            res = self.processar_ano_com_delta(ano, estado, force)
            if res:
                stats["linhas_in"] += res["linhas_in"]
                stats["linhas_out"] += res["linhas_out"]
                self._salvar_tracker(estado)
        return stats

    def _carregar_tracker(self):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.tracker_path)
            return json.loads(resp['Body'].read())
        except: return {}

    def _salvar_tracker(self, estado):
        self.s3.put_object(Bucket=self.bucket, Key=self.tracker_path, Body=json.dumps(estado))
