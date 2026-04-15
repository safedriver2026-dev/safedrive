import polars as pl
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
        # Credenciais Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        # Configuração de Conexão S3
        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}, max_pool_connections=50))
        
        self.base_path = self._localizar_datalake_real()
        self.tracker_path = f"{self.base_path}/prata/tracker_estado_bronze.json"
        self.malha_path = f"{self.base_path}/base_geografica/safedriver_geo_base_sp_h3_9.parquet"
        
        self._inicializar_dependencias()

    def _localizar_datalake_real(self):
        """Busca dinâmica para garantir o prefixo correto no R2."""
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, MaxKeys=100):
                for obj in page.get('Contents', []):
                    if "datalake/bronze/trusted/" in obj['Key']:
                        return obj['Key'].split("datalake/")[0] + "datalake"
            return "datalake"
        except: return "datalake"

    def _limpar_texto_extremo(self, coluna):
        """Normalização total: Uppercase, Sem Acentos e Remoção de Espaços."""
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
        """Carrega e normaliza a malha geográfica para garantir o Join."""
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            self.df_malha_lazy = (
                pl.read_parquet(io.BytesIO(resp['Body'].read()))
                .lazy()
                .with_columns([
                    self._limpar_texto_extremo("NM_MUN").alias("NM_MUN"),
                    self._limpar_texto_extremo("NM_BAIRRO").alias("NM_BAIRRO"),
                    self._limpar_texto_extremo("LOGRADOURO").alias("LOGRADOURO_GRID")
                ])
            )
            logger.info("PRATA: Malha geográfica normalizada carregada.")
        except Exception as e:
            logger.error(f"PRATA: Falha ao carregar malha H3: {e}")
            self.df_malha_lazy = None

    def processar_ano_com_delta(self, ano, estado, force=False):
        path_trusted = f"{self.base_path}/bronze/trusted/ssp_trusted_{ano}.parquet"
        path_prata = f"{self.base_path}/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            # 1. Verificação de existência e tamanho (Auditável)
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            tamanho_atual = meta['ContentLength'] # <--- CORREÇÃO: Definido no escopo correto
            
            if not force and estado.get(str(ano)) == tamanho_atual: 
                return None

            resp = self.s3.get_object(Bucket=self.bucket, Key=path_trusted)
            lf = pl.read_parquet(io.BytesIO(resp['Body'].read())).lazy()

            # --- 2. RESOLUÇÃO DE SCHEMA DRIFT & DUPLICATAS ---
            cols = lf.collect_schema().names()
            
            # Prioridade Local: Subtipo ganha de Tipo
            if "DESCR_SUBTIPOLOCAL" in cols:
                lf = lf.rename({"DESCR_SUBTIPOLOCAL": "TIPO_LOCAL"})
                if "DESCR_TIPOLOCAL" in cols: lf = lf.drop("DESCR_TIPOLOCAL")
            elif "DESCR_TIPOLOCAL" in cols:
                lf = lf.rename({"DESCR_TIPOLOCAL": "TIPO_LOCAL"})

            # Mapeamento Elástico (Atende 2022 a 2026)
            mapeamento = {
                "CIDADE": "NM_MUN_ORIGINAL",
                "NOME_MUNICIPIO": "NM_MUN_ORIGINAL",
                "MUNICIPIO": "NM_MUN_ORIGINAL",
                "BAIRRO": "NM_BAIRRO_ORIGINAL",
                "LOGRADOURO": "LOGRADOURO_ORIGINAL",
                "HORA_OCORRENCIA_BO": "HORA",
                "DESC_PERIODO": "PERIODO_TEXTO",
                "DESCR_PERIODO": "PERIODO_TEXTO",
                "DESCR_CONDUTA": "CONDUTA"
            }
            rename_dict = {old: new for old, new in mapeamento.items() if old in cols}
            if rename_dict: lf = lf.rename(rename_dict)

            # --- 3. NORMALIZAÇÃO TOTAL (Sem Acentos / Upper) ---
            total_in = lf.select(pl.len()).collect().item()
            campos_texto = ["NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL", "LOGRADOURO_ORIGINAL", 
                           "RUBRICA", "CONDUTA", "TIPO_LOCAL", "PERIODO_TEXTO"]
            
            lf = lf.with_columns([
                self._limpar_texto_extremo(c) for c in campos_texto if c in lf.collect_schema().names()
            ])

            # --- 4. CRUZAMENTO E CURA ---
            lf_enriquecido = lf.join(self.df_malha_lazy, on="H3_INDEX", how="left")
            
            lf_enriquecido = lf_enriquecido.with_columns([
                pl.coalesce([pl.col("NM_MUN"), pl.col("NM_MUN_ORIGINAL")]).alias("NM_MUN_FINAL"),
                pl.coalesce([pl.col("NM_BAIRRO"), pl.col("NM_BAIRRO_ORIGINAL")]).alias("NM_BAIRRO_FINAL")
            ])

            # Higiene: Remove registros onde não foi possível identificar o local
            lf_enriquecido = lf_enriquecido.filter(
                (pl.col("H3_INDEX").is_not_null()) & 
                (pl.col("NM_MUN_FINAL") != "INDEFINIDO") & 
                (pl.col("NM_BAIRRO_FINAL") != "INDEFINIDO")
            )

            # --- 5. LÓGICA DE NEGÓCIO ---
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

            # --- 6. AGREGAÇÃO E MÉTRICAS ---
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

            # Processamento em Streaming para economizar RAM
            df_final = lf_final.collect(engine="streaming")
            
            # Persistência no R2
            buffer = io.BytesIO()
            df_final.write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            # Atualiza o estado para o Tracker
            estado[str(ano)] = tamanho_atual 
            return {"linhas_in": total_in, "linhas_out": df_final.height}

        except Exception as e:
            logger.error(f"PRATA: Erro crítico no ano {ano}: {e}")
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

if __name__ == "__main__":
    prata = ProcessamentoPrata()
    prata.executar_todos_os_anos(force=True)
