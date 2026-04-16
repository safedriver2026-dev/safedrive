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
        
        # Variáveis de Auditoria de Cura
        self.campos_recuperados_grade = 0
        
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
            self.df_malha = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            
            # Normalização inicial
            self.df_malha = self.df_malha.with_columns([
                self._limpar_texto_extremo("NM_MUN").alias("NM_MUN"),
                self._limpar_texto_extremo("NM_BAIRRO").alias("NM_BAIRRO"),
                self._limpar_texto_extremo("LOGRADOURO").alias("LOGRADOURO")
            ])
            logger.info("PRATA: Malha geográfica carregada.")
        except Exception as e:
            logger.error(f"PRATA: Falha ao carregar malha H3: {e}")
            self.df_malha = None

    def _curar_malha_referencia(self, df_bo_limpo):
        """
        UPGRADE: O B.O. corrige a Malha. 
        Se o B.O. tem Município/Bairro e a Malha não tem para aquele H3, a Malha é atualizada.
        """
        logger.info("🛠️ Iniciando processo de cura da malha de referência...")
        
        # 1. Identifica o "Melhor Nome" para cada H3 baseado nos B.Os atuais
        df_conhecimento_novo = (
            df_bo_limpo.filter(pl.col("NM_MUN_ORIGINAL") != "INDEFINIDO")
            .group_by("H3_INDEX")
            .agg([
                pl.col("NM_MUN_ORIGINAL").first().alias("MUN_NOVO"),
                pl.col("NM_BAIRRO_ORIGINAL").first().alias("BAIRRO_NOVO")
            ])
        )

        # 2. Join com a malha atual
        malha_antes = self.df_malha.clone()
        
        self.df_malha = self.df_malha.join(df_conhecimento_novo, on="H3_INDEX", how="left")
        
        # 3. Preenche as lacunas da malha (Cura)
        self.df_malha = self.df_malha.with_columns([
            pl.when(pl.col("NM_MUN") == "INDEFINIDO").then(pl.col("MUN_NOVO")).otherwise(pl.col("NM_MUN")).alias("NM_MUN"),
            pl.when(pl.col("NM_BAIRRO") == "INDEFINIDO").then(pl.col("BAIRRO_NOVO")).otherwise(pl.col("NM_BAIRRO")).alias("NM_BAIRRO")
        ]).drop(["MUN_NOVO", "BAIRRO_NOVO"])

        # 4. Auditoria de quantos hexágonos foram "curados"
        recuperados = self.df_malha.filter(
            (malha_antes.filter(pl.col("NM_MUN") == "INDEFINIDO").height > 0)
        ).height
        
        if recuperados > 0:
            self.campos_recuperados_grade += recuperados
            logger.info(f"✨ Malha Corrigida: {recuperados} hexágonos enriquecidos via B.O.")
            self._persistir_malha_corrigida()

    def _persistir_malha_corrigida(self):
        """Salva a malha atualizada no R2 para que o próximo ciclo já a use corrigida."""
        try:
            buffer = io.BytesIO()
            self.df_malha.write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=self.malha_path, Body=buffer.getvalue())
            logger.info("💾 Malha de referência atualizada persistida no R2.")
        except Exception as e:
            logger.error(f"Erro ao salvar malha curada: {e}")

    def processar_ano_com_delta(self, ano, estado, force=False):
        path_trusted = f"{self.base_path}/bronze/trusted/ssp_trusted_{ano}.parquet"
        path_prata = f"{self.base_path}/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            tamanho_atual = meta['ContentLength']
            
            if not force and estado.get(str(ano)) == tamanho_atual: 
                return None

            resp = self.s3.get_object(Bucket=self.bucket, Key=path_trusted)
            lf = pl.read_parquet(io.BytesIO(resp['Body'].read())).lazy()

            # --- NORMALIZAÇÃO E SCHEMA ---
            cols = lf.collect_schema().names()
            mapeamento = {
                "MUNICIPIO": "NM_MUN_ORIGINAL", "BAIRRO": "NM_BAIRRO_ORIGINAL",
                "LOGRADOURO": "LOGRADOURO_ORIGINAL", "HORA_OCORRENCIA_BO": "HORA",
                "DESCR_PERIODO": "PERIODO_TEXTO"
            }
            rename_dict = {old: new for old, new in mapeamento.items() if old in cols}
            if rename_dict: lf = lf.rename(rename_dict)

            campos_texto = ["NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL", "RUBRICA", "PERIODO_TEXTO"]
            lf = lf.with_columns([self._limpar_texto_extremo(c) for c in campos_texto if c in lf.collect_schema().names()])

            # --- AÇÃO DE CURA (VIA DE VOLTA) ---
            df_atual_limpo = lf.select(["H3_INDEX", "NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL"]).collect()
            self._curar_malha_referencia(df_atual_limpo)

            # --- CRUZAMENTO (VIA DE IDA) ---
            # Agora usa a malha (potencialmente já curada acima)
            lf_enriquecido = lf.join(self.df_malha.lazy(), on="H3_INDEX", how="left")
            
            lf_enriquecido = lf_enriquecido.with_columns([
                pl.coalesce([pl.col("NM_MUN"), pl.col("NM_MUN_ORIGINAL")]).alias("NM_MUN_FINAL"),
                pl.coalesce([pl.col("NM_BAIRRO"), pl.col("NM_BAIRRO_ORIGINAL")]).alias("NM_BAIRRO_FINAL")
            ])

            # --- AGREGAÇÃO FINAL ---
            lf_agg = lf_enriquecido.group_by(["H3_INDEX", "NM_MUN_FINAL", "NM_BAIRRO_FINAL"]).agg([
                pl.len().alias("TOTAL_CRIMES"),
                pl.col("DENSIDADE_AJUSTADA").first().alias("DENSIDADE")
            ])

            df_final = lf_agg.collect(engine="streaming")
            
            buffer = io.BytesIO()
            df_final.write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            estado[str(ano)] = tamanho_atual 
            return {"linhas_in": df_atual_limpo.height, "linhas_out": df_final.height}

        except Exception as e:
            logger.error(f"PRATA: Erro crítico no ano {ano}: {e}")
            return None

    def executar_todos_os_anos(self, force=False):
        stats = {"linhas_in": 0, "linhas_out": 0, "recuperado_grade": 0}
        estado = self._carregar_tracker()
        
        for ano in range(2022, datetime.now().year + 1):
            res = self.processar_ano_com_delta(ano, estado, force)
            if res:
                stats["linhas_in"] += res["linhas_in"]
                stats["linhas_out"] += res["linhas_out"]
                self._salvar_tracker(estado)
        
        # Alimenta estatísticas de cura para o Comunicador
        stats["recuperado_grade"] = self.campos_recuperados_grade
        stats["taxa_recuperacao"] = round((stats["linhas_out"] / stats["linhas_in"] * 100), 2) if stats["linhas_in"] > 0 else 0
        
        return stats

    def _carregar_tracker(self):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.tracker_path)
            return json.loads(resp['Body'].read())
        except: return {}

    def _salvar_tracker(self, estado):
        self.s3.put_object(Bucket=self.bucket, Key=self.tracker_path, Body=json.dumps(estado))
