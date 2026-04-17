import polars as pl
import pandas as pd
import h3, boto3, io, os, json, logging, gc
from botocore.config import Config
from datetime import datetime

# Auditoria de Processamento de Alta Fidelidade
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
        # Conectividade Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}, max_pool_connections=50))
        
        self.base_path = self._localizar_datalake_real()
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
            self.df_malha = pl.read_parquet(io.BytesIO(resp['Body'].read())).unique(subset=["H3_INDEX"])
            logger.info("PRATA: Malha geográfica H3-9 carregada.")
        except Exception as e:
            logger.error(f"PRATA: Erro na malha base: {e}")
            self.df_malha = None

    def processar_ano_com_delta(self, ano, estado, force=False):
        path_trusted = f"{self.base_path}/bronze/trusted/ssp_trusted_{ano}.parquet"
        path_prata = f"{self.base_path}/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            if not force and estado.get(str(ano)) == meta['ContentLength']: 
                return False

            logger.info(f"PRATA: [{ano}] Iniciando processamento de alta resolução temporal...")
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_trusted)
            lf = pl.read_parquet(io.BytesIO(resp['Body'].read())).lazy()

            # 1. MOTOR DE COMPATIBILIDADE (2022-2026)
            cols = lf.collect_schema().names()
            rename_map = {}
            mapa_fontes = {
                "NM_MUN_ORIGINAL": ["NOME_MUNICIPIO", "CIDADE"],
                "DATA_BRUTA": ["DATA_OCORRENCIA_BO", "DATA_OCORRENCIA"],
                "HORA": ["HORA_OCORRENCIA_BO", "HORA"],
                "PERIODO_SSP": ["DESC_PERIODO", "DESCR_PERIODO"],
                "TIPO_LOCAL": ["DESCR_SUBTIPOLOCAL", "DESCR_TIPOLOCAL"],
                "NM_BAIRRO_ORIGINAL": ["BAIRRO"]
            }

            for target, sources in mapa_fontes.items():
                match = next((s for s in sources if s in cols), None)
                if match: rename_map[match] = target

            lf = lf.rename(rename_map).filter(pl.col("H3_INDEX").is_not_null())
            cols_atuais = lf.collect_schema().names()
            
            # 2. ENGENHARIA DE CONTEXTO HÍBRIDA
            lf = lf.with_columns([
                pl.col("DATA_BRUTA").str.to_date(format="%d/%m/%Y", strict=False).alias("DATA"),
                pl.col("HORA").str.split(":").list.first().cast(pl.Int8, strict=False).alias("HORA_INT"),
                (pl.col("PERIODO_SSP").str.to_uppercase() if "PERIODO_SSP" in cols_atuais else pl.lit("INDEFINIDO")).alias("PERIODO_RAW")
            ]).with_columns([
                pl.when(pl.col("HORA_INT").is_not_null())
                  .then(
                      pl.when(pl.col("HORA_INT") < 6).then(pl.lit("MADRUGADA"))
                        .when(pl.col("HORA_INT") < 12).then(pl.lit("MANHA"))
                        .when(pl.col("HORA_INT") < 18).then(pl.lit("TARDE"))
                        .otherwise(pl.lit("NOITE"))
                  )
                  .otherwise(
                      pl.when(pl.col("PERIODO_RAW").str.contains("MADRUGADA")).then(pl.lit("MADRUGADA"))
                        .when(pl.col("PERIODO_RAW").str.contains("MANHA")).then(pl.lit("MANHA"))
                        .when(pl.col("PERIODO_RAW").str.contains("TARDE")).then(pl.lit("TARDE"))
                        .otherwise(pl.lit("NOITE"))
                  ).alias("PERIODO_DIA"),

                pl.when(pl.col("RUBRICA").str.contains("(?i)HOMICIDIO|LATROCINIO|ESTUPRO|SEQUESTRO|MORTE")).then(pl.lit(10.0))
                  .when(pl.col("RUBRICA").str.contains("(?i)ROUBO")).then(pl.lit(5.0))
                  .when(pl.col("RUBRICA").str.contains("(?i)VEICULO|AUTO|MOTO")).then(pl.lit(3.0))
                  .otherwise(pl.lit(1.0)).alias("PESO_CRIME"),
                
                pl.when(pl.col("RUBRICA").str.contains("(?i)VEICULO|AUTO|MOTO")).then(pl.lit("MOTORISTA")).otherwise(pl.lit("PEDESTRE")).alias("PERFIL_ALVO")
            ])

            # 3. CONSOLIDAÇÃO DIMENSIONAL (SÉRIE TEMPORAL REAL)
            # 🚨 Correção: O Mês agora compõe a chave de agrupamento para podermos calcular o Lag.
            lf_agg = lf.group_by([
                "H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO", "TIPO_LOCAL", 
                "NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL",
                pl.col("DATA").dt.month().alias("MES_OCORRENCIA")
            ]).agg([
                pl.len().alias("TOTAL_CRIMES"),
                pl.col("PESO_CRIME").sum().alias("INDICE_GRAVIDADE"),
                pl.col("DATA").dt.weekday().first().alias("DIA_SEMANA")
            ])

            # 4. A CURA DO VAZAMENTO: DEFASAGEM TEMPORAL (LAG D-1)
            # Ordenamos cronologicamente e deslocamos (shift) os indicadores de risco
            # Assim, a linha de Junho terá os valores de Maio na coluna "HISTORICA".
            df_agg = lf_agg.sort(["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO", "MES_OCORRENCIA"]).with_columns([
                pl.col("INDICE_GRAVIDADE").shift(1).over(["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO"]).fill_null(0.0).alias("GRAVIDADE_HISTORICA"),
                pl.col("TOTAL_CRIMES").shift(1).over(["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO"]).fill_null(0.0).alias("VOLUME_HISTORICO")
            ]).collect()

            df_pd = df_agg.to_pandas()
            
            # 5. CONTAGIO ESPACIAL (Baseado no passado, não no presente)
            # Agregamos a gravidade histórica média do H3 para evitar vazar dados espaciais do momento atual
            gravidade_hist_por_h3 = df_pd.groupby("H3_INDEX")["GRAVIDADE_HISTORICA"].mean().to_dict()

            contagio_map = {}
            for hx in df_pd['H3_INDEX'].unique():
                try:
                    v1 = h3.k_ring(hx, 1)
                    contagio_map[hx] = sum(gravidade_hist_por_h3.get(v, 0.0) for v in v1 if v != hx)
                except: contagio_map[hx] = 0.0

            # --- INJEÇÃO DE FEATURES DISCRIMINANTES ---
            df_pd['CONTAGIO_PONDERADO'] = df_pd['H3_INDEX'].map(contagio_map).fillna(0.0)
            df_pd['PESO_TOTAL_H3'] = df_pd['H3_INDEX'].map(gravidade_hist_por_h3).fillna(0.0) 
            df_pd['PRESSAO_RISCO_LOCAL'] = df_pd['CONTAGIO_PONDERADO']
            
            df_pd['ANO_REF'] = ano
            df_pd['IS_FDS'] = (df_pd['DIA_SEMANA'] >= 5).astype(int)
            df_pd['IS_PAGAMENTO'] = 0 # Placeholder para o Treinador/Ouro

            # 6. JOIN GEOGRÁFICO FINAL
            df_final = pl.from_pandas(df_pd).join(
                self.df_malha.select(["H3_INDEX", "DENSIDADE_AJUSTADA", "TAXA_VACANCIA", "NM_MUN", "NM_BAIRRO"]), 
                on="H3_INDEX", how="left"
            ).with_columns([
                pl.col("DENSIDADE_AJUSTADA").alias("DENSIDADE"),
                pl.coalesce(["NM_MUN", "NM_MUN_ORIGINAL"]).alias("NM_MUN"),
                pl.coalesce(["NM_BAIRRO", "NM_BAIRRO_ORIGINAL"]).alias("NM_BAIRRO"),
                pl.col("TIPO_LOCAL").fill_null("VIA PUBLICA")
            ]).drop(["NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL", "DENSIDADE_AJUSTADA"])

            # 7. PERSISTÊNCIA LZ4
            buffer = io.BytesIO()
            df_final.write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            estado[str(ano)] = meta['ContentLength']
            return True
            
        except Exception as e:
            logger.error(f"PRATA: Erro crítico em {ano}: {e}")
            return False

    def executar_todos_os_anos(self, force=False):
        logger.info("PRATA: Iniciando consolidação com Memória Temporal (Lags de Segurança Ativados).")
        estado = self._carregar_tracker()
        for ano in range(2022, datetime.now().year + 1):
            if self.processar_ano_com_delta(ano, estado, force):
                self._salvar_tracker(estado)
        return True

    def _carregar_tracker(self):
        try:
            return json.loads(self.s3.get_object(Bucket=self.bucket, Key=f"{self.base_path}/prata/tracker.json")['Body'].read())
        except: return {}

    def _salvar_tracker(self, estado):
        self.s3.put_object(Bucket=self.bucket, Key=f"{self.base_path}/prata/tracker.json", Body=json.dumps(estado))

if __name__ == "__main__":
    prata = ProcessamentoPrata()
    prata.executar_todos_os_anos(force=True)
