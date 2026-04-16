import polars as pl
import pandas as pd
import h3
import boto3
from botocore.config import Config
import joblib
import io
import os
import json
import logging
import shap
from datetime import datetime
from google.cloud import bigquery
from google.oauth2 import service_account
from autobot.calendario_estrategico import CalendarioEstrategico

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class CamadaOuroSafeDriver:
    def __init__(self, dev_mode=False):
        self.dev_mode = dev_mode
        
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        self.project_id = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9").strip()
        self.dataset_id = os.getenv("BQ_DATASET_ID", "safedriver_gold").strip()

        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}, max_pool_connections=50))
        
        self.base_path = self._localizar_datalake_real()
        self.malha_path = f"{self.base_path}/base_geografica/safedriver_geo_base_sp_h3_9.parquet"
        self.cal = CalendarioEstrategico()
        
        self.pesos = {"catboost": 0.70, "lightgbm": 0.30}
        
        # --- SCHEMA ALINHADO COM A HEAVY SILVER E O TREINADOR ---
        self.features_numericas = [
            'DENSIDADE', 'TAXA_VACANCIA', 'RANKING_RISCO_LOCAL', 'INDICE_EXPOSICAO',
            'CONTAGIO_PONDERADO', 'PRESSAO_RISCO_LOCAL', 
            'MES_OCORRENCIA', 'DIA_SEMANA', 'IS_PAGAMENTO', 'IS_FDS' # <-- Ajuste vital
        ]
        
        self.features_categoricas = ['NM_BAIRRO', 'NM_MUN', 'PERIODO_DIA', 'PERFIL_ALVO', 'TIPO_LOCAL']
        self.features_full = self.features_numericas + self.features_categoricas

        self._conectar_bigquery()

    def _localizar_datalake_real(self):
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, MaxKeys=100):
                for obj in page.get('Contents', []):
                    if "datalake/prata/" in obj['Key']:
                        return obj['Key'].split("datalake/")[0] + "datalake"
            return "datalake"
        except: return "datalake"

    def _get_path(self, camada, filename):
        return f"{self.base_path}/{camada}/{filename}".replace("//", "/")

    def _conectar_bigquery(self):
        gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
        try:
            cred_info = json.loads(gcp_json)
            credentials = service_account.Credentials.from_service_account_info(cred_info)
            self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
        except Exception as e:
            logger.error(f"OURO: Falha na autenticação BigQuery: {e}")

    def executar_predicao_atual(self):
        logger.info(f"OURO: Iniciando Ciclo de Inteligência Preditiva.")
        try:
            modelos = self._carregar_modelos_producao()
            df_input = self._obter_contexto_preditivo_total()
            
            if df_input is None or modelos["cat"] is None: 
                logger.error("OURO: Insumos (modelos ou dados) não encontrados.")
                return False

            df_final = self._gerar_scores_ponderados(df_input, modelos)
            
            self._salvar_parquet_ouro(df_final)
            self._sincronizar_bq(df_final)
            
            return True
        except Exception as e:
            logger.error(f"OURO: Erro crítico no pipeline: {e}")
            return False

    def _carregar_modelos_producao(self):
        modelos = {"cat": None, "lgb": None}
        for alg in ["cat", "lgb"]:
            path = self._get_path("modelos_ml", f"latest_{alg}_geral.pkl")
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                modelos[alg] = joblib.load(io.BytesIO(obj['Body'].read()))
                logger.info(f"OURO: Modelo {alg} carregado da produção.")
            except:
                logger.warning(f"OURO: Modelo {alg} ausente no R2.")
        return modelos

    def _gerar_contagio_na_malha_total(self, df_malha, df_crimes_prata):
        logger.info("OURO: Projetando contágio espacial sobre a malha mestre...")
        
        df_full = df_malha.join(
            df_crimes_prata.select(["H3_INDEX", "TOTAL_CRIMES", "RANKING_PRATA", "EXPOSICAO_PRATA"]), 
            on="H3_INDEX", 
            how="left"
        ).with_columns(pl.col("TOTAL_CRIMES").fill_null(0))

        df_pd = df_full.to_pandas()
        crimes_dit = dict(zip(df_pd['H3_INDEX'], df_pd['TOTAL_CRIMES']))
        usar_grid_disk = hasattr(h3, 'grid_disk')
        
        contagio = []
        for h3_index in df_pd['H3_INDEX']:
            try:
                v1 = set(h3.grid_disk(h3_index, 1) if usar_grid_disk else h3.k_ring(h3_index, 1))
                v1.discard(h3_index)
                c1 = sum(crimes_dit.get(v, 0) for v in v1)
                
                v_total = set(h3.grid_disk(h3_index, 2) if usar_grid_disk else h3.k_ring(h3_index, 2))
                v2 = v_total - v1
                v2.discard(h3_index)
                c2 = sum(crimes_dit.get(v, 0) for v in v2)
                
                contagio.append((c1 * 1.0) + (c2 * 0.5))
            except: contagio.append(0.0)
            
        df_pd['CONTAGIO_PONDERADO'] = contagio
        df_pd['PRESSAO_RISCO_LOCAL'] = df_pd['CONTAGIO_PONDERADO'] / (df_pd['DENSIDADE_AJUSTADA'] + 0.001)
        
        return pl.from_pandas(df_pd)

    def _obter_contexto_preditivo_total(self):
        try:
            resp_m = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            df_malha = pl.read_parquet(io.BytesIO(resp_m['Body'].read()))

            ano_atual = datetime.now().year
            path_prata = self._get_path("prata", f"ssp_consolidada_{ano_atual}.parquet")
            resp_p = self.s3.get_object(Bucket=self.bucket, Key=path_prata)
            df_prata = pl.read_parquet(io.BytesIO(resp_p['Body'].read()))

            # Resumo da Prata: Médias Históricas
            df_prata_agg = df_prata.group_by("H3_INDEX").agg([
                pl.col("TOTAL_CRIMES").sum().alias("TOTAL_CRIMES"),
                pl.col("RANKING_RISCO_LOCAL").mean().alias("RANKING_PRATA"),
                pl.col("INDICE_EXPOSICAO").mean().alias("EXPOSICAO_PRATA")
            ])

            df_enriquecida = self._gerar_contagio_na_malha_total(df_malha, df_prata_agg)

            hoje = datetime.now()
            dia_atual = hoje.day
            
            # --- CONSTRUÇÃO DO CONTEXTO ATUAL PARA O MODELO ---
            df_final = df_enriquecida.with_columns([
                pl.col("NM_MUN").fill_null("SAO PAULO"),
                pl.col("NM_BAIRRO").fill_null("INDEFINIDO"),
                
                pl.lit("TARDE").alias("PERIODO_DIA"), 
                pl.lit("PEDESTRE").alias("PERFIL_ALVO"),
                pl.lit("VIA PUBLICA").alias("TIPO_LOCAL"),
                
                pl.lit(hoje.month).alias("MES_OCORRENCIA"),
                pl.lit(hoje.weekday()).alias("DIA_SEMANA"),
                
                # Inteligência de Negócio: Dia de Pagamento e Fim de Semana (injetados em tempo real)
                pl.lit(1 if 5 <= dia_atual <= 10 else 0).alias("IS_PAGAMENTO"),
                pl.lit(1 if hoje.weekday() >= 6 else 0).alias("IS_FDS"),
                
                pl.col("DENSIDADE_AJUSTADA").alias("DENSIDADE"),
                pl.col("TAXA_VACANCIA").fill_null(0.0),
                
                pl.col("RANKING_PRATA").fill_null(0.5).alias("RANKING_RISCO_LOCAL"),
                pl.col("EXPOSICAO_PRATA").fill_null(0.1).alias("INDICE_EXPOSICAO")
            ])

            df_pd = df_final.to_pandas()
            for col in self.features_categoricas:
                df_pd[col] = df_pd[col].astype(str).astype('category')
            for col in self.features_numericas:
                df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce').fillna(0.0)
            
            return df_pd
        except Exception as e:
            logger.error(f"OURO: Erro ao consolidar contexto: {e}")
            return None

    def _gerar_scores_ponderados(self, df, modelos):
        logger.info("OURO: Gerando scores e SHAP explainability...")
        mults = self.cal.obter_multiplicadores()
        
        p_cat = modelos["cat"].predict(df[self.features_full])
        p_lgb = modelos["lgb"].predict(df[self.features_full]) if modelos["lgb"] else p_cat
        
        score_base = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
        df['SCORE_RISCO_BRUTO'] = score_base * mults.get("geral", 1.0)
        
        max_val = df['SCORE_RISCO_BRUTO'].max() or 1
        df['SCORE_RISCO_FINAL'] = ((df['SCORE_RISCO_BRUTO'] / max_val) * 100).clip(0, 100).round(2)
        
        if 'TOTAL_CRIMES' in df.columns:
            df['FLAG_ALERTA_SILENCIOSO'] = (df['TOTAL_CRIMES'] == 0) & (df['SCORE_RISCO_FINAL'] > 75)
        
        df['DT_PROCESSAMENTO'] = datetime.now()

        try:
            explainer = shap.TreeExplainer(modelos["cat"])
            df_top = df.nlargest(min(len(df), 3000), 'SCORE_RISCO_FINAL')
            shap_values = explainer.shap_values(df_top[self.features_full])
            
            for i, feat in enumerate(['NM_MUN', 'PERIODO_DIA', 'PERFIL_ALVO']):
                col_name = f'SHAP_{feat}'
                df[col_name] = 0.0
                if feat in self.features_full:
                    feat_idx = self.features_full.index(feat)
                    df.loc[df_top.index, col_name] = shap_values[:, feat_idx]
        except Exception as e:
            logger.warning(f"OURO: SHAP ignorado por erro técnico: {e}")
            
        return df

    def _salvar_parquet_ouro(self, df):
        buffer = io.BytesIO()
        df_save = df.copy()
        for col in self.features_categoricas:
            df_save[col] = df_save[col].astype(str)
        pl.from_pandas(df_save).write_parquet(buffer, compression="lz4")
        key = self._get_path("ouro", "fato_risco_consolidada.parquet")
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue())
        logger.info("OURO: Persistido no R2 (Format: Parquet).")

    def _sincronizar_bq(self, df):
        tabela_final = f"{self.project_id}.{self.dataset_id}.fato_risco_h3_vigente"
        tabela_staging = f"{tabela_final}_staging"
        
        df_bq = df.copy()
        for col in self.features_categoricas:
            df_bq[col] = df_bq[col].astype(str)

        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df_bq, tabela_staging, job_config=job_config).result()
        
        sql_merge = f"""
        MERGE `{tabela_final}` T
        USING `{tabela_staging}` S
        ON T.H3_INDEX = S.H3_INDEX AND T.PERIODO_DIA = S.PERIODO_DIA AND T.PERFIL_ALVO = S.PERFIL_ALVO
        WHEN MATCHED THEN
          UPDATE SET 
            T.SCORE_RISCO_FINAL = S.SCORE_RISCO_FINAL,
            T.DT_PROCESSAMENTO = S.DT_PROCESSAMENTO,
            T.CONTAGIO_PONDERADO = S.CONTAGIO_PONDERADO
        WHEN NOT MATCHED THEN
          INSERT ROW
        """
        self.bq_client.query(sql_merge).result()
        logger.info("OURO: Data Warehouse BigQuery sincronizado.")

if __name__ == "__main__":
    ouro = CamadaOuroSafeDriver(dev_mode=False)
    ouro.executar_predicao_atual()
