import polars as pl
import pandas as pd
import boto3
from botocore.config import Config
import joblib
import io
import os
import json
import logging
import numpy as np
from datetime import datetime
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
from google.oauth2 import service_account
from autobot.calendario_estrategico import CalendarioEstrategico

# Otimizacao: carrega a explicabilidade pesada apenas sob demanda
import shap

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class CamadaOuroSafeDriver:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # Apontamento direto para o projeto de faturamento
        self.project_id = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9").strip()
        self.dataset_id = os.getenv("BQ_DATASET_ID", "safedriver_gold").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        
        self.cal = CalendarioEstrategico()
        self.pesos = {"catboost": 0.85, "lightgbm": 0.15}
        self._conectar_bigquery()

    def _conectar_bigquery(self):
        gcp_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
        try:
            cred_info = json.loads(gcp_json)
            credentials = service_account.Credentials.from_service_account_info(cred_info)
            self.bq_client = bigquery.Client(credentials=credentials, project=self.project_id)
        except Exception as e:
            logger.error(f"OURO: Erro na autenticacao GCP: {e}")

    def executar_predicao_atual(self):
        logger.info("OURO: Iniciando materializacao do Data Warehouse.")
        try:
            modelos = self._carregar_modelos_producao()
            df_input = self._obter_contexto_datalake()
            
            if df_input is None: 
                logger.error("OURO: Falha ao carregar dados de contexto do Data Lake.")
                return False

            # Gera predições e explicabilidade SHAP
            df_final = self._gerar_scores_ponderados(df_input, modelos)
            
            # Sincroniza DW
            self._sincronizar_infraestrutura_bq(df_final)
            
            return {
                "embeds": [{
                    "title": "SafeDriver - Sincronizacao Ouro Concluida",
                    "color": 3066993, 
                    "fields": [
                        {"name": "Registros Processados", "value": f"`{len(df_final):,}`", "inline": True},
                        {"name": "Status Camada Ouro", "value": "Esquema Estrela e Prova Real Atualizados", "inline": True},
                        {"name": "Explicabilidade", "value": "Matriz SHAP Nativa Integrada", "inline": False}
                    ]
                }]
            }
        except Exception as e:
            logger.error(f"OURO: Erro critico na Camada Ouro: {e}")
            return False

    def _carregar_modelos_producao(self):
        modelos = {"geral": {}}
        for alg in ["cat", "lgb"]:
            path = f"datalake/modelos_ml/latest_{alg}_geral.pkl"
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                modelos["geral"][alg] = joblib.load(io.BytesIO(obj['Body'].read()))
            except Exception as e:
                logger.warning(f"OURO: Modelo {alg} nao encontrado no R2: {e}")
                modelos["geral"][alg] = None
        return modelos

    def _obter_contexto_datalake(self):
        lista_dfs = []
        for ano in range(2022, datetime.now().year + 1):
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=f"datalake/prata/ssp_consolidada_{ano}.parquet")
                df = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                lista_dfs.append(df)
            except Exception: 
                continue
        
        if not lista_dfs: 
            return None

        # Processamento pesado em memoria via Polars
        df_full = pl.concat(lista_dfs, how="diagonal")
        
        # Agregacao historica para validacao de acuracia (Prova Real)
        df_agg = df_full.group_by("H3_INDEX").agg([
            pl.col("TOTAL_CRIMES").sum().alias("TOTAL_CRIMES_HISTORICO"),
            pl.col("NM_BAIRRO").first(),
            pl.col("NM_MUN").first(),
            pl.col("DENSIDADE").first().fill_null(0.0),
            pl.col("TAXA_VACANCIA").first().fill_null(0.0) if "TAXA_VACANCIA" in df_full.columns else pl.lit(0.0).alias("TAXA_VACANCIA")
        ]).to_pandas()

        df_agg['PERFIL_AREA'] = np.where(df_agg['DENSIDADE'] > 5000, "RESIDENCIAL", 
                                np.where(df_agg['DENSIDADE'] == 0, "COMERCIAL_INDUSTRIAL", "MISTO"))
        
        # Adequacao de tipos para o modelo de IA
        for col in ['NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']:
            df_agg[col] = df_agg[col].astype(str).fillna("DESCONHECIDO").astype('category')
        
        return df_agg

    def _gerar_scores_ponderados(self, df, modelos):
        logger.info("OURO: Computando predicacao espacial e decomposicao SHAP...")
        features = ['DENSIDADE', 'TAXA_VACANCIA', 'NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']
        mults = self.cal.obter_multiplicadores()
        mods = modelos["geral"]
        
        # Ensemble Scoring
        p_cat = mods["cat"].predict(df[features]) if mods["cat"] else 0
        p_lgb = mods["lgb"].predict(df[features]) if mods["lgb"] else 0
        
        score_base = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
        df['SCORE_RISCO_GERAL'] = score_base * mults.get("geral", 1.0)
        
        # Ajuste de Sazonalidade
        mask_comercial = df['PERFIL_AREA'] == "COMERCIAL_INDUSTRIAL"
        df.loc[mask_comercial, 'SCORE_RISCO_GERAL'] *= mults.get("comercial", 1.0)
        df.loc[~mask_comercial, 'SCORE_RISCO_GERAL'] *= mults.get("residencial", 1.0)
        
        # Normalizacao 0-100
        max_val = df['SCORE_RISCO_GERAL'].max() if df['SCORE_RISCO_GERAL'].max() > 0 else 1
        df['SCORE_RISCO_GERAL'] = (df['SCORE_RISCO_GERAL'] / max_val * 100).clip(0, 100).round(2)
        df['DT_ULTIMA_SINCRONIZACAO'] = datetime.now()

        # --- Extracao Analitica (SHAP) para Carga Direta no DW ---
        if mods["cat"]:
            explainer = shap.TreeExplainer(mods["cat"])
            shap_values = explainer.shap_values(df[features])
            
            df['SHAP_BASE'] = float(explainer.expected_value)
            # A ordem do shap_values bate com a ordem do array `features`
            df['SHAP_DENSIDADE'] = shap_values[:, 0].astype(float)
            df['SHAP_VACANCIA'] = shap_values[:, 1].astype(float)
            df['SHAP_BAIRRO'] = shap_values[:, 2].astype(float)
            df['SHAP_MUN'] = shap_values[:, 3].astype(float)
            df['SHAP_PERFIL'] = shap_values[:, 4].astype(float)
        else:
            for c in ['SHAP_BASE', 'SHAP_DENSIDADE', 'SHAP_VACANCIA', 'SHAP_BAIRRO', 'SHAP_MUN', 'SHAP_PERFIL']:
                df[c] = 0.0
        
        colunas_finais = [
            'H3_INDEX', 'SCORE_RISCO_GERAL', 'TOTAL_CRIMES_HISTORICO', 
            'NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA', 'DENSIDADE', 'TAXA_VACANCIA', 
            'SHAP_BASE', 'SHAP_DENSIDADE', 'SHAP_VACANCIA', 'SHAP_BAIRRO', 'SHAP_MUN', 'SHAP_PERFIL',
            'DT_ULTIMA_SINCRONIZACAO'
        ]
        return df[colunas_finais]

    def _sincronizar_infraestrutura_bq(self, df):
        dataset_path = f"{self.project_id}.{self.dataset_id}"
        tabela_fato = f"{dataset_path}.fato_risco_h3_atual"
        tabela_staging = f"{tabela_fato}_staging"
        
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df, tabela_staging, job_config=job_config).result()

        try:
            self.bq_client.get_table(tabela_fato)
        except NotFound:
            logger.info("OURO: Tabela Fato ausente. Realizando setup inicial de metadados...")
            sql_init = f"""
                CREATE TABLE `{tabela_fato}` AS 
                SELECT H3_INDEX, SCORE_RISCO_GERAL, TOTAL_CRIMES_HISTORICO, 
                       SHAP_BASE, SHAP_DENSIDADE, SHAP_VACANCIA, SHAP_BAIRRO, SHAP_MUN, SHAP_PERFIL, 
                       DT_ULTIMA_SINCRONIZACAO 
                FROM `{tabela_staging}`
            """
            self.bq_client.query(sql_init).result()

        self.bq_client.query(f"""
            MERGE `{tabela_fato}` T USING `{tabela_staging}` S ON T.H3_INDEX = S.H3_INDEX
            WHEN MATCHED THEN 
                UPDATE SET 
                    T.SCORE_RISCO_GERAL = S.SCORE_RISCO_GERAL, 
                    T.TOTAL_CRIMES_HISTORICO = S.TOTAL_CRIMES_HISTORICO, 
                    T.SHAP_BASE = S.SHAP_BASE,
                    T.SHAP_DENSIDADE = S.SHAP_DENSIDADE,
                    T.SHAP_VACANCIA = S.SHAP_VACANCIA,
                    T.SHAP_BAIRRO = S.SHAP_BAIRRO,
                    T.SHAP_MUN = S.SHAP_MUN,
                    T.SHAP_PERFIL = S.SHAP_PERFIL,
                    T.DT_ULTIMA_SINCRONIZACAO = S.DT_ULTIMA_SINCRONIZACAO
            WHEN NOT MATCHED THEN 
                INSERT (H3_INDEX, SCORE_RISCO_GERAL, TOTAL_CRIMES_HISTORICO, 
                        SHAP_BASE, SHAP_DENSIDADE, SHAP_VACANCIA, SHAP_BAIRRO, SHAP_MUN, SHAP_PERFIL, 
                        DT_ULTIMA_SINCRONIZACAO) 
                VALUES(S.H3_INDEX, S.SCORE_RISCO_GERAL, S.TOTAL_CRIMES_HISTORICO, 
                       S.SHAP_BASE, S.SHAP_DENSIDADE, S.SHAP_VACANCIA, S.SHAP_BAIRRO, S.SHAP_MUN, S.SHAP_PERFIL, 
                       S.DT_ULTIMA_SINCRONIZACAO)
        """).result()

        logger.info("OURO: Atualizando Star Schema e Views Semanticas...")
        self._build_star_schema(dataset_path, tabela_staging, tabela_fato)

    def _build_star_schema(self, dataset, staging, fato):
        self.bq_client.query(f"""
            CREATE OR REPLACE TABLE `{dataset}.dim_h3` AS
            SELECT DISTINCT H3_INDEX, NM_BAIRRO, NM_MUN as CIDADE, PERFIL_AREA, DENSIDADE, TAXA_VACANCIA 
            FROM `{staging}`
        """).result()

        self.bq_client.query(f"""
            CREATE OR REPLACE TABLE `{dataset}.dim_tempo` AS
            SELECT d as DATA, EXTRACT(DAYOFWEEK FROM d) as DIA_SEMANA, FORMAT_DATE('%A', d) as NOME_DIA,
            IF(EXTRACT(DAY FROM d) BETWEEN 1 AND 10, TRUE, FALSE) as SEMANA_PAGAMENTO
            FROM UNNEST(GENERATE_DATE_ARRAY('2022-01-01', '2026-12-31')) d
        """).result()

        self.bq_client.query(f"""
            CREATE OR REPLACE VIEW `{dataset}.v_safedriver_analitico` AS
            SELECT 
                f.H3_INDEX, 
                f.SCORE_RISCO_GERAL, 
                f.TOTAL_CRIMES_HISTORICO,
                g.NM_BAIRRO, 
                g.CIDADE, 
                g.PERFIL_AREA, 
                g.DENSIDADE,
                g.TAXA_VACANCIA,
                f.SHAP_BASE, 
                f.SHAP_DENSIDADE, 
                f.SHAP_VACANCIA, 
                f.SHAP_BAIRRO, 
                f.SHAP_MUN, 
                f.SHAP_PERFIL,
                t.NOME_DIA, 
                t.SEMANA_PAGAMENTO, 
                f.DT_ULTIMA_SINCRONIZACAO
            FROM `{fato}` f
            JOIN `{dataset}.dim_h3` g ON f.H3_INDEX = g.H3_INDEX
            LEFT JOIN `{dataset}.dim_tempo` t ON DATE(f.DT_ULTIMA_SINCRONIZACAO) = t.DATA
        """).result()
        
        logger.info("OURO: Infraestrutura DW sincronizada e pronta para consumo analitico.")

if __name__ == "__main__":
    camada_ouro = CamadaOuroSafeDriver()
    camada_ouro.executar_predicao_atual()
