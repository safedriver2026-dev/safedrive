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
        
        # Conectividade Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        # Conectividade Google BigQuery
        self.project_id = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9").strip()
        self.dataset_id = os.getenv("BQ_DATASET_ID", "safedriver_gold").strip()

        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}, max_pool_connections=50))
        
        self.base_path = self._localizar_datalake_real()
        self.malha_path = f"{self.base_path}/base_geografica/safedriver_geo_base_sp_h3_9.parquet"
        self.cal = CalendarioEstrategico()
        
        # Pesos do Ensemble (CatBoost domina pela estabilidade Tweedie)
        self.pesos = {"catboost": 0.70, "lightgbm": 0.30}
        
        # Features alinhadas 100% com o Treinador IA
        self.features_numericas = [
            'DENSIDADE', 'TAXA_VACANCIA', 'RANKING_RISCO_LOCAL', 'INDICE_EXPOSICAO',
            'CONTAGIO_PONDERADO', 'PRESSAO_RISCO_LOCAL', 
            'MES_OCORRENCIA', 'DIA_SEMANA_OCORRENCIA'
        ]
        
        # PERFIL_AREA removido para evitar erro de index no CatBoost
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
        """Fluxo principal de inteligência preditiva."""
        logger.info(f"OURO: Iniciando Materializacao Preditiva.")
        try:
            modelos = self._carregar_modelos_producao()
            df_input = self._obter_contexto_preditivo_total()
            
            if df_input is None or modelos["cat"] is None: 
                logger.error("OURO: Falha crítica no carregamento de insumos.")
                return False

            # Geração de Inteligência e Scores
            df_final = self._gerar_scores_ponderados(df_input, modelos)
            
            # Persistência no R2 e BigQuery
            self._salvar_parquet_ouro(df_final)
            self._sincronizar_bq(df_final)
            
            return True
        except Exception as e:
            logger.error(f"OURO: Erro no pipeline Ouro: {e}")
            return False

    def _carregar_modelos_producao(self):
        modelos = {"cat": None, "lgb": None}
        for alg in ["cat", "lgb"]:
            path = self._get_path("modelos_ml", f"latest_{alg}_geral.pkl")
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path)
                modelos[alg] = joblib.load(io.BytesIO(obj['Body'].read()))
                logger.info(f"OURO: Modelo {alg} carregado com sucesso.")
            except:
                logger.warning(f"OURO: Modelo {alg} não encontrado no R2.")
        return modelos

    def _gerar_contagio_na_malha_total(self, df_malha, df_crimes_prata):
        """Popula a malha com crimes recentes e calcula vizinhança."""
        logger.info("OURO: Calculando Contágio Espacial na Malha Total...")
        df_full = df_malha.join(
            df_crimes_prata.select(["H3_INDEX", "TOTAL_CRIMES"]), 
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
        """Prepara o DataFrame de entrada para a IA."""
        try:
            resp_m = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            df_malha = pl.read_parquet(io.BytesIO(resp_m['Body'].read()))

            ano_atual = datetime.now().year
            path_prata = self._get_path("prata", f"ssp_consolidada_{ano_atual}.parquet")
            resp_p = self.s3.get_object(Bucket=self.bucket, Key=path_prata)
            df_prata = pl.read_parquet(io.BytesIO(resp_p['Body'].read()))

            df_malha_enriquecida = self._gerar_contagio_na_malha_total(df_malha, df_prata)

            # Preenchimento de colunas padrão para predição (Cenário Base)
            df_final = df_malha_enriquecida.with_columns([
                pl.col("NM_MUN").fill_null("SAO PAULO"),
                pl.col("NM_BAIRRO").fill_null("INDEFINIDO"),
                pl.lit("TARDE").alias("PERIODO_DIA"), 
                pl.lit("PEDESTRE").alias("PERFIL_ALVO"),
                pl.lit("VIA PUBLICA").alias("TIPO_LOCAL"),
                pl.lit(datetime.now().month).alias("MES_OCORRENCIA"),
                pl.lit(datetime.now().weekday()).alias("DIA_SEMANA_OCORRENCIA"),
                pl.col("DENSIDADE_AJUSTADA").alias("DENSIDADE"),
                pl.lit(0.5).alias("RANKING_RISCO_LOCAL"),
                pl.lit(0.1).alias("INDICE_EXPOSICAO")
            ])

            df_pd = df_final.to_pandas()
            for col in self.features_categoricas:
                df_pd[col] = df_pd[col].astype(str).astype('category')
            for col in self.features_numericas:
                df_pd[col] = pd.to_numeric(df_pd[col], errors='coerce').fillna(0.0)
            
            return df_pd
        except Exception as e:
            logger.error(f"OURO: Erro no contexto preditivo: {e}")
            return None

    def _gerar_scores_ponderados(self, df, modelos):
        """Calcula risco e explicabilidade."""
        logger.info("OURO: Executando Ensemble e SHAP...")
        mults = self.cal.obter_multiplicadores()
        
        # Predição Ensemble
        p_cat = modelos["cat"].predict(df[self.features_full])
        p_lgb = modelos["lgb"].predict(df[self.features_full]) if modelos["lgb"] else p_cat
        
        score_base = (p_cat * self.pesos["catboost"]) + (p_lgb * self.pesos["lightgbm"])
        df['SCORE_RISCO_BRUTO'] = score_base * mults.get("geral", 1.0)
        
        # Normalização 0-100
        max_val = df['SCORE_RISCO_BRUTO'].max() or 1
        df['SCORE_RISCO_FINAL'] = ((df['SCORE_RISCO_BRUTO'] / max_val) * 100).clip(0, 100).round(2)
        
        # Alerta Silencioso (Risco alto em local sem histórico)
        if 'TOTAL_CRIMES' in df.columns:
            df['FLAG_ALERTA_SILENCIOSO'] = (df['TOTAL_CRIMES'] == 0) & (df['SCORE_RISCO_FINAL'] > 75)
        
        df['DT_PROCESSAMENTO'] = datetime.now()

        # SHAP para explicar por que o bairro está perigoso
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
            logger.warning(f"OURO: Aviso no processamento SHAP: {e}")
            
        return df

    def _salvar_parquet_ouro(self, df):
        buffer = io.BytesIO()
        df_save = df.copy()
        for col in self.features_categoricas:
            df_save[col] = df_save[col].astype(str)
        pl.from_pandas(df_save).write_parquet(buffer, compression="lz4")
        key = self._get_path("ouro", "fato_risco_consolidada.parquet")
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue())
        logger.info("OURO: Parquet materializado no R2.")

    def _sincronizar_bq(self, df):
        """Sincronização atômica com BigQuery."""
        tabela_final = f"{self.project_id}.{self.dataset_id}.fato_risco_h3_vigente"
        tabela_staging = f"{tabela_final}_staging"
        
        df_bq = df.copy()
        for col in self.features_categoricas:
            df_bq[col] = df_bq[col].astype(str)

        # Truncate na Staging e MERGE na Final
        job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
        self.bq_client.load_table_from_dataframe(df_bq, tabela_staging, job_config=job_config).result()
        
        # SQL MERGE (Corrigido: sem o 'AS S' no INSERT)
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
        logger.info("OURO: Sincronização BigQuery concluída.")

if __name__ == "__main__":
    ouro = CamadaOuroSafeDriver(dev_mode=False)
    ouro.executar_predicao_atual()
