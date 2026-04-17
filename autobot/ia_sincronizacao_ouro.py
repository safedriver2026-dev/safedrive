import polars as pl
import pandas as pd
import h3
import boto3
import joblib
import io
import os
import json
import logging
import shap
import sys
import gc
from botocore.config import Config
from datetime import datetime
from google.cloud import bigquery
from google.api_core import exceptions
from google.oauth2 import service_account
from autobot.calendario_estrategico import CalendarioEstrategico

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfiguracaoOuro:
    NOME_BUCKET = os.getenv("R2_BUCKET_NAME", "").strip()
    ID_PROJETO = os.getenv("BQ_PROJECT_ID", "safe-driver-fc3a9").strip()
    ID_DATASET = os.getenv("BQ_DATASET_ID", "safedriver_gold").strip()
    PESOS_MODELOS = {"catboost": 0.85, "lightgbm": 0.15}
    
    ATRIBUTOS_NUMERICOS = [
        'DENSIDADE', 'TAXA_VACANCIA', 'CONTAGIO_PONDERADO', 'PRESSAO_RISCO_LOCAL', 
        'GRAVIDADE_HISTORICA', 'VOLUME_HISTORICO', 
        'MES_OCORRENCIA', 'DIA_SEMANA', 'IS_PAGAMENTO', 'IS_FDS'
    ]
    ATRIBUTOS_CATEGORICOS = ['NM_BAIRRO', 'NM_MUN', 'PERIODO_DIA', 'PERFIL_ALVO', 'TIPO_LOCAL']
    ATRIBUTOS_COMPLETOS = ATRIBUTOS_NUMERICOS + ATRIBUTOS_CATEGORICOS

class SincronizadorOuro:
    def __init__(self, modo_desenvolvimento: bool = False):
        self.modo_desenvolvimento = modo_desenvolvimento
        self.configuracao = ConfiguracaoOuro()
        self.calendario = CalendarioEstrategico()
        self.cliente_armazenamento = self._inicializar_cliente_armazenamento()
        self.caminho_raiz = self._descobrir_raiz_datalake()
        self.caminho_malha = f"{self.caminho_raiz}/base_geografica/safedriver_geo_base_sp_h3_9.parquet"
        self.cliente_bigquery = self._inicializar_bigquery()

    def _inicializar_cliente_armazenamento(self):
        return boto3.client(
            's3',
            endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(
                signature_version='s3v4',
                s3={'addressing_style': 'path'},
                max_pool_connections=50
            )
        )

    def _descobrir_raiz_datalake(self) -> str:
        try:
            resposta = self.cliente_armazenamento.list_objects_v2(Bucket=self.configuracao.NOME_BUCKET, MaxKeys=50)
            for objeto in resposta.get('Contents', []):
                if "datalake/" in objeto['Key']:
                    return objeto['Key'].split("datalake/")[0] + "datalake"
            return "datalake"
        except Exception:
            return "datalake"

    def _inicializar_bigquery(self):
        credenciais_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
        try:
            informacoes_credencial = json.loads(credenciais_json)
            credenciais = service_account.Credentials.from_service_account_info(informacoes_credencial)
            cliente = bigquery.Client(credentials=credenciais, project=self.configuracao.ID_PROJETO)
            
            referencia_dataset = bigquery.DatasetReference(self.configuracao.ID_PROJETO, self.configuracao.ID_DATASET)
            try:
                cliente.get_dataset(referencia_dataset)
            except exceptions.NotFound:
                dataset = bigquery.Dataset(referencia_dataset)
                dataset.location = "US"
                cliente.create_dataset(dataset)
            return cliente
        except Exception:
            return None

    def _carregar_modelos_treinados(self) -> dict:
        modelos = {"cat": None, "lgb": None}
        for algoritmo in ["cat", "lgb"]:
            caminho = f"{self.caminho_raiz}/modelos_ml/latest_{algoritmo}_geral.pkl"
            try:
                objeto = self.cliente_armazenamento.get_object(Bucket=self.configuracao.NOME_BUCKET, Key=caminho)
                modelos[algoritmo] = joblib.load(io.BytesIO(objeto['Body'].read()))
            except Exception:
                pass
        return modelos

    def _construir_matriz_preditiva(self) -> pd.DataFrame:
        try:
            logger.info("Lendo malha geografica (H3)...")
            resposta_malha = self.cliente_armazenamento.get_object(Bucket=self.configuracao.NOME_BUCKET, Key=self.caminho_malha)
            dataframe_malha = pl.read_parquet(io.BytesIO(resposta_malha['Body'].read())).unique(subset=["H3_INDEX"])
            
            # --- OTIMIZAÇÃO 1: Foco no Biênio (Ano Atual e Ano Passado) ---
            ano_atual = datetime.now().year
            anos_alvo = [ano_atual - 1, ano_atual]
            logger.info(f"Otimizacao de Carga ativada: Lendo historico apenas dos anos {anos_alvo}")
            
            lista_dfs = []
            for ano in anos_alvo:
                caminho_arquivo = f"{self.caminho_raiz}/prata/ssp_consolidada_{ano}.parquet"
                try:
                    resposta = self.cliente_armazenamento.get_object(Bucket=self.configuracao.NOME_BUCKET, Key=caminho_arquivo)
                    lista_dfs.append(pl.read_parquet(io.BytesIO(resposta['Body'].read())))
                except Exception:
                    logger.warning(f"Ano {ano} não encontrado na Prata. Ignorando.")

            if not lista_dfs: 
                logger.error("Sem base historica recente. Abortando.")
                return None
                
            dataframe_prata = pl.concat(lista_dfs, how="vertical")
            dataframe_prata = dataframe_prata.with_columns(pl.col("MES_OCORRENCIA").cast(pl.Int8, strict=False))

            # --- OTIMIZAÇÃO 2: Poda de Hexágonos Fantasmas ---
            # Filtra a malha para conter APENAS os hexágonos que tiveram algum crime registrado.
            # Isso joga fora áreas rurais e florestas, poupando a memória e o limite do GitHub Actions.
            h3_ativos = dataframe_prata.select("H3_INDEX").unique()
            tamanho_malha_antes = dataframe_malha.height
            dataframe_malha = dataframe_malha.join(h3_ativos, on="H3_INDEX", how="inner")
            logger.info(f"Poda Espacial: Malha reduzida de {tamanho_malha_antes} para {dataframe_malha.height} hexagonos urbanos relevantes.")

            memoria_historica = dataframe_prata.group_by(["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO", "MES_OCORRENCIA"]).agg([
                pl.col("GRAVIDADE_HISTORICA").max().alias("GRAVIDADE_HISTORICA"),
                pl.col("VOLUME_HISTORICO").max().alias("VOLUME_HISTORICO"),
                pl.col("CONTAGIO_PONDERADO").max().alias("CONTAGIO_PONDERADO"),
                pl.col("TOTAL_CRIMES").sum().alias("VOLUME_REAL_OCORRIDO") 
            ])

            meses_disponiveis = pl.DataFrame({"MES_OCORRENCIA": pl.Series(range(1, 13), dtype=pl.Int8)})
            periodos = pl.DataFrame({"PERIODO_DIA": ["MANHA", "TARDE", "NOITE", "MADRUGADA"]})
            perfis = pl.DataFrame({"PERFIL_ALVO": ["PEDESTRE", "MOTORISTA"]})
            
            logger.info(f"Novas Dimensoes para Join -> Malha Poda: {dataframe_malha.height} | Meses: 12")
            
            estrutura_base = dataframe_malha.select(["H3_INDEX", "DENSIDADE_AJUSTADA", "TAXA_VACANCIA", "NM_MUN", "NM_BAIRRO"]).join(
                periodos, how="cross"
            ).join(
                perfis, how="cross"
            ).join(
                meses_disponiveis, how="cross"
            )

            dataframe_completo = estrutura_base.join(
                memoria_historica, on=["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO", "MES_OCORRENCIA"], how="left"
            ).with_columns([
                pl.col("GRAVIDADE_HISTORICA").fill_null(0.0).cast(pl.Float32),
                pl.col("VOLUME_HISTORICO").fill_null(0.0).cast(pl.Float32),
                pl.col("CONTAGIO_PONDERADO").fill_null(0.0).cast(pl.Float32),
                pl.col("VOLUME_REAL_OCORRIDO").fill_null(0.0).cast(pl.Float32),
                pl.col("DENSIDADE_AJUSTADA").alias("DENSIDADE").cast(pl.Float32)
            ])

            logger.info("Convertendo para Pandas com seguranca de RAM...")
            dataframe_pandas = dataframe_completo.to_pandas()
            
            del estrutura_base
            del memoria_historica
            del dataframe_completo
            gc.collect()

            dataframe_pandas['PRESSAO_RISCO_LOCAL'] = (dataframe_pandas['CONTAGIO_PONDERADO'] / (dataframe_pandas['DENSIDADE'] + 0.001)).astype('float32')

            contexto_temporal = self.calendario.obter_contexto_ia()
            dataframe_pandas['DIA_SEMANA'] = self.calendario.hoje.weekday()
            dataframe_pandas['IS_PAGAMENTO'] = contexto_temporal['IS_PAGAMENTO']
            dataframe_pandas['IS_FDS'] = contexto_temporal['IS_FDS']
            dataframe_pandas['TIPO_LOCAL'] = "VIA PUBLICA" 
            
            return dataframe_pandas
            
        except Exception as erro:
            logger.error(f"Falha ao construir a matriz base: {erro}")
            return None

    def _exportar_datalake(self, dataframe: pd.DataFrame):
        try:
            dataframe_exportacao = pl.from_pandas(dataframe)
            nome_temp = "temp_ouro_backup.parquet"
            dataframe_exportacao.write_parquet(nome_temp, compression="lz4")
            
            data_atual = datetime.now().strftime("%Y%m%d")
            caminho_arquivo = f"{self.caminho_raiz}/ouro/fato_risco_consolidada_{data_atual}.parquet"
            
            with open(nome_temp, "rb") as f:
                self.cliente_armazenamento.put_object(Bucket=self.configuracao.NOME_BUCKET, Key=caminho_arquivo, Body=f)
            os.remove(nome_temp)
        except Exception:
            pass

    def _exportar_bigquery(self, dataframe: pd.DataFrame) -> bool:
        if not self.cliente_bigquery: return False

        logger.info("Modelando Star Schema e convertendo tipos temporais...")
        dataframe['DT_REF'] = pd.to_datetime(dataframe['DT_REF'])

        dataframe['SK_CENARIO'] = (
            dataframe['PERIODO_DIA'].astype(str) + "_" + 
            dataframe['PERFIL_ALVO'].astype(str) + "_" + 
            dataframe['TIPO_LOCAL'].astype(str)
        ).apply(lambda x: hash(x) % ((sys.maxsize + 1) * 2)).astype(str)

        dataframe['SK_TEMPO'] = (
            dataframe['MES_OCORRENCIA'].astype(str) + "_" + 
            dataframe['DIA_SEMANA'].astype(str) + "_" + 
            dataframe['IS_FDS'].astype(str) + "_" + 
            dataframe['IS_PAGAMENTO'].astype(str)
        ).apply(lambda x: hash(x) % ((sys.maxsize + 1) * 2)).astype(str)

        dim_geografia = dataframe[['H3_INDEX', 'NM_MUN', 'NM_BAIRRO', 'DENSIDADE', 'TAXA_VACANCIA']].drop_duplicates(subset=['H3_INDEX'])
        dim_cenario = dataframe[['SK_CENARIO', 'PERIODO_DIA', 'PERFIL_ALVO', 'TIPO_LOCAL']].drop_duplicates(subset=['SK_CENARIO'])
        dim_tempo = dataframe[['SK_TEMPO', 'MES_OCORRENCIA', 'DIA_SEMANA', 'IS_FDS', 'IS_PAGAMENTO']].drop_duplicates(subset=['SK_TEMPO'])

        colunas_fato = [
            'H3_INDEX', 'SK_CENARIO', 'SK_TEMPO', 'DT_REF',
            'VOLUME_HISTORICO', 'GRAVIDADE_HISTORICA', 'CONTAGIO_PONDERADO', 'PRESSAO_RISCO_LOCAL',
            'SCORE_RISCO_BRUTO', 'SCORE_RISCO_FINAL', 'VOLUME_REAL_OCORRIDO'
        ]
        
        colunas_shap = [c for c in dataframe.columns if c.startswith('SHAP_')]
        
        logger.info("Escrevendo tabela Fato Otimizada...")
        fato_inferencia_risco = dataframe[colunas_fato + colunas_shap]
        fato_inferencia_risco.to_parquet("temp_fato_bq.parquet", engine="pyarrow", compression="lz4", index=False)
        
        del fato_inferencia_risco
        gc.collect()

        try:
            tabelas_para_carregar = {
                "dim_geografia": dim_geografia,
                "dim_cenario": dim_cenario,
                "dim_tempo": dim_tempo,
            }

            for nome_tabela, df_tabela in tabelas_para_carregar.items():
                job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
                id_tabela = f"{self.configuracao.ID_PROJETO}.{self.configuracao.ID_DATASET}.{nome_tabela}"
                self.cliente_bigquery.load_table_from_dataframe(df_tabela, id_tabela, job_config=job_config).result()

            logger.info("Fazendo upload da tabela Fato para o BigQuery...")
            configuracao_fato = bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE", 
                source_format=bigquery.SourceFormat.PARQUET,
                time_partitioning=bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field="DT_REF"),
                clustering_fields=["H3_INDEX", "SK_CENARIO", "SK_TEMPO"] 
            )
            id_tabela_fato = f"{self.configuracao.ID_PROJETO}.{self.configuracao.ID_DATASET}.fato_inferencia_risco"
            
            with open("temp_fato_bq.parquet", "rb") as source_file:
                self.cliente_bigquery.load_table_from_file(source_file, id_tabela_fato, job_config=configuracao_fato).result()
                
            os.remove("temp_fato_bq.parquet")
            return True
        except Exception as erro:
            logger.error(f"Falha ao subir tabelas pro BigQuery: {erro}")
            return False

    def _criar_view_looker_studio(self):
        if not self.cliente_bigquery: return

        dataset_ref = f"{self.configuracao.ID_PROJETO}.{self.configuracao.ID_DATASET}"
        nome_view = f"{dataset_ref}.vw_safedriver_dashboard"
        
        query_sql = f"""
        CREATE OR REPLACE VIEW `{nome_view}` AS
        SELECT 
            f.DT_REF, g.H3_INDEX, g.NM_MUN, g.NM_BAIRRO, g.DENSIDADE, g.TAXA_VACANCIA,
            c.PERIODO_DIA, c.PERFIL_ALVO, c.TIPO_LOCAL,
            t.MES_OCORRENCIA, t.DIA_SEMANA, t.IS_FDS, t.IS_PAGAMENTO,
            f.VOLUME_HISTORICO, f.GRAVIDADE_HISTORICA, f.CONTAGIO_PONDERADO, f.PRESSAO_RISCO_LOCAL,
            f.SCORE_RISCO_FINAL, f.VOLUME_REAL_OCORRIDO,
            (f.SCORE_RISCO_FINAL - f.VOLUME_REAL_OCORRIDO) AS DELTA_ALERTA_IA
        FROM `{dataset_ref}.fato_inferencia_risco` f
        LEFT JOIN `{dataset_ref}.dim_geografia` g ON f.H3_INDEX = g.H3_INDEX
        LEFT JOIN `{dataset_ref}.dim_cenario` c ON f.SK_CENARIO = c.SK_CENARIO
        LEFT JOIN `{dataset_ref}.dim_tempo` t ON f.SK_TEMPO = t.SK_TEMPO;
        """
        try:
            self.cliente_bigquery.query(query_sql).result()
            logger.info("View 'vw_safedriver_dashboard' gerada para o BI.")
        except Exception:
            pass

    def executar_pipeline_preditivo(self) -> bool:
        logger.info("=== Iniciando Pipeline Ouro (Modo Performance) ===")
        try:
            modelos = self._carregar_modelos_treinados()
            if modelos["cat"] is None: return False

            dados_entrada = self._construir_matriz_preditiva()
            if dados_entrada is None or dados_entrada.empty: return False

            for coluna in self.configuracao.ATRIBUTOS_CATEGORICOS:
                dados_entrada[coluna] = dados_entrada[coluna].astype('category')
            for coluna in self.configuracao.ATRIBUTOS_NUMERICOS:
                dados_entrada[coluna] = dados_entrada[coluna].astype('float32')

            logger.info("Executando Inferencia da IA...")
            matriz_features = dados_entrada[self.configuracao.ATRIBUTOS_COMPLETOS]
            
            previsao_catboost = modelos["cat"].predict(matriz_features)
            previsao_lightgbm = modelos["lgb"].predict(matriz_features) if modelos["lgb"] else previsao_catboost

            del matriz_features
            gc.collect()

            dados_entrada['SCORE_RISCO_BRUTO'] = (previsao_catboost * self.configuracao.PESOS_MODELOS["catboost"]) + (previsao_lightgbm * self.configuracao.PESOS_MODELOS["lightgbm"])

            dados_entrada['SCORE_RISCO_FINAL'] = dados_entrada.groupby('PERFIL_ALVO')['SCORE_RISCO_BRUTO'].transform(
                lambda x: ((x - x.min()) / (x.max() - x.min() + 0.001) * 100)
            ).clip(0, 100).round(2)

            dados_entrada['DT_REF'] = datetime.now() 
            
            try:
                explicador = shap.TreeExplainer(modelos["cat"])
                amostra_topo = dados_entrada.nlargest(100, 'SCORE_RISCO_FINAL')
                valores_shap = explicador.shap_values(amostra_topo[self.configuracao.ATRIBUTOS_COMPLETOS])
                for atributo in ['NM_MUN', 'PERIODO_DIA', 'PERFIL_ALVO']:
                    nome_coluna = f'SHAP_{atributo}'
                    dados_entrada[nome_coluna] = 0.0
                    dados_entrada.loc[amostra_topo.index, nome_coluna] = valores_shap[:, self.configuracao.ATRIBUTOS_COMPLETOS.index(atributo)]
            except Exception:
                pass

            self._exportar_datalake(dados_entrada)
            carga_sucesso = self._exportar_bigquery(dados_entrada)
            
            if carga_sucesso:
                self._criar_view_looker_studio() 
                logger.info("=== Pipeline Ouro Concluído com Sucesso ===")
                return True
            return False
            
        except Exception as erro:
            logger.error(f"Falha na Ouro: {erro}")
            return False

if __name__ == "__main__":
    processador_ouro = SincronizadorOuro()
    processamento_ok = processador_ouro.executar_pipeline_preditivo()
    if not processamento_ok:
        sys.exit(1)
