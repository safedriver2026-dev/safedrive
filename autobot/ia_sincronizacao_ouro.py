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
import sys  # Necessário para as Surrogate Keys
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
    
    # ATENÇÃO: VOLUME_REAL_OCORRIDO em quarentena (não entra aqui para evitar Data Leakage na IA)
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
            resposta = self.cliente_armazenamento.get_paginator('list_objects_v2').paginate(Bucket=self.configuracao.NOME_BUCKET, MaxKeys=100)
            for pagina in resposta:
                for objeto in pagina.get('Contents', []):
                    if "datalake/prata/" in objeto['Key']: 
                        return objeto['Key'].split("datalake/")[0] + "datalake"
            return "datalake"
        except Exception:
            return "datalake"

    def _inicializar_bigquery(self):
        """AUTOREGENERAÇÃO 1: Cria o dataset se ele não existir"""
        credenciais_json = os.getenv("BQ_SERVICE_ACCOUNT_JSON", "").strip()
        try:
            informacoes_credencial = json.loads(credenciais_json)
            credenciais = service_account.Credentials.from_service_account_info(informacoes_credencial)
            cliente = bigquery.Client(credentials=credenciais, project=self.configuracao.ID_PROJETO)
            
            referencia_dataset = bigquery.DatasetReference(self.configuracao.ID_PROJETO, self.configuracao.ID_DATASET)
            try:
                cliente.get_dataset(referencia_dataset)
                logger.info(f"Dataset '{self.configuracao.ID_DATASET}' validado.")
            except exceptions.NotFound:
                logger.warning(f"Dataset '{self.configuracao.ID_DATASET}' ausente. Autoregenerando...")
                dataset = bigquery.Dataset(referencia_dataset)
                dataset.location = "US"
                cliente.create_dataset(dataset)
                logger.info("Dataset recriado com sucesso.")
            
            return cliente
        except Exception as erro:
            logger.error(f"Erro de credencial/infraestrutura no BigQuery: {erro}")
            return None

    def _carregar_modelos_treinados(self) -> dict:
        """AUTOREGENERAÇÃO 2: Fallback suave se o modelo não existir no Datalake"""
        modelos = {"cat": None, "lgb": None}
        for algoritmo in ["cat", "lgb"]:
            caminho = f"{self.caminho_raiz}/modelos_ml/latest_{algoritmo}_geral.pkl"
            try:
                objeto = self.cliente_armazenamento.get_object(Bucket=self.configuracao.NOME_BUCKET, Key=caminho)
                modelos[algoritmo] = joblib.load(io.BytesIO(objeto['Body'].read()))
            except Exception:
                logger.error(f"FALHA: Modelo {algoritmo} não encontrado no R2 ({caminho}). O pipeline precisará pular este modelo.")
        return modelos

    def _construir_matriz_preditiva(self) -> pd.DataFrame:
        try:
            resposta_malha = self.cliente_armazenamento.get_object(Bucket=self.configuracao.NOME_BUCKET, Key=self.caminho_malha)
            dataframe_malha = pl.read_parquet(io.BytesIO(resposta_malha['Body'].read())).unique(subset=["H3_INDEX"])
            
            ano_atual = datetime.now().year
            caminho_prata = f"{self.caminho_raiz}/prata/ssp_consolidada_{ano_atual}.parquet"
            resposta_prata = self.cliente_armazenamento.get_object(Bucket=self.configuracao.NOME_BUCKET, Key=caminho_prata)
            dataframe_prata = pl.read_parquet(io.BytesIO(resposta_prata['Body'].read()))
            
            # Extração da Sazonalidade Histórica
            meses_disponiveis = dataframe_prata.select("MES_OCORRENCIA").unique().drop_nulls()

            # Agrupamento resgatando o Volume Real para o Delta do Dashboard
            memoria_historica = dataframe_prata.group_by(["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO", "MES_OCORRENCIA"]).agg([
                pl.col("GRAVIDADE_HISTORICA").max().alias("GRAVIDADE_HISTORICA"),
                pl.col("VOLUME_HISTORICO").max().alias("VOLUME_HISTORICO"),
                pl.col("CONTAGIO_PONDERADO").max().alias("CONTAGIO_PONDERADO"),
                pl.col("TOTAL_CRIMES").sum().alias("VOLUME_REAL_OCORRIDO") 
            ])

            periodos = pl.DataFrame({"PERIODO_DIA": ["MANHA", "TARDE", "NOITE", "MADRUGADA"]})
            perfis = pl.DataFrame({"PERFIL_ALVO": ["PEDESTRE", "MOTORISTA"]})
            
            # Cross Join da malha
            estrutura_base = dataframe_malha.select(["H3_INDEX", "DENSIDADE_AJUSTADA", "TAXA_VACANCIA", "NM_MUN", "NM_BAIRRO"]).join(
                periodos, how="cross"
            ).join(
                perfis, how="cross"
            ).join(
                meses_disponiveis, how="cross"
            )

            # Join da base com a memória
            dataframe_completo = estrutura_base.join(
                memoria_historica, on=["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO", "MES_OCORRENCIA"], how="left"
            ).with_columns([
                pl.col("GRAVIDADE_HISTORICA").fill_null(0.0),
                pl.col("VOLUME_HISTORICO").fill_null(0.0),
                pl.col("CONTAGIO_PONDERADO").fill_null(0.0),
                pl.col("VOLUME_REAL_OCORRIDO").fill_null(0.0),
                pl.col("DENSIDADE_AJUSTADA").alias("DENSIDADE")
            ])

            dataframe_pandas = dataframe_completo.to_pandas()
            dataframe_pandas['PRESSAO_RISCO_LOCAL'] = dataframe_pandas['CONTAGIO_PONDERADO'] / (dataframe_pandas['DENSIDADE'] + 0.001)

            contexto_temporal = self.calendario.obter_contexto_ia()
            dataframe_pandas['DIA_SEMANA'] = self.calendario.hoje.weekday()
            dataframe_pandas['IS_PAGAMENTO'] = contexto_temporal['IS_PAGAMENTO']
            dataframe_pandas['IS_FDS'] = contexto_temporal['IS_FDS']
            dataframe_pandas['TIPO_LOCAL'] = "VIA PUBLICA" 
            
            return dataframe_pandas
        except Exception as erro:
            logger.error(f"Falha ao construir a matriz base (Verifique as tabelas Prata/Malha no R2): {erro}")
            return None

    def _exportar_datalake(self, dataframe: pd.DataFrame):
        try:
            dataframe_exportacao = pl.from_pandas(dataframe)
            buffer = io.BytesIO()
            dataframe_exportacao.write_parquet(buffer, compression="lz4")
            
            data_atual = datetime.now().strftime("%Y%m%d")
            caminho_arquivo = f"{self.caminho_raiz}/ouro/fato_risco_consolidada_{data_atual}.parquet"
            
            self.cliente_armazenamento.put_object(
                Bucket=self.configuracao.NOME_BUCKET, 
                Key=caminho_arquivo, 
                Body=buffer.getvalue()
            )
            logger.info(f"Backup Parquet Ouro salvo com sucesso: {caminho_arquivo}")
        except Exception as erro:
            logger.error(f"Aviso: Falha ao salvar backup no Datalake R2: {erro}")

    def _exportar_bigquery(self, dataframe: pd.DataFrame) -> bool:
        """Exportação otimizada com Esquema Estrela"""
        if not self.cliente_bigquery:
            logger.warning("BigQuery inacessível. Pulando carga analítica.")
            return False

        logger.info("Modelando DataFrame para Star Schema (Autoregenerativo)...")
        df_estrela = dataframe.copy()

        # Criação de Chaves
        df_estrela['SK_CENARIO'] = (
            df_estrela['PERIODO_DIA'].astype(str) + "_" + 
            df_estrela['PERFIL_ALVO'].astype(str) + "_" + 
            df_estrela['TIPO_LOCAL'].astype(str)
        ).apply(lambda x: hash(x) % ((sys.maxsize + 1) * 2)).astype(str)

        df_estrela['SK_TEMPO'] = (
            df_estrela['MES_OCORRENCIA'].astype(str) + "_" + 
            df_estrela['DIA_SEMANA'].astype(str) + "_" + 
            df_estrela['IS_FDS'].astype(str) + "_" + 
            df_estrela['IS_PAGAMENTO'].astype(str)
        ).apply(lambda x: hash(x) % ((sys.maxsize + 1) * 2)).astype(str)

        # Divisão das Dimensões
        dim_geografia = df_estrela[['H3_INDEX', 'NM_MUN', 'NM_BAIRRO', 'DENSIDADE', 'TAXA_VACANCIA']].drop_duplicates(subset=['H3_INDEX'])
        dim_cenario = df_estrela[['SK_CENARIO', 'PERIODO_DIA', 'PERFIL_ALVO', 'TIPO_LOCAL']].drop_duplicates(subset=['SK_CENARIO'])
        dim_tempo = df_estrela[['SK_TEMPO', 'MES_OCORRENCIA', 'DIA_SEMANA', 'IS_FDS', 'IS_PAGAMENTO']].drop_duplicates(subset=['SK_TEMPO'])

        # Separação da Fato
        colunas_fato = [
            'H3_INDEX', 'SK_CENARIO', 'SK_TEMPO', 'DT_REF',
            'VOLUME_HISTORICO', 'GRAVIDADE_HISTORICA', 'CONTAGIO_PONDERADO', 'PRESSAO_RISCO_LOCAL',
            'SCORE_RISCO_BRUTO', 'SCORE_RISCO_FINAL',
            'VOLUME_REAL_OCORRIDO'
        ]
        
        colunas_shap = [c for c in df_estrela.columns if c.startswith('SHAP_')]
        fato_inferencia_risco = df_estrela[colunas_fato + colunas_shap].copy()
        fato_inferencia_risco['DT_REF'] = pd.to_datetime(fato_inferencia_risco['DT_REF'])

        # Carga Autoregenerativa (WRITE_TRUNCATE limpa o obsoleto e cria o novo schema)
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

            configuracao_fato = bigquery.LoadJobConfig(
                write_disposition="WRITE_TRUNCATE", 
                time_partitioning=bigquery.TimePartitioning(type_=bigquery.TimePartitioningType.DAY, field="DT_REF"),
                clustering_fields=["H3_INDEX", "SK_CENARIO", "SK_TEMPO"] 
            )
            id_tabela_fato = f"{self.configuracao.ID_PROJETO}.{self.configuracao.ID_DATASET}.fato_inferencia_risco"
            self.cliente_bigquery.load_table_from_dataframe(fato_inferencia_risco, id_tabela_fato, job_config=configuracao_fato).result()
            logger.info("Estrutura Star Schema atualizada com sucesso no BigQuery.")
            return True
        except Exception as erro:
            logger.error(f"Falha Crítica ao subir tabelas pro BigQuery: {erro}")
            return False

    def _criar_view_looker_studio(self):
        """AUTOREGENERAÇÃO 3: Auditoria de dependências antes de montar a View"""
        if not self.cliente_bigquery:
            return

        dataset_ref = f"{self.configuracao.ID_PROJETO}.{self.configuracao.ID_DATASET}"
        nome_view = f"{dataset_ref}.vw_safedriver_dashboard"
        
        # Check de Segurança (Evita que o Action quebre se a carga da tabela falhou)
        tabelas_necessarias = ['fato_inferencia_risco', 'dim_geografia', 'dim_cenario', 'dim_tempo']
        for tabela in tabelas_necessarias:
            tabela_id = f"{dataset_ref}.{tabela}"
            try:
                self.cliente_bigquery.get_table(tabela_id)
            except exceptions.NotFound:
                logger.error(f"ABORTADO: View não pode ser criada pois a tabela '{tabela}' não existe.")
                return False

        logger.info("Dependências verificadas. Compilando View Analítica...")
        
        query_sql = f"""
        CREATE OR REPLACE VIEW `{nome_view}` AS
        SELECT 
            f.DT_REF,
            g.H3_INDEX,
            g.NM_MUN,
            g.NM_BAIRRO,
            g.DENSIDADE,
            g.TAXA_VACANCIA,
            c.PERIODO_DIA,
            c.PERFIL_ALVO,
            c.TIPO_LOCAL,
            t.MES_OCORRENCIA,
            t.DIA_SEMANA,
            t.IS_FDS,
            t.IS_PAGAMENTO,
            f.VOLUME_HISTORICO,
            f.GRAVIDADE_HISTORICA,
            f.CONTAGIO_PONDERADO,
            f.PRESSAO_RISCO_LOCAL,
            f.SCORE_RISCO_FINAL,
            f.VOLUME_REAL_OCORRIDO,
            (f.SCORE_RISCO_FINAL - f.VOLUME_REAL_OCORRIDO) AS DELTA_ALERTA_IA
        FROM `{dataset_ref}.fato_inferencia_risco` f
        LEFT JOIN `{dataset_ref}.dim_geografia` g ON f.H3_INDEX = g.H3_INDEX
        LEFT JOIN `{dataset_ref}.dim_cenario` c ON f.SK_CENARIO = c.SK_CENARIO
        LEFT JOIN `{dataset_ref}.dim_tempo` t ON f.SK_TEMPO = t.SK_TEMPO;
        """
        
        try:
            self.cliente_bigquery.query(query_sql).result()
            logger.info("Sucesso! View 'vw_safedriver_dashboard' gerada e protegida.")
        except Exception as erro:
            logger.error(f"Erro ao orquestrar a View no BigQuery: {erro}")

    def executar_pipeline_preditivo(self) -> bool:
        logger.info("=== Iniciando Pipeline Ouro (SafeDriver) ===")
        try:
            modelos = self._carregar_modelos_treinados()
            
            # Se o Catboost (Modelo principal) não carregou, não temos como predizer.
            if modelos["cat"] is None:
                logger.error("Interrompendo Ouro: Modelo preditivo ausente ou corrompido.")
                return False

            dados_entrada = self._construir_matriz_preditiva()
            if dados_entrada is None or dados_entrada.empty: 
                logger.error("Interrompendo Ouro: Matriz gerou resultado vazio.")
                return False

            # Preparação de Features Segura
            matriz_features = dados_entrada[self.configuracao.ATRIBUTOS_COMPLETOS].copy()
            for coluna in self.configuracao.ATRIBUTOS_CATEGORICOS:
                matriz_features[coluna] = matriz_features[coluna].astype('category')
            for coluna in self.configuracao.ATRIBUTOS_NUMERICOS:
                matriz_features[coluna] = matriz_features[coluna].astype('float32')

            # Predição
            previsao_catboost = modelos["cat"].predict(matriz_features)
            previsao_lightgbm = modelos["lgb"].predict(matriz_features) if modelos["lgb"] else previsao_catboost

            # Ensemble
            pontuacao_bruta = (previsao_catboost * self.configuracao.PESOS_MODELOS["catboost"]) + (previsao_lightgbm * self.configuracao.PESOS_MODELOS["lightgbm"])
            dados_entrada['SCORE_RISCO_BRUTO'] = pontuacao_bruta

            # Normalização (0 a 100) por Perfil Alvo
            dados_entrada['SCORE_RISCO_FINAL'] = dados_entrada.groupby('PERFIL_ALVO')['SCORE_RISCO_BRUTO'].transform(
                lambda x: ((x - x.min()) / (x.max() - x.min() + 0.001) * 100)
            ).clip(0, 100).round(2)

            dados_entrada['DT_REF'] = datetime.now() 
            
            # SHAP Explainer (Opcional, protegido por try/except para não quebrar a carga principal)
            try:
                explicador = shap.TreeExplainer(modelos["cat"])
                amostra_topo = dados_entrada.nlargest(1000, 'SCORE_RISCO_FINAL')
                valores_shap = explicador.shap_values(amostra_topo[self.configuracao.ATRIBUTOS_COMPLETOS])
                
                for atributo in ['NM_MUN', 'PERIODO_DIA', 'PERFIL_ALVO']:
                    nome_coluna = f'SHAP_{atributo}'
                    dados_entrada[nome_coluna] = 0.0
                    indice_atributo = self.configuracao.ATRIBUTOS_COMPLETOS.index(atributo)
                    dados_entrada.loc[amostra_topo.index, nome_coluna] = valores_shap[:, indice_atributo]
            except Exception as shap_erro:
                logger.warning(f"Cálculo SHAP ignorado por instabilidade computacional: {shap_erro}")

            # Persistência
            self._exportar_datalake(dados_entrada)
            carga_sucesso = self._exportar_bigquery(dados_entrada)
            
            if carga_sucesso:
                self._criar_view_looker_studio() 
                logger.info("=== Pipeline Ouro Concluído com Sucesso ===")
                return True
            else:
                logger.error("Pipeline Finalizado com falhas na exportação.")
                return False
            
        except Exception as erro:
            logger.error(f"Falha estrutural não mapeada na camada analítica: {erro}")
            return False

if __name__ == "__main__":
    processador_ouro = SincronizadorOuro()
    processamento_ok = processador_ouro.executar_pipeline_preditivo()
    
    # Sai com código de erro 1 se falhar, para que o GitHub Actions sinalize "Failed" no Card
    if not processamento_ok:
        sys.exit(1)
