import polars as pl
import pandas as pd
import boto3
from botocore.config import Config
import joblib
import io
import os
import gc
import logging
from datetime import datetime
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_absolute_error

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfiguracaoTreinamento:
    NOME_BUCKET = os.getenv("R2_BUCKET_NAME", "").strip()
    ALVO = "INDICE_GRAVIDADE"
    
    ATRIBUTOS_NUMERICOS = [
        'DENSIDADE', 
        'TAXA_VACANCIA', 
        'CONTAGIO_PONDERADO', 
        'PRESSAO_RISCO_LOCAL', 
        'GRAVIDADE_HISTORICA', 
        'VOLUME_HISTORICO',    
        'MES_OCORRENCIA', 
        'DIA_SEMANA', 
        'IS_PAGAMENTO', 
        'IS_FDS'
    ]
    
    ATRIBUTOS_CATEGORICOS = [
        'NM_BAIRRO', 
        'NM_MUN', 
        'PERIODO_DIA', 
        'PERFIL_ALVO', 
        'TIPO_LOCAL'
    ]
    
    ATRIBUTOS_COMPLETOS = ATRIBUTOS_NUMERICOS + ATRIBUTOS_CATEGORICOS

class TreinadorModelos:
    def __init__(self, modo_desenvolvimento: bool = False):
        self.modo_desenvolvimento = modo_desenvolvimento 
        self.configuracao = ConfiguracaoTreinamento()
        self.cliente_armazenamento = self._inicializar_cliente_armazenamento()
        self.caminho_raiz = self._descobrir_raiz_datalake()
        self.versao_modelo = datetime.now().strftime("%Y%m%d_%H%M")
        self.estatisticas_treinamento = {}

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

    def _obter_caminho(self, camada: str, nome_arquivo: str) -> str:
        return f"{self.caminho_raiz}/{camada}/{nome_arquivo}".replace("//", "/")

    def _carregar_dados_consolidados(self) -> pd.DataFrame:
        lista_dataframes = []
        anos = [datetime.now().year] if self.modo_desenvolvimento else range(2022, datetime.now().year + 1)
        
        for ano in anos:
            chave = self._obter_caminho("prata", f"ssp_consolidada_{ano}.parquet")
            try:
                resposta = self.cliente_armazenamento.get_object(Bucket=self.configuracao.NOME_BUCKET, Key=chave)
                dataframe = pl.read_parquet(io.BytesIO(resposta['Body'].read()))
                
                if "ANO_REF" not in dataframe.columns:
                    dataframe = dataframe.with_columns(pl.lit(ano).cast(pl.Int16).alias("ANO_REF"))
                
                if "GRAVIDADE_HISTORICA" not in dataframe.columns:
                    dataframe = dataframe.with_columns(pl.col("INDICE_GRAVIDADE").alias("GRAVIDADE_HISTORICA"))
                if "VOLUME_HISTORICO" not in dataframe.columns:
                    dataframe = dataframe.with_columns(pl.col("TOTAL_CRIMES").alias("VOLUME_HISTORICO"))

                lista_dataframes.append(dataframe)
            except Exception:
                continue
        
        if not lista_dataframes:
            return None
            
        dataframe_polars = pl.concat(lista_dataframes, how="diagonal")
        
        if dataframe_polars.height > 800000:
            dataframe_polars = dataframe_polars.sample(n=800000, seed=42)

        return dataframe_polars.to_pandas()

    def _exportar_modelo(self, modelo, nome: str):
        chave_versao = self._obter_caminho("modelos_ml/versions", f"v{self.versao_modelo}_{nome}.pkl")
        chave_recente = self._obter_caminho("modelos_ml", f"latest_{nome}.pkl")
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        carga = buffer.getvalue()
        self.cliente_armazenamento.put_object(Bucket=self.configuracao.NOME_BUCKET, Key=chave_versao, Body=carga)
        self.cliente_armazenamento.put_object(Bucket=self.configuracao.NOME_BUCKET, Key=chave_recente, Body=carga)

    def treinar_modelos(self) -> bool:
        dados_completos = self._carregar_dados_consolidados()
        
        if dados_completos is None or len(dados_completos) < 100:
            logger.error("Massa de dados insuficiente.")
            return False

        for coluna in self.configuracao.ATRIBUTOS_CATEGORICOS:
            dados_completos[coluna] = dados_completos[coluna].astype(str).fillna("INDEFINIDO").astype('category')
            
        for coluna in self.configuracao.ATRIBUTOS_NUMERICOS:
            dados_completos[coluna] = pd.to_numeric(dados_completos[coluna], errors='coerce').fillna(0.0).astype('float32')
            
        dados_completos[self.configuracao.ALVO] = dados_completos[self.configuracao.ALVO].astype('float32')

        dados_completos = dados_completos.sort_values(by=['ANO_REF', 'MES_OCORRENCIA'], ascending=[True, True])
        indice_corte = int(len(dados_completos) * 0.8)

        dados_treinamento = dados_completos.iloc[:indice_corte]
        dados_teste = dados_completos.iloc[indice_corte:]

        x_treinamento = dados_treinamento[self.configuracao.ATRIBUTOS_COMPLETOS]
        y_treinamento = dados_treinamento[self.configuracao.ALVO]
        x_teste = dados_teste[self.configuracao.ATRIBUTOS_COMPLETOS]
        y_teste = dados_teste[self.configuracao.ALVO]

        logger.info(f"Treinamento iniciado. {len(x_treinamento)} registros.")
        del dados_completos, dados_treinamento, dados_teste
        gc.collect()

        modelo_catboost = CatBoostRegressor(
            iterations=1200,
            depth=7,
            learning_rate=0.02,
            l2_leaf_reg=7,
            cat_features=self.configuracao.ATRIBUTOS_CATEGORICOS,
            loss_function='Tweedie:variance_power=1.5',
            early_stopping_rounds=100,
            verbose=100
        )
        modelo_catboost.fit(x_treinamento, y_treinamento, eval_set=(x_teste, y_teste))

        modelo_lightgbm = LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.03,
            objective='tweedie',
            tweedie_variance_power=1.6,
            importance_type='gain',
            n_jobs=-1,
            verbosity=-1
        )
        
        modelo_lightgbm.fit(
            x_treinamento, y_treinamento,
            eval_set=[(x_teste, y_teste)],
            eval_metric='mae',
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(period=100)]
        )

        previsao_catboost = modelo_catboost.predict(x_teste)
        previsao_lightgbm = modelo_lightgbm.predict(x_teste)
        
        erro_absoluto_catboost = mean_absolute_error(y_teste, previsao_catboost)
        erro_absoluto_lightgbm = mean_absolute_error(y_teste, previsao_lightgbm)

        logger.info(f"Erro Medio Absoluto CatBoost: {erro_absoluto_catboost:.4f} | Erro Medio Absoluto LightGBM: {erro_absoluto_lightgbm:.4f}")

        self._exportar_modelo(modelo_catboost, "cat_geral")
        self._exportar_modelo(modelo_lightgbm, "lgb_geral")
        
        self.estatisticas_treinamento = {
            "mae": round(min(erro_absoluto_catboost, erro_absoluto_lightgbm), 4),
            "modelo_vencedor": "CatBoost" if erro_absoluto_catboost < erro_absoluto_lightgbm else "LightGBM",
            "data_versao": self.versao_modelo
        }
        return True

    def obter_estatisticas(self):
        return self.estatisticas_treinamento

if __name__ == "__main__":
    treinador = TreinadorModelos(modo_desenvolvimento=False)
    treinador.treinar_modelos()
