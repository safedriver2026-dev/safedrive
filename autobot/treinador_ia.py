import polars as pl
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import shap
import joblib
import boto3
import os
import io
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TreinadorSafeDriver:
    def __init__(self):
        try:
            self.s3 = boto3.client('s3',
                                    endpoint_url=os.getenv("R2_ENDPOINT_URL"),
                                    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
                                    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"))
            self.bucket = os.getenv("R2_BUCKET_NAME")
        except Exception as e:
            logger.error(f"Erro ao conectar ao R2: {e}")
            raise

        self.personas = ["MOTORISTA", "PEDESTRE", "MOTOCICLISTA"]
        
 
        self.features_modelo = [
            'DENSIDADE_TOTAL_ENDERECOS', 
            'TAXA_RESIDENCIAL',
           
        ]
        
        self.modelos_treinados = {}

    def _carregar_e_preparar_dados(self):
        logger.info(" 1. Carregando dados da Camada Prata e Master Geo Table...")
        
      
        resp_prata = self.s3.get_object(Bucket=self.bucket, Key="datalake/prata/ssp_consolidada_2024.parquet")
        lf_crimes = pl.scan_parquet(io.BytesIO(resp_prata['Body'].read()))

      
        resp_ibge = self.s3.get_object(Bucket=self.bucket, Key="datalake/base_geografica/ibge_censo_enderecos.parquet")
        lf_ibge = pl.scan_parquet(io.BytesIO(resp_ibge['Body'].read()))

        logger.info(" 2. Agregando crimes por Hexágono (H3) e Persona...")
      
        lf_risco_alvo = (
            lf_crimes
            .group_by(["H3_INDEX", "GEO_CD_SETOR", "PERFIL_PERSONA"])
            .agg(pl.len().alias("TOTAL_CRIMES")) 
        )

        logger.info(" 3. Cruzando com variáveis do IBGE...")
       
        lf_treino_completo = lf_risco_alvo.join(
            lf_ibge.select(["GEO_CD_SETOR", "DENSIDADE_TOTAL_ENDERECOS", "TAXA_RESIDENCIAL"]),
            on="GEO_CD_SETOR",
            how="left"
        ).fill_null(0) 

        return lf_treino_completo.collect().to_pandas()

    def executar_treinamento(self):
        logger.info(" INICIANDO LABORATÓRIO DE INTELIGÊNCIA ARTIFICIAL")
        df_historico = self._carregar_e_preparar_dados()

        for persona in self.personas:
            logger.info(f"\n" + "="*40)
            logger.info(f" TREINANDO REDE PARA: {persona}")
            logger.info("="*40)

            
            df_persona = df_historico[df_historico['PERFIL_PERSONA'] == persona].copy()
            
            if df_persona.empty:
                logger.warning(f"⚠️ Sem dados suficientes para treinar {persona}. Pulando...")
                continue

            X = df_persona[self.features_modelo]
            y = df_persona['TOTAL_CRIMES']

           
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

           
            modelo = CatBoostRegressor(
                iterations=1000,         
                learning_rate=0.03,      
                depth=6,                 
                loss_function='RMSE',
                eval_metric='MAE',
                random_seed=42,
                verbose=200              
            )

           
            logger.info(f" Ajustando os pesos (Fit) em {len(X_train)} hexágonos...")
       
            modelo.fit(
                X_train, y_train, 
                eval_set=(X_test, y_test), 
                early_stopping_rounds=50
            )

           
            predicoes = modelo.predict(X_test)
            mae = mean_absolute_error(y_test, predicoes)
            rmse = mean_squared_error(y_test, predicoes, squared=False)
            
            logger.info(f" RESULTADO DA AVALIAÇÃO ({persona}):")
            logger.info(f"   - Erro Médio Absoluto (MAE): {mae:.3f} crimes/hexágono")
            logger.info(f"   - Raiz do Erro Quadrático (RMSE): {rmse:.3f}")

            self.modelos_treinados[persona] = modelo
            
         
            self._extrair_explicabilidade(modelo, X_train, persona)

        self._exportar_modelos()

    def _extrair_explicabilidade(self, modelo, X_train, persona):
        logger.info(f" Extraindo DNA da Predição (SHAP) para {persona}...")
        
      
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X_train)
        
        importancia = pd.DataFrame({
            'Variavel': self.features_modelo,
            'Impacto_Absoluto': abs(shap_values).mean(axis=0)
        }).sort_values(by='Impacto_Absoluto', ascending=False)
        
        logger.info(f" Fatores Direcionadores de Risco para {persona}:")
        logger.info("\n" + importancia.to_string(index=False))

    def _exportar_modelos(self):
        logger.info("\n Salvando Cérebro da IA (.pkl) no Cloudflare R2...")
        
        for persona, modelo in self.modelos_treinados.items():
            buffer = io.BytesIO()
            joblib.dump(modelo, buffer)
            buffer.seek(0)
            
            caminho_modelo = f"modelos_ml/catboost_{persona.lower()}.pkl"
            
            self.s3.put_object(
                Bucket=self.bucket, 
                Key=caminho_modelo, 
                Body=buffer.getvalue()
            )
            logger.info(f"✅ Modelo {persona} salvo: {caminho_modelo}")


if __name__ == "__main__":
    treinador = TreinadorSafeDriver()
    treinador.executar_treinamento()
