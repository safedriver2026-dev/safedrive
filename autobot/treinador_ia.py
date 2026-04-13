import polars as pl
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import boto3
import os
import io
import logging
from datetime import datetime
from autobot.comunicador import ComunicadorSafeDriver

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class TreinadorEvolutivo:
    def __init__(self):
        self.s3 = boto3.client('s3',
                                endpoint_url=os.getenv("R2_ENDPOINT_URL"),
                                aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
                                aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"))
        self.bucket = os.getenv("R2_BUCKET_NAME")
        self.comunicador = ComunicadorSafeDriver()
        self.features = ['DENSIDADE_TOTAL_ENDERECOS', 'TAXA_RESIDENCIAL']

    def treinar_modelo_mestre(self):
        logger.info("🧠 Iniciando Treinamento Evolutivo (Base 2022+)...")
        
        try:
            # 1. Carregar todos os anos disponíveis na Prata (Escalabilidade)
            dfs_prata = []
            anos_disponiveis = [2022, 2023, 2024, 2025] # O loop detecta o que existe no R2
            
            for ano in anos_disponiveis:
                try:
                    path = f"datalake/prata/ssp_consolidada_{ano}.parquet"
                    resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                    dfs_prata.append(pl.read_parquet(io.BytesIO(resp['Body'].read())))
                    logger.info(f"✅ Ano {ano} incorporado ao treino.")
                except:
                    continue

            # Consolida a base histórica total
            df_total = pl.concat(dfs_prata).to_pandas()

            # 2. Validação "Out-of-Time" (Inteligência de Equipe)
            # Treinamos com o passado e testamos com o "futuro" (ano mais recente)
            ano_max = df_total['ANO_REFERENCIA'].max()
            train = df_total[df_total['ANO_REFERENCIA'] < ano_max]
            test = df_total[df_total['ANO_REFERENCIA'] == ano_max]

            logger.info(f"📊 Treinando com dados até {ano_max-1} e validando em {ano_max}")

            for persona in ["MOTORISTA", "PEDESTRE", "MOTOCICLISTA"]:
                # Filtro por Persona
                df_train_p = train[train['PERFIL_PERSONA'] == persona]
                df_test_p = test[test['PERFIL_PERSONA'] == persona]

                modelo = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, verbose=0)
                
                modelo.fit(df_train_p[self.features], df_train_p['TOTAL_CRIMES'])

                # Avaliação de Performance
                preds = modelo.predict(df_test_p[self.features])
                mae = mean_absolute_error(df_test_p['TOTAL_CRIMES'], preds)
                
                logger.info(f"📈 Persona {persona} - MAE: {mae:.4f}")

                # 3. Persistência do Modelo (.pkl)
                self._salvar_modelo_r2(modelo, persona.lower())

            self.comunicador.relatar_sucesso("Treinamento IA", "10 min", "Modelo Mestre Atualizado")

        except Exception as e:
            logger.error(f"❌ Erro no treinamento: {e}")
            self.comunicador.relatar_erro("Treinador IA", e)

    def _salvar_modelo_r2(self, modelo, nome):
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        buffer.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=f"modelos_ml/catboost_{nome}.pkl", Body=buffer.getvalue())

if __name__ == "__main__":
    TreinadorEvolutivo().treinar_modelo_mestre()
