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

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class TreinadorEvolutivo:
    def __init__(self):
        # 🛡️ SANITIZAÇÃO: .strip() evita erros de conexão nos segredos do GitHub
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        
        self.comunicador = ComunicadorSafeDriver()
        
        # Engenharia de Atributos: Features geradas na Camada Prata
        self.features = [
            'INDICE_RESIDENCIAL', 
            'TOTAL_NAO_RESIDENCIAL', 
            'DENSIDADE_ENDERECOS'
        ]

    def treinar_modelo_mestre(self):
        """
        Carrega o histórico completo da Prata (2022+), treina a IA 
        e valida a acurácia antes de salvar os modelos no R2.
        """
        logger.info("🧠 [IA] Iniciando ciclo de treinamento evolutivo...")
        
        try:
            # 1. Carregamento Escalável de Dados
            dfs_prata = []
            ano_atual = datetime.now().year
            
            for ano in range(2022, ano_atual + 1):
                path = f"datalake/prata/ssp_consolidada_{ano}.parquet"
                try:
                    resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                    dfs_prata.append(pl.read_parquet(io.BytesIO(resp['Body'].read())))
                    logger.info(f"✅ Dados de {ano} carregados para treino.")
                except:
                    logger.warning(f"⚠️ Ano {ano} ainda não disponível na Prata. Pulando.")

            if not dfs_prata:
                logger.error("❌ Erro fatal: Nenhuma base de dados encontrada para treinamento.")
                return False

            # Consolidação da base histórica
            df_total = pl.concat(dfs_prata).to_pandas()

            # 2. Validação Out-of-Time (Inteligência de Produção)
            # Treinamos com o passado e validamos com o ano mais recente disponível
            ano_max = df_total['ANO_REFERENCIA'].max()
            train = df_total[df_total['ANO_REFERENCIA'] < ano_max]
            test = df_total[df_total['ANO_REFERENCIA'] == ano_max]
            
            # Se só tivermos um ano, fazemos split 80/20 padrão
            if train.empty:
                logger.info("ℹ️ Base histórica curta. Usando split aleatório 80/20.")
                from sklearn.model_selection import train_test_split
                train, test = train_test_split(df_total, test_size=0.2, random_state=42)

            # 3. Loop de Treinamento por Persona
            # O SafeDriver personaliza o risco para quem dirige, quem caminha e quem pilota
            for persona in ["MOTORISTA", "PEDESTRE", "MOTOCICLISTA"]:
                logger.info(f"🚀 Treinando cérebro da persona: {persona}")
                
                # O target é o total de crimes que afetam aquela persona específica
                # (Coluna gerada conforme a lógica de filtragem da Prata)
                target = f'TOTAL_CRIMES_{persona}'
                
                model = CatBoostRegressor(
                    iterations=1000,
                    learning_rate=0.03,
                    depth=6,
                    l2_leaf_reg=3,
                    loss_function='MAE', # Foco em minimizar o erro absoluto de ocorrências
                    verbose=False
                )

                model.fit(train[self.features], train[target], eval_set=(test[self.features], test[target]))
                
                # Métrica de Performance (MAE)
                # $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$
                preds = model.predict(test[self.features])
                mae = mean_absolute_error(test[target], preds)
                
                logger.info(f"📊 Acurácia {persona}: MAE de {mae:.4f} crimes por hexágono.")

                # 4. Persistência no R2
                self._salvar_modelo_r2(model, persona.lower())

            logger.info("✨ Todos os modelos foram atualizados e sincronizados no Data Lake.")
            return True

        except Exception as e:
            logger.error(f"❌ Falha no ciclo de IA: {e}")
            self.comunicador.relatar_erro("Treinador IA", str(e))
            return False

    def _salvar_modelo_r2(self, model, nome):
        """Salva o arquivo .pkl diretamente no Cloudflare R2"""
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        
        path_modelo = f"modelos_ml/catboost_{nome}.pkl"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=path_modelo,
            Body=buffer.getvalue()
        )
        logger.info(f"💾 Modelo {nome} exportado para {path_modelo}")

if __name__ == "__main__":
    TreinadorEvolutivo().treinar_modelo_mestre()
