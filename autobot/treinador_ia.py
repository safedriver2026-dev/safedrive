import polars as pl
import pandas as pd
import boto3
import joblib
import io
import os
import json
import logging
from datetime import datetime
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class TreinadorEvolutivo:
    def __init__(self):
        # Conexão R2 (Limpando strings de ambiente)
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        
        self.version = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Mapeamento de Personas e Alvos
        self.personas = {
            "motorista": "TOTAL_CRIMES_MOTORISTA",
            "pedestre": "TOTAL_CRIMES_PEDESTRE",
            "motociclista": "TOTAL_CRIMES_MOTOCICLISTA"
        }
        
        # Definição do Dataset de Treino (Contexto + Tendência + Memória)
        self.features_base = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS']
        self.features_delta = ['DELTA_MOTORISTA', 'DELTA_PEDESTRE', 'DELTA_MOTOCICLISTA']
        self.meta_features = ['ULTIMO_MAE_CAT', 'ULTIMO_MAE_LGB']

    def treinar_modelo_mestre(self):
        """Executa o ciclo completo de treinamento para todas as personas."""
        logger.info(f"IA: Iniciando Treinamento Evolutivo de Produção [{self.version}]")
        
        # 1. Carrega todos os anos da Prata (Datalake completo)
        df_treino = self._carregar_base_completa()
        
        if df_treino is None or df_treino.shape[0] < 50:
            logger.warning("IA: Base insuficiente para treino (0 ou poucas linhas). Abortando.")
            return False

        colunas_modelo = self.features_base + self.features_delta + self.meta_features
        logger.info(f"IA: Base consolidada com {df_treino.shape[0]} registros e {len(colunas_modelo)} features.")

        for persona, target in self.personas.items():
            try:
                logger.info(f"--- Evoluindo Persona: {persona.upper()} ---")
                
                X = df_treino[colunas_modelo]
                y = df_treino[target]
                
                # Split Temporal Adaptativo
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Algoritmo 1: CatBoost (Robusto a outliers de criminalidade)
                model_cat = CatBoostRegressor(iterations=800, depth=6, learning_rate=0.04, verbose=0)
                model_cat.fit(X_train, y_train)
                mae_cat = mean_absolute_error(y_test, model_cat.predict(X_test))

                # Algoritmo 2: LightGBM (Ágil para captar micro-tendências)
                model_lgb = LGBMRegressor(n_estimators=300, learning_rate=0.03, verbosity=-1)
                model_lgb.fit(X_train, y_train)
                mae_lgb = mean_absolute_error(y_test, model_lgb.predict(X_test))

                logger.info(f"📊 {persona}: MAE Cat: {mae_cat:.4f} | MAE LGB: {mae_lgb:.4f}")

                # Versionamento de Modelos no R2
                self._exportar_modelo(model_cat, f"cat_{persona}")
                self._exportar_modelo(model_lgb, f"lgb_{persona}")
                
                # Salva a performance para ser a Meta-Feature do próximo ciclo
                self._persistir_performance(persona, mae_cat, mae_lgb)
                
            except Exception as e:
                logger.error(f"Erro no treino da persona {persona}: {e}")

        return True

    def _carregar_base_completa(self):
        """Une todos os arquivos Parquet da Prata em um único DataFrame."""
        lista_dfs = []
        prefixo = "safedriver/datalake/prata/"
        
        try:
            resp_objs = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefixo)
            for obj in resp_objs.get('Contents', []):
                if obj['Key'].endswith('.parquet'):
                    resp = self.s3.get_object(Bucket=self.bucket, Key=obj['Key'])
                    df_ano = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                    lista_dfs.append(df_ano)
            
            if not lista_dfs: return None
            
            # Concatenação diagonal para lidar com possíveis variações de colunas entre anos
            full_df = pl.concat(lista_dfs, how="diagonal").to_pandas().fillna(0)
            
            # Injeta as Meta-Features (Último erro conhecido)
            for p in self.personas.keys():
                meta = self._buscar_performance_anterior(p)
                full_df['ULTIMO_MAE_CAT'] = meta.get('mae_cat', 0.0)
                full_df['ULTIMO_MAE_LGB'] = meta.get('mae_lgb', 0.0)
                
            return full_df
        except Exception as e:
            logger.error(f"Erro ao consolidar Datalake para treino: {e}")
            return None

    def _buscar_performance_anterior(self, persona):
        path = f"modelos_ml/meta_perf_{persona}.json"
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path)
            return json.loads(resp['Body'].read())
        except:
            return {"mae_cat": 0.0, "mae_lgb": 0.0}

    def _persistir_performance(self, persona, mae_cat, mae_lgb):
        path = f"modelos_ml/meta_perf_{persona}.json"
        meta = {"mae_cat": mae_cat, "mae_lgb": mae_lgb, "versao": self.version}
        self.s3.put_object(Bucket=self.bucket, Key=path, Body=json.dumps(meta))

    def _exportar_modelo(self, modelo, nome_base):
        """Salva a versão datada e o ponteiro 'latest' para a Camada Ouro."""
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()
        
        # Versionamento histórico
        self.s3.put_object(
            Bucket=self.bucket, 
            Key=f"modelos_ml/versions/v{self.version}_{nome_base}.pkl", 
            Body=payload
        )
        # Ponteiro de produção
        self.s3.put_object(
            Bucket=self.bucket, 
            Key=f"modelos_ml/latest_{nome_base}.pkl", 
            Body=payload
        )
        logger.info(f"💾 Modelo {nome_base} persistido no R2.")

if __name__ == "__main__":
    TreinadorEvolutivo().treinar_modelo_mestre()
