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
        # Conexão R2
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
        
        # O NOVO SET DE FEATURES (Base + Tendência + Memória)
        self.features_base = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS']
        self.features_delta = ['DELTA_MOTORISTA', 'DELTA_PEDESTRE', 'DELTA_MOTOCICLISTA']
        self.meta_features = ['ULTIMO_MAE_CAT', 'ULTIMO_MAE_LGB']

    def treinar_modelo_mestre(self):
        logger.info(f"IA: Iniciando Treinamento Evolutivo [{self.version}]")
        
        # 1. Carrega a base consolidada de todos os anos disponíveis
        df_treino = self._carregar_base_completa()
        
        if df_treino is None or df_treino.shape[0] < 50:
            logger.warning("IA: Dados insuficientes para um treino eficaz. Abortando.")
            return False

        # Lista final de colunas que o modelo vai olhar
        colunas_modelo = self.features_base + self.features_delta + self.meta_features
        logger.info(f"IA: Treinando com {len(colunas_modelo)} features estratégicas.")

        for persona, target in self.personas.items():
            logger.info(f"--- Treinando Persona: {persona.upper()} ---")
            
            X = df_treino[colunas_modelo]
            y = df_treino[target]
            
            # Split Temporal (80/20)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # CatBoost (Foco em Precisão)
            model_cat = CatBoostRegressor(iterations=700, depth=7, learning_rate=0.05, verbose=0)
            model_cat.fit(X_train, y_train)
            mae_cat = mean_absolute_error(y_test, model_cat.predict(X_test))

            # LightGBM (Foco em Generalização)
            model_lgb = LGBMRegressor(n_estimators=200, learning_rate=0.03, verbosity=-1)
            model_lgb.fit(X_train, y_train)
            mae_lgb = mean_absolute_error(y_test, model_lgb.predict(X_test))

            logger.info(f"📈 Resultado {persona}: MAE Cat: {mae_cat:.4f} | MAE LGB: {mae_lgb:.4f}")

            # Salva Versão e Latest no R2
            self._exportar_modelo(model_cat, f"cat_{persona}")
            self._exportar_modelo(model_lgb, f"lgb_{persona}")
            
            # Persiste o erro para ser a "Meta-Feature" de amanhã
            self._salvar_performance_historica(persona, mae_cat, mae_lgb)

        return True

    def _carregar_base_completa(self):
        """Busca todos os arquivos Parquet da Prata e une em um DataFrame mestre."""
        lista_dfs = []
        prefixo = "safedriver/datalake/prata/"
        
        try:
            # Lista arquivos para pegar todos os anos disponíveis
            resp_objs = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefixo)
            for obj in resp_objs.get('Contents', []):
                if obj['Key'].endswith('.parquet'):
                    resp = self.s3.get_object(Bucket=self.bucket, Key=obj['Key'])
                    df_ano = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                    lista_dfs.append(df_ano)
            
            if not lista_dfs: return None
            
            # Concatenação diagonal (trata colunas faltantes entre anos)
            full_df = pl.concat(lista_dfs, how="diagonal").to_pandas().fillna(0)
            
            # Injeção das Meta-Features de Performance
            for persona in self.personas.keys():
                meta = self._buscar_performance_anterior(persona)
                full_df['ULTIMO_MAE_CAT'] = meta.get('mae_cat', 0.0)
                full_df['ULTIMO_MAE_LGB'] = meta.get('mae_lgb', 0.0)
                
            return full_df
        except Exception as e:
            logger.error(f"Erro ao carregar base de treino: {e}")
            return None

    def _buscar_performance_anterior(self, persona):
        path = f"modelos_ml/meta_perf_{persona}.json"
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path)
            return json.loads(resp['Body'].read())
        except:
            return {"mae_cat": 0.0, "mae_lgb": 0.0}

    def _salvar_performance_historica(self, persona, mae_cat, mae_lgb):
        path = f"modelos_ml/meta_perf_{persona}.json"
        meta = {
            "mae_cat": mae_cat,
            "mae_lgb": mae_lgb,
            "data_treino": self.version
        }
        self.s3.put_object(Bucket=self.bucket, Key=path, Body=json.dumps(meta))

    def _exportar_modelo(self, modelo, nome_base):
        """Versionamento MLOps: Salva com data e atualiza o ponteiro de produção."""
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()
        
        # Caminho da Versão (Auditabilidade)
        self.s3.put_object(
            Bucket=self.bucket, 
            Key=f"modelos_ml/versions/v{self.version}_{nome_base}.pkl", 
            Body=payload
        )
        # Caminho Latest (Produção)
        self.s3.put_object(
            Bucket=self.bucket, 
            Key=f"modelos_ml/latest_{nome_base}.pkl", 
            Body=payload
        )
        logger.info(f"💾 Modelo {nome_base} versionado.")

if __name__ == "__main__":
    TreinadorEvolutivo().treinar_modelo_mestre()
