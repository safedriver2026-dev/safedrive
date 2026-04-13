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
        
        # Identificador de Versão (Timestamp)
        self.version = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Definição do Core de Inteligência
        self.personas = {
            "motorista": "TOTAL_CRIMES_MOTORISTA",
            "pedestre": "TOTAL_CRIMES_PEDESTRE",
            "motociclista": "TOTAL_CRIMES_MOTOCICLISTA"
        }
        
        # Features Base + Meta-Features (Performance do Passado)
        self.features_base = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS']
        self.meta_features = ['ULTIMO_MAE_CAT', 'ULTIMO_MAE_LGB']

    def treinar_modelo_mestre(self):
        logger.info(f"IA: Iniciando Ciclo de Treinamento Versionado [{self.version}]")
        
        # 1. Carrega dados históricos injetando a "memória" de testes passados
        df_treino = self._carregar_base_com_meta_features()
        
        if df_treino is None or df_treino.shape[0] < 50:
            logger.warning("IA: Base insuficiente para evoluir o modelo.")
            return False

        features_finais = self.features_base + self.meta_features

        for persona, target in self.personas.items():
            logger.info(f"Treinando Ensemble Evolutivo para: {persona.upper()}")
            
            X = df_treino[features_finais]
            y = df_treino[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # --- Treino CatBoost ---
            model_cat = CatBoostRegressor(iterations=600, depth=7, learning_rate=0.08, verbose=0)
            model_cat.fit(X_train, y_train)
            mae_cat = mean_absolute_error(y_test, model_cat.predict(X_test))

            # --- Treino LightGBM ---
            model_lgb = LGBMRegressor(n_estimators=150, learning_rate=0.05, verbosity=-1)
            model_lgb.fit(X_train, y_train)
            mae_lgb = mean_absolute_error(y_test, model_lgb.predict(X_test))

            logger.info(f"Persona {persona.upper()} | MAE Cat: {mae_cat:.4f} | MAE LGB: {mae_lgb:.4f}")

            # 2. Exportação com Versionamento e Ponteiro 'Latest'
            self._exportar_modelo(model_cat, f"cat_{persona}", mae_cat)
            self._exportar_modelo(model_lgb, f"lgb_{persona}", mae_lgb)
            
            # 3. Salva a performance para ser feature na próxima rodada
            self._persistir_metadados_performance(persona, mae_cat, mae_lgb)

        return True

    def _carregar_base_com_meta_features(self):
        """Une os dados da Prata com os resultados dos últimos testes (Features de Memória)."""
        lista_dfs = []
        ano_atual = datetime.now().year
        
        # Busca histórico de 2022 até hoje no R2
        for ano in range(2022, ano_atual + 1):
            path = f"datalake/prata/ssp_consolidada_{ano}.parquet"
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                df_temp = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                
                # Padronização semântica das colunas
                if "TOTAL_NAO_RESIDENCIAIS_H3" in df_temp.columns:
                    df_temp = df_temp.rename({"TOTAL_NAO_RESIDENCIAIS_H3": "TOTAL_NAO_RES_H3"})
                
                lista_dfs.append(df_temp)
            except:
                continue
        
        if not lista_dfs: return None
        
        full_df = pl.concat(lista_dfs, how="diagonal").to_pandas().fillna(0)

        # Injeção de Meta-Features (Último MAE registrado)
        for persona in self.personas.keys():
            meta = self._buscar_metadados_performance(persona)
            full_df['ULTIMO_MAE_CAT'] = meta.get('mae_cat', 0)
            full_df['ULTIMO_MAE_LGB'] = meta.get('mae_lgb', 0)
            
        return full_df

    def _buscar_metadados_performance(self, persona):
        """Lê o JSON de performance anterior do R2."""
        path = f"modelos_ml/meta_perf_{persona}.json"
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path)
            return json.loads(resp['Body'].read())
        except:
            return {"mae_cat": 0, "mae_lgb": 0}

    def _persistir_metadados_performance(self, persona, mae_cat, mae_lgb):
        """Salva a performance atual para virar feature no treino de amanhã."""
        path = f"modelos_ml/meta_perf_{persona}.json"
        meta = {
            "mae_cat": mae_cat,
            "mae_lgb": mae_lgb,
            "ultima_atualizacao": self.version
        }
        self.s3.put_object(Bucket=self.bucket, Key=path, Body=json.dumps(meta))

    def _exportar_modelo(self, modelo, nome_base, mae):
        """Salva a versão datada e o link simbólico 'latest'."""
        # 1. Salva versão histórica (Auditabilidade)
        path_v = f"modelos_ml/versions/v{self.version}_{nome_base}.pkl"
        # 2. Salva como latest (Produção)
        path_l = f"modelos_ml/latest_{nome_base}.pkl"
        
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        conteudo = buffer.getvalue()

        try:
            # Upload da versão datada
            self.s3.put_object(Bucket=self.bucket, Key=path_v, Body=conteudo)
            # Upload do ponteiro latest
            self.s3.put_object(Bucket=self.bucket, Key=path_l, Body=conteudo)
            logger.info(f"Modelo {nome_base} versionado e atualizado como 'latest'.")
        except Exception as e:
            logger.error(f"Erro ao exportar modelo {nome_base}: {e}")

if __name__ == "__main__":
    TreinadorEvolutivo().treinar_modelo_mestre()
