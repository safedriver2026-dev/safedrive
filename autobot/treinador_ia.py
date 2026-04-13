import polars as pl
import pandas as pd
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
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

# Configuração de logging padrão corporativo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class TreinadorEvolutivo:
    def __init__(self):
        # Conectividade Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        
        self.versao_modelo = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Alvos da IA
        self.personas = {
            "motorista": "TOTAL_CRIMES_MOTORISTA",
            "pedestre": "TOTAL_CRIMES_PEDESTRE",
            "motociclista": "TOTAL_CRIMES_MOTOCICLISTA"
        }
        
        # Features definidas no Datalake Prata
        self.features_base = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS']
        self.features_delta = ['DELTA_MOTORISTA', 'DELTA_PEDESTRE', 'DELTA_MOTOCICLISTA']
        self.meta_features = ['ULTIMO_MAE_CAT', 'ULTIMO_MAE_LGB']

    def treinar_modelo_mestre(self):
        """Executa o treinamento do Ensemble e versiona os modelos no R2."""
        logger.info(f"IA: Iniciando pipeline de Treinamento Evolutivo [v{self.versao_modelo}]")
        
        df_treino = self._carregar_datalake_consolidado()
        
        if df_treino is None or df_treino.shape[0] < 50:
            logger.warning("IA: Base de dados insuficiente para o treinamento. Abortando.")
            return False

        colunas_ia = self.features_base + self.features_delta + self.meta_features
        logger.info(f"IA: Datalake carregado. Registros: {df_treino.shape[0]} | Features: {len(colunas_ia)}")

        for persona, target in self.personas.items():
            try:
                logger.info(f"--- Treinando Persona: {persona.upper()} ---")
                X = df_treino[colunas_ia]
                y = df_treino[target]
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # CatBoost: Resiliência a Outliers
                model_cat = CatBoostRegressor(iterations=800, depth=6, learning_rate=0.04, verbose=0)
                model_cat.fit(X_train, y_train)
                mae_cat = mean_absolute_error(y_test, model_cat.predict(X_test))

                # LightGBM: Velocidade e Generalização
                model_lgb = LGBMRegressor(n_estimators=300, learning_rate=0.03, verbosity=-1)
                model_lgb.fit(X_train, y_train)
                mae_lgb = mean_absolute_error(y_test, model_lgb.predict(X_test))

                logger.info(f"Resultados [{persona}]: MAE CatBoost: {mae_cat:.4f} | MAE LightGBM: {mae_lgb:.4f}")

                # Persistência
                self._exportar_artefactos(model_cat, f"cat_{persona}")
                self._exportar_artefactos(model_lgb, f"lgb_{persona}")
                self._atualizar_meta_features(persona, mae_cat, mae_lgb)
                
            except Exception as e:
                logger.error(f"Erro no treinamento da persona {persona}: {e}")
                return False

        return True

    def _carregar_datalake_consolidado(self):
        """Carrega todos os anos da Prata garantindo a compatibilidade de tipos (Int32)."""
        lista_dfs = []
        ano_atual = datetime.now().year
        
        for ano in range(2022, ano_atual + 1):
            path = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                df_ano = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                
                # --- VACINA DE TIPOS ---
                # Forçamos todos os inteiros para Int32 (assinado) para permitir o concat
                # Isso resolve o conflito entre UInt32 (arquivos velhos) e Int32 (arquivos novos)
                df_ano = df_ano.with_columns([
                    pl.col(pl.INTEGER_DTYPES).cast(pl.Int32)
                ])
                
                lista_dfs.append(df_ano)
                logger.info(f"IA: Dados de {ano} carregados e tipados (Int32).")
            except ClientError:
                continue
        
        if not lista_dfs: return None
            
        # Concatenação diagonal para lidar com colunas extras (ex: CMD, BTL em 2024+)
        df_consolidado = pl.concat(lista_dfs, how="diagonal").to_pandas().fillna(0)
        
        # Injeção das memórias de erro (Meta-Features)
        for persona in self.personas.keys():
            metricas = self._recuperar_metricas_historicas(persona)
            df_consolidado['ULTIMO_MAE_CAT'] = metricas.get('mae_cat', 0.0)
            df_consolidado['ULTIMO_MAE_LGB'] = metricas.get('mae_lgb', 0.0)
            
        return df_consolidado

    def _recuperar_metricas_historicas(self, persona):
        path = f"safedriver/modelos_ml/meta_perf_{persona}.json"
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path)
            return json.loads(resp['Body'].read())
        except ClientError:
            return {"mae_cat": 0.0, "mae_lgb": 0.0}

    def _atualizar_meta_features(self, persona, mae_cat, mae_lgb):
        path = f"safedriver/modelos_ml/meta_perf_{persona}.json"
        dados_meta = {"mae_cat": mae_cat, "mae_lgb": mae_lgb, "versao": self.versao_modelo}
        self.s3.put_object(Bucket=self.bucket, Key=path, Body=json.dumps(dados_meta))

    def _exportar_artefactos(self, modelo, nome_base):
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()
        
        caminho_versao = f"safedriver/modelos_ml/versions/v{self.versao_modelo}_{nome_base}.pkl"
        caminho_producao = f"safedriver/modelos_ml/latest_{nome_base}.pkl"
        
        self.s3.put_object(Bucket=self.bucket, Key=caminho_versao, Body=payload)
        self.s3.put_object(Bucket=self.bucket, Key=caminho_producao, Body=payload)
        logger.info(f"IA: Modelo {caminho_producao} atualizado.")

if __name__ == "__main__":
    TreinadorEvolutivo().treinar_modelo_mestre()
