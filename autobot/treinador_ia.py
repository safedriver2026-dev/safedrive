import polars as pl
import pandas as pd
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import joblib
import io
import os
import logging
import numpy as np
from datetime import datetime
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class TreinadorEvolutivo:
    def __init__(self):
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
        self.personas = {"geral": "TOTAL_CRIMES"}
        
        # Features atualizadas baseadas na Camada Ouro (Censo 2022)
        self.features_numericas = ['DENSIDADE', 'TAXA_VACANCIA']
        self.features_categoricas = ['NM_BAIRRO', 'NM_MUN', 'PERFIL_AREA']
        self.metricas_detalhadas = []

    def treinar_modelo_mestre(self):
        df_treino = self._carregar_datalake_consolidado()
        
        if df_treino is None or df_treino.shape[0] < 100:
            logger.error("IA: Dados insuficientes para treinamento.")
            return False

        colunas_ia = self.features_numericas + self.features_categoricas

        for persona, target in self.personas.items():
            try:
                X = df_treino[colunas_ia].copy()
                y = df_treino[target]
                
                # Conversao de categorias exigida pelo LightGBM e CatBoost
                for col in self.features_categoricas:
                    X[col] = X[col].astype(str).fillna("DESCONHECIDO").astype('category')
                
                # Tipagem de seguranca para features matematicas
                for col in self.features_numericas:
                    X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0.0)

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                logger.info(f"IA: Iniciando treinamento CatBoost ({persona})...")
                # Treino CatBoost - Distribuicao Tweedie (Zero-Inflated)
                model_cat = CatBoostRegressor(
                    iterations=1000, 
                    depth=8, 
                    learning_rate=0.03,
                    cat_features=self.features_categoricas, 
                    loss_function='Tweedie:variance_power=1.5',
                    verbose=0
                )
                model_cat.fit(X_train, y_train)
                
                logger.info(f"IA: Iniciando treinamento LightGBM ({persona})...")
                # Treino LightGBM - Distribuicao Tweedie (Zero-Inflated)
                model_lgb = LGBMRegressor(
                    n_estimators=500, 
                    learning_rate=0.02, 
                    num_leaves=64, 
                    objective='tweedie',
                    tweedie_variance_power=1.5,
                    verbosity=-1
                )
                model_lgb.fit(X_train, y_train)

                # Auditoria de Erro Absoluto Medio (MAE)
                mae_cat = mean_absolute_error(y_test, model_cat.predict(X_test))
                mae_lgb = mean_absolute_error(y_test, model_lgb.predict(X_test))
                
                self.metricas_detalhadas.append({
                    "persona": persona,
                    "mae_cat": round(float(mae_cat), 4),
                    "mae_lgb": round(float(mae_lgb), 4),
                    "total_treino": len(X_train),
                    "distribuicao": "Tweedie"
                })

                self._exportar_artefactos(model_cat, f"cat_{persona}")
                self._exportar_artefactos(model_lgb, f"lgb_{persona}")
                
            except Exception as e:
                logger.error(f"IA: Erro no treinamento evolutivo ({persona}): {e}")
                return False

        logger.info("IA: Processamento de treinamento e exportacao de artefatos concluido com sucesso.")
        return True

    def obter_metricas_finais(self):
        """Retorna os KPIs de performance para o orquestrador/Discord."""
        return self.metricas_detalhadas

    def _carregar_datalake_consolidado(self):
        lista_dfs = []
        
        # Colunas esperadas para tipagem estrita no Polars
        cols_num = ["TOTAL_CRIMES", "DENSIDADE", "TAXA_VACANCIA"]
        cols_cat = ["NM_BAIRRO", "NM_MUN", "H3_INDEX"]

        for ano in range(2022, datetime.now().year + 1):
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=f"datalake/prata/ssp_consolidada_{ano}.parquet")
                df = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                
                # Tipagem segura nativa em Rust (Polars)
                df = df.with_columns([
                    pl.col(c).cast(pl.Float64, strict=False).fill_null(0.0) for c in cols_num if c in df.columns
                ]).with_columns([
                    pl.col(c).cast(pl.String).fill_null("DESCONHECIDO") for c in cols_cat if c in df.columns
                ])
                
                # Garante que a coluna TAXA_VACANCIA exista mesmo em processamentos antigos
                if "TAXA_VACANCIA" not in df.columns:
                    df = df.with_columns(pl.lit(0.0).alias("TAXA_VACANCIA"))

                lista_dfs.append(df)
            except Exception: 
                continue
        
        if not lista_dfs: 
            return None
            
        # Converte para Pandas apenas no limite do Scikit-Learn/CatBoost
        df = pl.concat(lista_dfs, how="diagonal").to_pandas()
        
        # Feature Engineering Contextual
        condicoes = [
            (df['TAXA_VACANCIA'] > 0.3),
            (df['DENSIDADE'] > 5000),
            (df['DENSIDADE'] == 0)
        ]
        classes = ["ALTA_VACANCIA", "RESIDENCIAL_DENSO", "COMERCIAL_INDUSTRIAL"]
        
        df['PERFIL_AREA'] = np.select(condicoes, classes, default="MISTO")
        
        return df

    def _exportar_artefactos(self, modelo, nome):
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()
        self.s3.put_object(Bucket=self.bucket, Key=f"datalake/modelos_ml/versions/v{self.versao_modelo}_{nome}.pkl", Body=payload)
        self.s3.put_object(Bucket=self.bucket, Key=f"datalake/modelos_ml/latest_{nome}.pkl", Body=payload)

if __name__ == "__main__":
    treinador = TreinadorEvolutivo()
    treinador.treinar_modelo_mestre()
    print(treinador.obter_metricas_finais())
