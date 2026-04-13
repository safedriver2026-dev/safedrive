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

# Configuração de logs para visibilidade total no GitHub Actions
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class TreinadorEvolutivo:
    def __init__(self):
        # Configurações do R2 (Limpando URLs e espaços)
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
        
        # Identificador de Versão (MLOps)
        self.version = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Definição das Personas
        self.personas = {
            "motorista": "TOTAL_CRIMES_MOTORISTA",
            "pedestre": "TOTAL_CRIMES_PEDESTRE",
            "motociclista": "TOTAL_CRIMES_MOTOCICLISTA"
        }
        
        # Features do Sistema (Base + Memória de Performance)
        self.features_base = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS']
        self.meta_features = ['ULTIMO_MAE_CAT', 'ULTIMO_MAE_LGB']

    def treinar_modelo_mestre(self):
        logger.info(f"IA: Iniciando Ciclo de MLOps [{self.version}]")
        
        # 1. Carrega dados históricos com o caminho corrigido (safedriver/datalake/prata/)
        df_treino = self._carregar_base_com_meta_features()
        
        # Lógica de Cold Start: Se não houver dados no R2, cria modelos base para destravar o pipeline
        if df_treino is None or df_treino.shape[0] < 5:
            logger.warning("IA: Base insuficiente ou não encontrada no R2. Iniciando modo Cold Start...")
            return self._gerar_modelos_cold_start()

        features_finais = self.features_base + self.meta_features
        logger.info(f"IA: Treinando com {df_treino.shape[0]} registros e features: {features_finais}")

        for persona, target in self.personas.items():
            try:
                X = df_treino[features_finais]
                y = df_treino[target]
                
                # Split adaptativo
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Treino CatBoost
                model_cat = CatBoostRegressor(iterations=500, depth=6, learning_rate=0.1, verbose=0)
                model_cat.fit(X_train, y_train)
                mae_cat = mean_absolute_error(y_test, model_cat.predict(X_test))

                # Treino LightGBM
                model_lgb = LGBMRegressor(n_estimators=100, learning_rate=0.05, verbosity=-1)
                model_lgb.fit(X_train, y_train)
                mae_lgb = mean_absolute_error(y_test, model_lgb.predict(X_test))

                logger.info(f"📊 {persona.upper()} | MAE Cat: {mae_cat:.4f} | MAE LGB: {mae_lgb:.4f}")

                # Exportação Versionada
                self._exportar_modelo(model_cat, f"cat_{persona}")
                self._exportar_modelo(model_lgb, f"lgb_{persona}")
                
                # Salva metadados de performance para servir de feature no amanhã
                self._persistir_metadados_performance(persona, mae_cat, mae_lgb)
            except Exception as e:
                logger.error(f"IA: Erro ao treinar persona {persona}: {e}")

        return True

    def _carregar_base_com_meta_features(self):
        """Busca dados na Prata e injeta os últimos MAEs conhecidos."""
        df = self._carregar_base_historica_direta() 
        if df is None: return None
        
        for persona in self.personas.keys():
            meta = self._buscar_metadados_performance(persona)
            df['ULTIMO_MAE_CAT'] = meta.get('mae_cat', 0.0)
            df['ULTIMO_MAE_LGB'] = meta.get('mae_lgb', 0.0)
        return df

    def _carregar_base_historica_direta(self):
        """Acesso direto aos arquivos conforme o caminho da sua imagem do R2."""
        lista_dfs = []
        ano_atual = datetime.now().year
        
        # O prefixo 'safedriver/' foi adicionado para bater com o print do seu console
        for ano in range(2022, ano_atual + 1):
            path = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
            try:
                resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                df_temp = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                
                # Normalização de colunas
                if "TOTAL_NAO_RESIDENCIAIS_H3" in df_temp.columns:
                    df_temp = df_temp.rename({"TOTAL_NAO_RESIDENCIAIS_H3": "TOTAL_NAO_RES_H3"})
                
                lista_dfs.append(df_temp)
                logger.info(f"✅ Arquivo {ano} carregado com sucesso.")
            except:
                continue
        
        if not lista_dfs: return None
        return pl.concat(lista_dfs, how="diagonal").to_pandas().fillna(0)

    def _gerar_modelos_cold_start(self):
        """Gera modelos mínimos para não travar a Camada Ouro na primeira execução."""
        import numpy as np
        X_fake = pd.DataFrame(np.zeros((10, 5)), columns=self.features_base + self.meta_features)
        y_fake = np.zeros(10)
        
        for persona in self.personas.keys():
            m_cat = CatBoostRegressor(iterations=10, verbose=0).fit(X_fake, y_fake)
            m_lgb = LGBMRegressor(n_estimators=10, verbosity=-1).fit(X_fake, y_fake)
            self._exportar_modelo(m_cat, f"cat_{persona}")
            self._exportar_modelo(m_lgb, f"lgb_{persona}")
            self._persistir_metadados_performance(persona, 0.0, 0.0)
            
        logger.info("IA: Modelos de Cold Start criados para manter o pipeline ativo.")
        return True

    def _buscar_metadados_performance(self, persona):
        path = f"modelos_ml/meta_perf_{persona}.json"
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path)
            return json.loads(resp['Body'].read())
        except:
            return {"mae_cat": 0.0, "mae_lgb": 0.0}

    def _persistir_metadados_performance(self, persona, mae_cat, mae_lgb):
        path = f"modelos_ml/meta_perf_{persona}.json"
        meta = {"mae_cat": mae_cat, "mae_lgb": mae_lgb, "versao": self.version}
        self.s3.put_object(Bucket=self.bucket, Key=path, Body=json.dumps(meta))

    def _exportar_modelo(self, modelo, nome_base):
        """Salva a versão histórica e o ponteiro latest."""
        path_v = f"modelos_ml/versions/v{self.version}_{nome_base}.pkl"
        path_l = f"modelos_ml/latest_{nome_base}.pkl"
        
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()

        try:
            self.s3.put_object(Bucket=self.bucket, Key=path_v, Body=payload)
            self.s3.put_object(Bucket=self.bucket, Key=path_l, Body=payload)
            logger.info(f"💾 {nome_base} exportado/versionado.")
        except Exception as e:
            logger.error(f"Erro ao exportar {nome_base}: {e}")

if __name__ == "__main__":
    TreinadorEvolutivo().treinar_modelo_mestre()
