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

# Configuração de Log para máxima visibilidade no GitHub Actions
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
        
        # Identificador de Versão para MLOps
        self.version = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Definição do Core de Negócio
        self.personas = {
            "motorista": "TOTAL_CRIMES_MOTORISTA",
            "pedestre": "TOTAL_CRIMES_PEDESTRE",
            "motociclista": "TOTAL_CRIMES_MOTOCICLISTA"
        }
        
        # Features: Base + Memória de Performance (Meta-Features)
        self.features_base = ['INDICE_RESIDENCIAL', 'TOTAL_NAO_RES_H3', 'DENSIDADE_ENDERECOS']
        self.meta_features = ['ULTIMO_MAE_CAT', 'ULTIMO_MAE_LGB']

    def treinar_modelo_mestre(self):
        logger.info(f"IA: Invocando Ciclo de Treinamento Versionado [{self.version}]")
        
        # 1. Carregamento com Injeção de Meta-Features
        df_treino = self._carregar_base_com_meta_features()
        
        # Validação de Segurança (Cold Start / Data Quality)
        if df_treino is None or df_treino.shape[0] < 50:
            logger.warning(f"IA: Abortando. Base insuficiente para evolução ({df_treino.shape[0] if df_treino is not None else 0} linhas).")
            return False

        features_finais = self.features_base + self.meta_features
        logger.info(f"IA: Treinando com features: {features_finais}")

        for persona, target in self.personas.items():
            logger.info(f"IA: Processando Ensemble para {persona.upper()}")
            
            X = df_treino[features_finais]
            y = df_treino[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # --- Modelo A: CatBoost (O Líder) ---
            model_cat = CatBoostRegressor(iterations=600, depth=7, learning_rate=0.08, verbose=0)
            model_cat.fit(X_train, y_train)
            mae_cat = mean_absolute_error(y_test, model_cat.predict(X_test))

            # --- Modelo B: LightGBM (O Suporte) ---
            model_lgb = LGBMRegressor(n_estimators=150, learning_rate=0.05, verbosity=-1)
            model_lgb.fit(X_train, y_train)
            mae_lgb = mean_absolute_error(y_test, model_lgb.predict(X_test))

            logger.info(f"📊 {persona.upper()} -> MAE Cat: {mae_cat:.4f} | MAE LGB: {mae_lgb:.4f}")

            # 2. Exportação Versionada (MLOps)
            self._exportar_modelo(model_cat, f"cat_{persona}")
            self._exportar_modelo(model_lgb, f"lgb_{persona}")
            
            # 3. Persistência de Performance (Criação da Memória para o próximo treino)
            self._persistir_metadados_performance(persona, mae_cat, mae_lgb)

        return True

    def _carregar_base_com_meta_features(self):
        """Une os dados da Prata com o histórico de erro (Cold Start Friendly)."""
        df = self._carregar_base_historica_debug() 
        
        if df is None: return None
        
        for persona in self.personas.keys():
            meta = self._buscar_metadados_performance(persona)
            df['ULTIMO_MAE_CAT'] = meta.get('mae_cat', 0.0)
            df['ULTIMO_MAE_LGB'] = meta.get('mae_lgb', 0.0)
            
        return df

    def _carregar_base_historica_debug(self):
        """Versão com logs detalhados para rastrear falhas de leitura no R2."""
        lista_dfs = []
        ano_atual = datetime.now().year
        
        logger.info(f"🔍 Buscando arquivos em: {self.bucket}/datalake/prata/")

        for ano in range(2022, ano_atual + 1):
            path = f"datalake/prata/ssp_consolidada_{ano}.parquet"
            try:
                # Validação de existência
                self.s3.head_object(Bucket=self.bucket, Key=path)
                
                resp = self.s3.get_object(Bucket=self.bucket, Key=path)
                df_temp = pl.read_parquet(io.BytesIO(resp['Body'].read()))
                
                if df_temp.shape[0] > 0:
                    # Normalização de nomes de colunas
                    if "TOTAL_NAO_RESIDENCIAIS_H3" in df_temp.columns:
                        df_temp = df_temp.rename({"TOTAL_NAO_RESIDENCIAIS_H3": "TOTAL_NAO_RES_H3"})
                    
                    lista_dfs.append(df_temp)
                    logger.info(f"✅ Sucesso ao carregar {ano} ({df_temp.shape[0]} linhas).")
                else:
                    logger.warning(f"⚠️ Arquivo de {ano} existe mas está vazio.")

            except Exception as e:
                logger.debug(f"❌ Falha no ano {ano}: {str(e)}")
                continue
        
        if not lista_dfs:
            logger.error("🚨 CRÍTICO: Nenhum dado carregado da Prata. O pipeline de ML não tem o que processar.")
            return None
        
        # Consolida tudo e converte para Pandas (exigência do CatBoost/LGBM)
        full_df = pl.concat(lista_dfs, how="diagonal").to_pandas().fillna(0)
        logger.info(f"📈 Base consolidada final: {full_df.shape[0]} registros.")
        return full_df

    def _buscar_metadados_performance(self, persona):
        """Lógica de Cold Start: busca erro anterior ou assume zero."""
        path = f"modelos_ml/meta_perf_{persona}.json"
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path)
            return json.loads(resp['Body'].read())
        except:
            return {"mae_cat": 0.0, "mae_lgb": 0.0}

    def _persistir_metadados_performance(self, persona, mae_cat, mae_lgb):
        """Salva a 'memória' do erro para o treino subsequente."""
        path = f"modelos_ml/meta_perf_{persona}.json"
        meta = {
            "mae_cat": mae_cat,
            "mae_lgb": mae_lgb,
            "versao_treino": self.version
        }
        self.s3.put_object(Bucket=self.bucket, Key=path, Body=json.dumps(meta))

    def _exportar_modelo(self, modelo, nome_base):
        """Salva a versão histórica e o atalho 'latest' para a Ouro."""
        path_version = f"modelos_ml/versions/v{self.version}_{nome_base}.pkl"
        path_latest = f"modelos_ml/latest_{nome_base}.pkl"
        
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        payload = buffer.getvalue()

        try:
            # Salva histórico (Auditabilidade)
            self.s3.put_object(Bucket=self.bucket, Key=path_version, Body=payload)
            # Salva produção (Uso imediato pela Ouro)
            self.s3.put_object(Bucket=self.bucket, Key=path_latest, Body=payload)
            logger.info(f"💾 {nome_base} atualizado no R2.")
        except Exception as e:
            logger.error(f"❌ Erro ao exportar modelo {nome_base}: {e}")

if __name__ == "__main__":
    TreinadorEvolutivo().treinar_modelo_mestre()
