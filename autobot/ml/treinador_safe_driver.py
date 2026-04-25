import os
import io
import json
import boto3
import polars as pl
import pandas as pd
import numpy as np
import time
import warnings
from datetime import datetime
from catboost import CatBoostRegressor, Pool
from botocore.config import Config

warnings.filterwarnings("ignore", category=FutureWarning)

class TreinadorSafeDriver:
    """
    Motor de Treinamento Anti-Diluição.
    Implementa Tweedie Regression para dados zero-inflados.
    """
    def __init__(self):
        # 1. IDENTIDADE SINCRONIZADA
        self.projeto = os.getenv("NOME_PROJETO", "safedriver").strip().lower()
        self.bucket = os.getenv("R2_BUCKET_NAME", "safedriver").strip()
        
        # 2. INFRAESTRUTURA R2 (Configuração Ultra-Compatível)
        endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        
        # R2 exige 'auto' como região e 'path' como estilo de endereçamento em muitos casos
        self.s3 = boto3.client(
            's3', 
            endpoint_url=endpoint,
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            region_name='auto', 
            config=Config(
                signature_version='s3v4',
                retries={'max_attempts': 5},
                s3={'addressing_style': 'path'} 
            )
        )
        
        # 3. CAMINHOS E METADADOS
        self.ouro_dir = "datalake/ouro"
        self.modelo_nome = "modelo_safedriver_catboost.cbm"
        self.log_nome = "AUDITORIA_TREINAMENTO_MODELO.json"
        self.target = "LABEL_PESO_RISCO"
        self.cat_features = [
            "H3_INDEX", "SAZON_PERIODO", "FEAT_DIA_SEMANA", "FEAT_MES", 
            "FEAT_PERFIL_VITIMA", "FEAT_CONTEXTO_CRITICO", "FEAT_TIPO_DIA"
        ]
        
        self.telemetria = {
            "projeto": "SafeDriver",
            "instancia_bucket": self.bucket,
            "timestamp": str(datetime.now()),
            "metricas_final": {},
            "importancia_features": {},
            "distribuicao_target": {}
        }

    def _sincronizar_r2(self, local_path, s3_key):
        print(f"☁️ Enviando artefato: {local_path} -> {s3_key}...", flush=True)
        self.s3.upload_file(local_path, self.bucket, s3_key)

    def carregar_dados(self):
        """Tenta carregar a ABT Ouro com lógica de contingência."""
        # Caminho oficial: datalake/ouro/safedriver_abt_treino.parquet
        key_ouro = f"{self.ouro_dir}/{self.projeto}_abt_treino.parquet"
        
        print(f"🕵️ Buscando ABT Ouro no bucket '{self.bucket}'...")
        print(f"📍 Key alvo: {key_ouro}")
        
        try:
            # Tenta a leitura direta
            obj = self.s3.get_object(Bucket=self.bucket, Key=key_ouro)
            df = pl.read_parquet(io.BytesIO(obj['Body'].read()))
            
            # Telemetria de sinais
            zeros = df.filter(pl.col(self.target) == 0).height
            total = df.height
            self.telemetria["distribuicao_target"] = {
                "total_registros": total,
                "percentual_zero": round((zeros / total) * 100, 2),
                "media_risco": round(df[self.target].mean(), 4)
            }
            print("✅ Dados carregados com sucesso!")
            return df

        except Exception as e:
            print(f"❌ Falha direta na Key '{key_ouro}'. Iniciando varredura de emergência...")
            
            # VARREDURA DE EMERGÊNCIA: Lista tudo no bucket para achar o arquivo
            try:
                paginator = self.s3.get_paginator('list_objects_v2')
                found_keys = []
                for page in paginator.paginate(Bucket=self.bucket):
                    if 'Contents' in page:
                        for obj in page['Contents']:
                            found_keys.append(obj['Key'])
                
                if found_keys:
                    print("📂 Arquivos encontrados no bucket:")
                    for k in found_keys: print(f"  -> {k}")
                else:
                    print("⚠️ O bucket parece estar vazio ou o Token não tem permissão de listagem.")
            except Exception as list_err:
                print(f"🚨 Erro crítico de permissão/conexão: {list_err}")
                print("💡 Verifique se o seu Token API no R2 tem permissão 'Admin' ou 'ListBucket'.")
            
            raise e

    def treinar(self):
        inicio_geral = time.time()
        
        # 1. Carga e conversão
        df = self.carregar_dados()
        df_pd = df.to_pandas()

        # 2. Seleção de Features (Elimina IDs e nomes de texto puro)
        colunas_excluir = [self.target, "DATAOCORRENCIA", "ANO_JOIN", "CIDADE", "BAIRRO", "BAIRRO_right"]
        features = [col for col in df_pd.columns if col in self.cat_features or "MACRO_" in col or "FS_" in col or "INFRA_" in col or "CENSO_" in col]
        
        X = df_pd[features]
        y = df_pd[self.target]
        
        # 3. Dosimetria de Pesos (Amplifica o sinal de crime real em 10x)
        pesos = np.where(y > 0, 10.0, 1.0)

        # 4. Tratamento de Categóricas
        for col in self.cat_features:
            if col in X.columns:
                X[col] = X[col].fillna("DESCONHECIDO").astype(str)

        print(f"🚀 Treinando CatBoost Tweedie ($p=1.6$) para o projeto {self.projeto}...")
        train_pool = Pool(X, y, cat_features=[c for c in self.cat_features if c in features], weight=pesos)

        # Hiperparâmetros otimizados para detecção de anomalias criminais
        params = {
            'iterations': 3000,
            'learning_rate': 0.03,
            'depth': 8,
            'l2_leaf_reg': 2.0,
            'loss_function': 'Tweedie:variance_power=1.6',
            'eval_metric': 'MAE',
            'random_seed': 42,
            'verbose': 250,
            'task_type': 'CPU'
        }

        modelo = CatBoostRegressor(**params)
        modelo.fit(train_pool)

        duracao = round(time.time() - inicio_geral, 2)
        
        # 5. Inteligência e Persistência
        importancias = modelo.get_feature_importance()
        self.telemetria["importancia_features"] = {f: round(i, 2) for f, i in sorted(zip(features, importancias), key=lambda x: x[1], reverse=True)}
        self.telemetria["metricas_final"] = {"tempo_seg": duracao, "mae": modelo.get_best_score().get('learn', {}).get('MAE', 0)}

        modelo.save_model(self.modelo_nome)
        with open(self.log_nome, 'w', encoding='utf-8') as f:
            json.dump(self.telemetria, f, indent=4, ensure_ascii=False)

        # 6. Sincronização Cloud
        self._sincronizar_r2(self.modelo_nome, f"modelos/{self.modelo_nome}")
        self._sincronizar_r2(self.log_nome, f"modelos/{self.log_nome}")

        print(f"\n🏆 TREINAMENTO CONCLUÍDO EM {duracao}s")
        print(f"📊 Top Sinais: {list(self.telemetria['importancia_features'].keys())[:3]}")

if __name__ == "__main__":
    TreinadorSafeDriver().treinar()
