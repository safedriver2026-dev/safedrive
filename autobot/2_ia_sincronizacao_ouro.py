import pandas as pd
import numpy as np
import os
import traceback
import shap
import json
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from comunicador import RoboComunicador
import holidays
from datetime import datetime

def executar_ia_ouro(robo: RoboComunicador):
    s3 = boto3.client('s3',
                      endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                      aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))
    bucket = os.environ.get("R2_BUCKET_NAME")
    ano = datetime.now().year

    try:
        print("🧠 A iniciar o Motor Preditivo Ouro (Ensemble + SHAP)...")

        caminho_silver_r2 = f"safedriver/datalake/silver/prata_{ano}.parquet"
        s3.download_file(bucket, caminho_silver_r2, "camada_prata.parquet")
        df_prata = pd.read_parquet("camada_prata.parquet")

        features_urbanas = ['DENSIDADE_LOGRADOUROS', 'PROPORCAO_RESIDENCIAL_H3', 'TOTAL_EDIFICACOES_H3']

        hoje = pd.Timestamp.now().normalize()
        FERIADOS = holidays.Brazil(subdiv='SP')
        eh_feriado_hoje = 1 if hoje in FERIADOS else 0
        dia_pagamento_hoje = 1 if (1 <= hoje.day <= 7) or (19 <= hoje.day <= 22) else 0

        df_prata['EH_FERIADO_HOJE'] = eh_feriado_hoje
        df_prata['DIA_PAGAMENTO_HOJE'] = dia_pagamento_hoje

        features_temporais_hoje = ['EH_FERIADO_HOJE', 'DIA_PAGAMENTO_HOJE']

        df_ouro = df_prata.groupby(['INDICE_H3'] + features_urbanas).agg(
            RISCO_OBSERVADO=('PESO_CRIME', 'sum')
        ).reset_index()

        df_ouro['EH_FERIADO_HOJE'] = eh_feriado_hoje
        df_ouro['DIA_PAGAMENTO_HOJE'] = dia_pagamento_hoje

        X = df_ouro[features_urbanas + features_temporais_hoje].fillna(0)
        y = df_ouro['RISCO_OBSERVADO']

        modelo = VotingRegressor([
            ('lgb', LGBMRegressor(n_estimators=150, verbosity=-1, random_state=42)),
            ('cat', CatBoostRegressor(iterations=150, verbose=False, nan_mode='Max', random_seed=42))
        ]).fit(X, y)

        def predict_ensemble(X_input):
            return modelo.predict(X_input)

        explainer = shap.Explainer(predict_ensemble, X)
        shap_values = explainer(X)

        shap_values_array = shap_values.values
        fator_mestre_idx = np.argmax(np.abs(shap_values_array).mean(axis=0))
        fator_mestre_nome = X.columns[fator_mestre_idx]

        df_ouro['SCORE_RISCO'] = np.round(modelo.predict(X), 0).astype(int)
        df_ouro['ULTIMA_SYNC'] = pd.Timestamp.now()

        info_credenciais = json.loads(os.environ.get("BQ_SERVICE_ACCOUNT_JSON"))
        credenciais_bq = service_account.Credentials.from_service_account_info(info_credenciais)
        client = bigquery.Client(credentials=credenciais_bq, project=os.environ.get("BQ_PROJECT_ID"))

        dataset_id = os.environ.get('BQ_DATASET_ID')
        tabela_destino = f"{dataset_id}.tabela_ouro_risco"

        print("☁️ A sincronizar as predições com o BigQuery (Delta Sync)...")
        staging_table_id = f"staging_delta_ouro_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("INDICE_H3", "STRING"),
                bigquery.SchemaField("DENSIDADE_LOGRADOUROS", "FLOAT"),
                bigquery.SchemaField("PROPORCAO_RESIDENCIAL_H3", "FLOAT"),
                bigquery.SchemaField("TOTAL_EDIFICACOES_H3", "FLOAT"),
                bigquery.SchemaField("EH_FERIADO_HOJE", "INTEGER"),
                bigquery.SchemaField("DIA_PAGAMENTO_HOJE", "INTEGER"),
                bigquery.SchemaField("RISCO_OBSERVADO", "INTEGER"),
                bigquery.SchemaField("SCORE_RISCO", "INTEGER"),
                bigquery.SchemaField("ULTIMA_SYNC", "TIMESTAMP"),
            ],
            write_disposition="WRITE_TRUNCATE",
        )

        job = client.load_table_from_dataframe(df_ouro, f"{dataset_id}.{staging_table_id}", job_config=job_config)
        job.result()

        sql_merge = f"""
        MERGE `{tabela_destino}` T 
        USING `{dataset_id}.{staging_table_id}` S 
        ON T.INDICE_H3 = S.INDICE_H3
        WHEN MATCHED AND (ABS(T.SCORE_RISCO - S.SCORE_RISCO) >= 1 OR T.EH_FERIADO_HOJE != S.EH_FERIADO_HOJE OR T.DIA_PAGAMENTO_HOJE != S.DIA_PAGAMENTO_HOJE) THEN
          UPDATE SET SCORE_RISCO = S.SCORE_RISCO, ULTIMA_SYNC = S.ULTIMA_SYNC, EH_FERIADO_HOJE = S.EH_FERIADO_HOJE, DIA_PAGAMENTO_HOJE = S.DIA_PAGAMENTO_HOJE
        WHEN NOT MATCHED THEN
          INSERT (INDICE_H3, DENSIDADE_LOGRADOUROS, PROPORCAO_RESIDENCIAL_H3, TOTAL_EDIFICACOES_H3, EH_FERIADO_HOJE, DIA_PAGAMENTO_HOJE, RISCO_OBSERVADO, SCORE_RISCO, ULTIMA_SYNC) 
          VALUES (S.INDICE_H3, S.DENSIDADE_LOGRADOUROS, S.PROPORCAO_RESIDENCIAL_H3, S.TOTAL_EDIFICACOES_H3, S.EH_FERIADO_HOJE, S.DIA_PAGAMENTO_HOJE, S.RISCO_OBSERVADO, S.SCORE_RISCO, S.ULTIMA_SYNC)
        """
        client.query(sql_merge).result()

        client.delete_table(f"{dataset_id}.{staging_table_id}")

        if os.path.exists("camada_prata.parquet"):
            os.remove("camada_prata.parquet")

        dicionario_shap = {
            "PROPORCAO_RESIDENCIAL_H3": "Perfil Residencial da Zona", 
            "DIA_PAGAMENTO_HOJE": "Ciclo Económico (Dia de Pagamento)", 
            "DENSIDADE_LOGRADOUROS": "Densidade da Malha Viária", 
            "EH_FERIADO_HOJE": "Sazonalidade de Feriados",
            "TOTAL_EDIFICACOES_H3": "Volume Urbano de Edificações"
        }
        fator_explicado = dicionario_shap.get(fator_mestre_nome, fator_mestre_nome)

        insight_discord = f"A auditoria matemática identificou que o fator **{fator_explicado}** foi o principal impulsionador do recálculo de risco neste ciclo."

        robo.enviar_relatorio_operacional(
            "O Mapa de Risco Geocriminal foi recalibrado. O Delta Sync operou com sucesso no BigQuery, ignorando flutuações estatísticas menores.", 
            {"Zonas Monitorizadas": len(df_ouro), "Metodologia": "XAI / Ensemble Learning"}, 
            auditoria_ia=insight_discord
        )

    except Exception:
        robo.enviar_alerta_tecnico("Motor de IA e BigQuery (Ouro)", traceback.format_exc())
