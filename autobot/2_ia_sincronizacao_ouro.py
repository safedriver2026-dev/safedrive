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

def executar_ia_ouro(robo: RoboComunicador):
    try:
        print("🧠 A iniciar o Motor Preditivo Ouro (Ensemble + SHAP)...")

        df_prata = pd.read_parquet("camada_prata.parquet")

        features_urbanas = ['DENSIDADE_LOGRADOUROS', 'PROPORCAO_RESIDENCIAL_H3', 'TOTAL_EDIFICACOES_H3']
        features_temporais = ['EH_FERIADO', 'DIA_PAGAMENTO']

        df_ouro = df_prata.groupby(['INDICE_H3'] + features_urbanas + features_temporais).agg(
            RISCO_OBSERVADO=('PESO_CRIME', 'sum')
        ).reset_index()

        X = df_ouro[features_urbanas + features_temporais].fillna(0)
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

        df_ouro['SCORE_RISCO'] = np.round(modelo.predict(X), 0).astype(int) # Preferência por valores numéricos como inteiros
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
                bigquery.SchemaField("EH_FERIADO", "INTEGER"),
                bigquery.SchemaField("DIA_PAGAMENTO", "INTEGER"),
                bigquery.SchemaField("RISCO_OBSERVADO", "INTEGER"), # Alterado para INTEGER
                bigquery.SchemaField("SCORE_RISCO", "INTEGER"), # Alterado para INTEGER
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
        WHEN MATCHED AND ABS(T.SCORE_RISCO - S.SCORE_RISCO) >= 1 THEN -- Comparação com 1 para inteiros
          UPDATE SET SCORE_RISCO = S.SCORE_RISCO, ULTIMA_SYNC = S.ULTIMA_SYNC
        WHEN NOT MATCHED THEN
          INSERT (INDICE_H3, DENSIDADE_LOGRADOUROS, PROPORCAO_RESIDENCIAL_H3, TOTAL_EDIFICACOES_H3, EH_FERIADO, DIA_PAGAMENTO, RISCO_OBSERVADO, SCORE_RISCO, ULTIMA_SYNC) 
          VALUES (S.INDICE_H3, S.DENSIDADE_LOGRADOUROS, S.PROPORCAO_RESIDENCIAL_H3, S.TOTAL_EDIFICACOES_H3, S.EH_FERIADO, S.DIA_PAGAMENTO, S.RISCO_OBSERVADO, S.SCORE_RISCO, S.ULTIMA_SYNC)
        """
        client.query(sql_merge).result()

        client.delete_table(f"{dataset_id}.{staging_table_id}")

        if os.path.exists("camada_prata.parquet"):
            os.remove("camada_prata.parquet")

        dicionario_shap = {
            "PROPORCAO_RESIDENCIAL_H3": "Perfil Residencial da Zona", 
            "DIA_PAGAMENTO": "Ciclo Económico (Dia de Pagamento)", 
            "DENSIDADE_LOGRADOUROS": "Densidade da Malha Viária", 
            "EH_FERIADO": "Sazonalidade de Feriados",
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
