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
from sklearn.metrics import r2_score, mean_absolute_error

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
        caminho_crime_real_agregado_r2 = f"safedriver/datalake/validation/crime_real_agregado_{ano}.parquet"

        s3.download_file(bucket, caminho_silver_r2, "camada_prata.parquet")
        df_prata = pd.read_parquet("camada_prata.parquet")

        # Baixa os dados de crime real agregado para validação
        s3.download_file(bucket, caminho_crime_real_agregado_r2, "crime_real_agregado.parquet")
        df_crime_real_agregado = pd.read_parquet("crime_real_agregado.parquet")

        features_urbanas = ['DENSIDADE_LOGRADOUROS', 'PROPORCAO_RESIDENCIAL_H3', 'TOTAL_EDIFICACOES_H3']
        features_temporais = ['EH_FERIADO', 'DIA_PAGAMENTO', 'DIA_SEMANA']
        features_modelo = ['INDICE_H3'] + features_urbanas + features_temporais + ['PESO_CRIME']

        df_modelo = df_prata[features_modelo].copy()

        # Agrega o PESO_CRIME por INDICE_H3 para o RISCO_OBSERVADO histórico
        df_modelo_agregado = df_modelo.groupby('INDICE_H3').agg(
            RISCO_OBSERVADO=('PESO_CRIME', 'sum'),
            **{col: ('PESO_CRIME', 'first') for col in features_urbanas}, # Mantém as features urbanas
            **{col: ('PESO_CRIME', 'first') for col in features_temporais} # Mantém as features temporais (serão sobrescritas)
        ).reset_index()

        # Calcula os fatores temporais para HOJE
        hoje = pd.Timestamp.now().normalize()
        feriados_sp = holidays.Brazil(subdiv='SP')
        eh_feriado_hoje = 1 if hoje in feriados_sp else 0
        dia_pagamento_hoje = 1 if (1 <= hoje.day <= 7) or (19 <= hoje.day <= 22) else 0
        dia_semana_hoje = hoje.dayofweek

        # Sobrescreve as features temporais com os valores de HOJE
        df_modelo_agregado['EH_FERIADO'] = eh_feriado_hoje
        df_modelo_agregado['DIA_PAGAMENTO'] = dia_pagamento_hoje
        df_modelo_agregado['DIA_SEMANA'] = dia_semana_hoje

        # Seleciona as features para o modelo de IA
        X = df_modelo_agregado[features_urbanas + ['EH_FERIADO', 'DIA_PAGAMENTO', 'DIA_SEMANA']]
        y = df_modelo_agregado['RISCO_OBSERVADO']

        # Treinamento do modelo Ensemble
        lgbm = LGBMRegressor(random_state=42)
        cat = CatBoostRegressor(random_state=42, verbose=0)

        ensemble = VotingRegressor(estimators=[('lgbm', lgbm), ('cat', cat)])
        ensemble.fit(X, y)

        # Predição
        df_modelo_agregado['SCORE_RISCO'] = ensemble.predict(X).astype(int)
        df_modelo_agregado['SCORE_RISCO'] = df_modelo_agregado['SCORE_RISCO'].apply(lambda x: max(0, x)) # Garante que não há risco negativo

        # SHAP para explicabilidade
        def predict_ensemble(X_input):
            return ensemble.predict(X_input)

        explainer = shap.Explainer(predict_ensemble, X)
        shap_values = explainer(X)

        # Identifica o fator mais relevante para o risco
        avg_abs_shap = np.abs(shap_values.values).mean(axis=0)
        fator_mestre_idx = np.argmax(avg_abs_shap)
        fator_mestre_nome = X.columns[fator_mestre_idx]

        # Prepara df_ouro para o BigQuery
        df_ouro = df_modelo_agregado[['INDICE_H3'] + features_urbanas + ['RISCO_OBSERVADO', 'SCORE_RISCO']].copy()
        df_ouro['EH_FERIADO_HOJE'] = eh_feriado_hoje
        df_ouro['DIA_PAGAMENTO_HOJE'] = dia_pagamento_hoje
        df_ouro['ULTIMA_SYNC'] = datetime.utcnow()

        # --- Inclusão de dados de crime real para validação no Looker Studio ---
        # Pega os dados de crime real do dia anterior para comparação
        data_comparacao = hoje - pd.Timedelta(days=1) # Ou outro período relevante
        df_crime_real_dia = df_crime_real_agregado[df_crime_real_agregado['DATA_OCORRENCIA'] == data_comparacao].copy()

        df_ouro = df_ouro.merge(df_crime_real_dia[['INDICE_H3', 'NUMERO_OCORRENCIAS_REAIS', 'RISCO_OBSERVADO_REAL']], 
                                how='left', on='INDICE_H3')

        # Preenche NaN com 0 para hexágonos sem ocorrências no dia de comparação
        df_ouro['NUMERO_OCORRENCIAS_REAIS'] = df_ouro['NUMERO_OCORRENCIAS_REAIS'].fillna(0).astype(int)
        df_ouro['RISCO_OBSERVADO_REAL'] = df_ouro['RISCO_OBSERVADO_REAL'].fillna(0).astype(int)

        # --- Cálculo de Métricas de Eficiência para o Relatório ---
        # Filtra para hexágonos que tiveram alguma ocorrência real ou predita para evitar divisão por zero ou métricas inflacionadas
        df_comparacao = df_ouro[(df_ouro['SCORE_RISCO'] > 0) | (df_ouro['RISCO_OBSERVADO_REAL'] > 0)]

        correlacao_score_real = df_comparacao['SCORE_RISCO'].corr(df_comparacao['RISCO_OBSERVADO_REAL'])
        mae_score_real = mean_absolute_error(df_comparacao['RISCO_OBSERVADO_REAL'], df_comparacao['SCORE_RISCO'])

        # Exemplo de acurácia (se binarizarmos, ex: risco > 5 é alto)
        limiar_risco = 5
        acuracia_alto_risco = np.mean((df_comparacao['SCORE_RISCO'] > limiar_risco) == (df_comparacao['RISCO_OBSERVADO_REAL'] > limiar_risco))


        # --- Sincronização com BigQuery ---
        credenciais_bq = service_account.Credentials.from_service_account_info(json.loads(os.environ.get("BQ_SERVICE_ACCOUNT_JSON")))
        client = bigquery.Client(credentials=credenciais_bq, project=os.environ.get("BQ_PROJECT_ID"))

        dataset_id = os.environ.get('BQ_DATASET_ID')
        tabela_destino = f"{dataset_id}.tabela_ouro_risco"

        print("☁️ A sincronizar as predições com o BigQuery (Delta Sync)...")
        staging_table_id = f"staging_delta_ouro_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        job_config = bigquery.LoadJobConfig(
            schema=[
                bigquery.SchemaField("INDICE_H3", "STRING"),
                bigquery.SchemaField("DENSIDADE_LOGRADOUROS", "INTEGER"), # Alterado para INTEGER
                bigquery.SchemaField("PROPORCAO_RESIDENCIAL_H3", "FLOAT"),
                bigquery.SchemaField("TOTAL_EDIFICACOES_H3", "INTEGER"), # Alterado para INTEGER
                bigquery.SchemaField("EH_FERIADO_HOJE", "INTEGER"),
                bigquery.SchemaField("DIA_PAGAMENTO_HOJE", "INTEGER"),
                bigquery.SchemaField("RISCO_OBSERVADO", "INTEGER"),
                bigquery.SchemaField("SCORE_RISCO", "INTEGER"),
                bigquery.SchemaField("NUMERO_OCORRENCIAS_REAIS", "INTEGER"), # Nova coluna
                bigquery.SchemaField("RISCO_OBSERVADO_REAL", "INTEGER"), # Nova coluna
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
        WHEN MATCHED AND (
            ABS(T.SCORE_RISCO - S.SCORE_RISCO) >= 1 
            OR T.EH_FERIADO_HOJE != S.EH_FERIADO_HOJE 
            OR T.DIA_PAGAMENTO_HOJE != S.DIA_PAGAMENTO_HOJE
            OR T.NUMERO_OCORRENCIAS_REAIS != S.NUMERO_OCORRENCIAS_REAIS
            OR T.RISCO_OBSERVADO_REAL != S.RISCO_OBSERVADO_REAL
        ) THEN
          UPDATE SET 
            SCORE_RISCO = S.SCORE_RISCO, 
            ULTIMA_SYNC = S.ULTIMA_SYNC, 
            EH_FERIADO_HOJE = S.EH_FERIADO_HOJE, 
            DIA_PAGAMENTO_HOJE = S.DIA_PAGAMENTO_HOJE,
            NUMERO_OCORRENCIAS_REAIS = S.NUMERO_OCORRENCIAS_REAIS,
            RISCO_OBSERVADO_REAL = S.RISCO_OBSERVADO_REAL
        WHEN NOT MATCHED THEN
          INSERT (INDICE_H3, DENSIDADE_LOGRADOUROS, PROPORCAO_RESIDENCIAL_H3, TOTAL_EDIFICACOES_H3, EH_FERIADO_HOJE, DIA_PAGAMENTO_HOJE, RISCO_OBSERVADO, SCORE_RISCO, NUMERO_OCORRENCIAS_REAIS, RISCO_OBSERVADO_REAL, ULTIMA_SYNC) 
          VALUES (S.INDICE_H3, S.DENSIDADE_LOGRADOUROS, S.PROPORCAO_RESIDENCIAL_H3, S.TOTAL_EDIFICACOES_H3, S.EH_FERIADO_HOJE, S.DIA_PAGAMENTO_HOJE, S.RISCO_OBSERVADO, S.SCORE_RISCO, S.NUMERO_OCORRENCIAS_REAIS, S.RISCO_OBSERVADO_REAL, S.ULTIMA_SYNC)
        """
        client.query(sql_merge).result()

        client.delete_table(f"{dataset_id}.{staging_table_id}")

        if os.path.exists("camada_prata.parquet"):
            os.remove("camada_prata.parquet")
        if os.path.exists("crime_real_agregado.parquet"):
            os.remove("crime_real_agregado.parquet")

        dicionario_shap = {
            "PROPORCAO_RESIDENCIAL_H3": "Perfil Residencial da Zona", 
            "DIA_PAGAMENTO": "Ciclo Econômico (Dia de Pagamento)", # Alterado para DIA_PAGAMENTO
            "DENSIDADE_LOGRADOUROS": "Densidade da Malha Viária", 
            "EH_FERIADO": "Sazonalidade de Feriados", # Alterado para EH_FERIADO
            "TOTAL_EDIFICACOES_H3": "Volume Urbano de Edificações",
            "DIA_SEMANA": "Dia da Semana" # Adicionado
        }
        fator_explicado = dicionario_shap.get(fator_mestre_nome, fator_mestre_nome)

        insight_discord = f"A auditoria matemática identificou que o fator **{fator_explicado}** foi o principal impulsionador do recálculo de risco neste ciclo. "
        insight_discord += f"A correlação entre o risco predito e o real foi de **{correlacao_score_real:.2f}**. "
        insight_discord += f"O erro médio absoluto (MAE) foi de **{mae_score_real:.2f}**."

        robo.enviar_relatorio_operacional(
            "O Mapa de Risco Geocriminal foi recalibrado. O Delta Sync operou com sucesso no BigQuery, ignorando flutuações estatísticas menores.", 
            {"Zonas Monitorizadas": len(df_ouro), 
             "Metodologia": "XAI / Ensemble Learning",
             "Correlação Predito vs Real": f"{correlacao_score_real:.2f}",
             "MAE Predito vs Real": f"{mae_score_real:.2f}"}, 
            auditoria_ia=insight_discord
        )

    except Exception:
        robo.enviar_alerta_tecnico("Motor de IA e BigQuery (Ouro)", traceback.format_exc())
