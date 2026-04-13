import os
import pandas as pd
import boto3
import traceback
from datetime import datetime
from comunicador import RoboComunicador
import hashlib
import h3

def executar_processamento(robo: RoboComunicador):
    s3 = boto3.client('s3',
                      endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                      aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))

    bucket = os.environ.get("R2_BUCKET_NAME")
    ano = datetime.now().year
    path_raw = f"safedriver/datalake/raw/ssp_{ano}_incremental.parquet"
    path_silver = f"safedriver/datalake/silver/prata_{ano}.parquet"

    lgpd_salt = os.environ.get("LGPD_SALT")
    if not lgpd_salt:
        lgpd_salt = "default_salt_para_testes_lgpd"

    try:
        print(f"DEBUG PRATA: Tentando baixar arquivo RAW do R2: {path_raw}")
        s3.download_file(bucket, path_raw, "raw_data_temp.parquet")
        df_raw = pd.read_parquet("raw_data_temp.parquet")
        os.remove("raw_data_temp.parquet")
        print(f"DEBUG PRATA: Arquivo RAW baixado e lido. {len(df_raw)} registros.")

        print("DEBUG PRATA: Aplicando anonimização LGPD...")
        df_raw['ID_ANONIMO'] = df_raw['NUM_BO'].astype(str) + df_raw['ANO_BO'].astype(str) + df_raw['NOME_DELEGACIA']
        df_raw['ID_ANONIMO'] = df_raw['ID_ANONIMO'].apply(lambda x: hashlib.sha256(f"{x}{lgpd_salt}".encode()).hexdigest()[:12])

        print("DEBUG PRATA: Aplicando geoprocessamento H3...")
        if 'LATITUDE' not in df_raw.columns or 'LONGITUDE' not in df_raw.columns:
            print("Aviso: Colunas LATITUDE/LONGITUDE não encontradas. Criando INDICE_H3 fictício para teste.")
            df_raw['INDICE_H3'] = '8928308280fffff'
        else:
             df_raw['INDICE_H3'] = df_raw.apply(lambda row: h3.geo_to_h3(row['LATITUDE'], row['LONGITUDE'], 9), axis=1)

        print("DEBUG PRATA: Enriquecendo dados com features urbanas e temporais...")
        df_raw['DENSIDADE_LOGRADOUROS'] = 0.5
        df_raw['PROPORCAO_RESIDENCIAL_H3'] = 0.7
        df_raw['TOTAL_EDIFICACOES_H3'] = 100
        df_raw['EH_FERIADO'] = False
        df_raw['DIA_PAGAMENTO'] = False
        df_raw['PESO_CRIME'] = 1

        print(f"DEBUG PRATA: Salvando arquivo PRATA no R2: {path_silver}")
        df_raw.to_parquet("silver_data_temp.parquet", index=False)
        s3.upload_file("silver_data_temp.parquet", bucket, path_silver)
        os.remove("silver_data_temp.parquet")
        print("DEBUG PRATA: Camada Prata salva com sucesso no R2.")

        robo.enviar_relatorio_operacional("✅ Camada Prata processada e salva no R2.", 
                                   {"Registros Processados": len(df_raw),
                                    "Camada": "Prata"})
        return True
    except Exception:
        robo.enviar_alerta_tecnico("Processamento Camada Prata", traceback.format_exc())
        print(f"❌ Erro no processamento da Camada Prata: {traceback.format_exc()}")
        return False
