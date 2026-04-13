import pandas as pd
import h3
import holidays
import boto3
import os
import traceback
import hashlib
from datetime import datetime
from comunicador import RoboComunicador

FERIADOS = holidays.Brazil(subdiv='SP')
SALT = os.environ.get("LGPD_SALT", "default_salt_seguranca")

class SchemaBrain:
    @staticmethod
    def normalizar(df):
        mapa = {
            "LAT": ["LATITUDE", "LATITUDE_BO", "COORD_Y"], 
            "LON": ["LONGITUDE", "LONGITUDE_BO", "COORD_X"], 
            "DATA": ["DATA_OCORRENCIA_BO", "DT_OCORRENCIA", "DATA_FATO"],
            "RUBRICA": ["NATUREZA_APURADA", "RUBRICA", "CRIME"],
            "NUM_BO": ["NUM_BO", "NUMERO_BOLETIM"],
            "ANO_BO": ["ANO_BO", "ANO_BOLETIM"],
            "NOME_DELEGACIA": ["NOME_DELEGACIA", "DELEGACIA_NOME"]
        }
        for padrao, sinonimos in mapa.items():
            for col in sinonimos:
                if col in df.columns:
                    df = df.rename(columns={col: padrao})
                    break
        return df

def executar_silver(robo: RoboComunicador):
    s3 = boto3.client('s3',
                      endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                      aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                      aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))
    bucket = os.environ.get("R2_BUCKET_NAME")
    ano = datetime.now().year

    try:
        print("⚙️ A iniciar processamento da Camada Prata...")

        caminho_raw = f"safedriver/datalake/raw/ssp_{ano}_incremental.parquet"
        caminho_master = "safedriver/datalake/base_geografica/safedriver_geo_base_sp_h3_8.parquet"
        caminho_silver_r2 = f"safedriver/datalake/silver/prata_{ano}.parquet"
        caminho_crime_real_agregado_r2 = f"safedriver/datalake/validation/crime_real_agregado_{ano}.parquet"

        s3.download_file(bucket, caminho_raw, "raw.parquet")
        s3.download_file(bucket, caminho_master, "master.parquet")

        df_raw = pd.read_parquet("raw.parquet")
        df_master = pd.read_parquet("master.parquet")

        df_raw = SchemaBrain.normalizar(df_raw)

        df_raw['DATA'] = pd.to_datetime(df_raw['DATA'], errors='coerce')

        df_raw = df_raw.dropna(subset=['LAT', 'LON', 'DATA'])

        df_raw['INDICE_H3'] = df_raw.apply(lambda x: h3.geo_to_h3(float(x['LAT']), float(x['LON']), 8), axis=1)

        df_silver = df_raw.merge(df_master, on='INDICE_H3', how='left')

        df_silver['ID_ANONIMO'] = df_silver['NUM_BO'].apply(lambda x: hashlib.sha256(f"{x}{SALT}".encode()).hexdigest()[:12])

        df_silver['EH_FERIADO'] = df_silver['DATA'].apply(lambda x: 1 if x in FERIADOS else 0)
        df_silver['DIA_PAGAMENTO'] = df_silver['DATA'].apply(lambda d: 1 if (1 <= d.day <= 7) or (19 <= d.day <= 22) else 0)
        df_silver['DIA_SEMANA'] = df_silver['DATA'].dt.dayofweek

        df_silver['PESO_CRIME'] = df_silver['RUBRICA'].apply(
            lambda x: 10 if any(c in str(x).upper() for c in ["ROUBO", "LATROCINIO", "ESTUPRO", "HOMICIDIO"]) else 4
        ).astype(int)

        df_silver.to_parquet("camada_prata_temp.parquet", index=False)
        s3.upload_file("camada_prata_temp.parquet", bucket, caminho_silver_r2)

        # Geração e persistência da tabela de crime real agregado para validação
        df_crime_real_agregado = df_silver.groupby(['INDICE_H3', df_silver['DATA'].dt.normalize()]).agg(
            NUMERO_OCORRENCIAS_REAIS=('NUM_BO', 'count'),
            RISCO_OBSERVADO_REAL=('PESO_CRIME', 'sum')
        ).reset_index()
        df_crime_real_agregado = df_crime_real_agregado.rename(columns={'DATA': 'DATA_OCORRENCIA'})

        df_crime_real_agregado.to_parquet("crime_real_agregado_temp.parquet", index=False)
        s3.upload_file("crime_real_agregado_temp.parquet", bucket, caminho_crime_real_agregado_r2)

        if os.path.exists("raw.parquet"):
            os.remove("raw.parquet")
        if os.path.exists("master.parquet"):
            os.remove("master.parquet")
        if os.path.exists("camada_prata_temp.parquet"):
            os.remove("camada_prata_temp.parquet")
        if os.path.exists("crime_real_agregado_temp.parquet"):
            os.remove("crime_real_agregado_temp.parquet")

        robo.enviar_relatorio_operacional("Camada Prata processada com sucesso.", 
                                   {"Registros Fundidos": len(df_silver), 
                                    "Segurança": "LGPD Ativa",
                                    "Camada": "Silver (Prata)"})

    except Exception:
        robo.enviar_alerta_tecnico("Processamento Prata (Silver)", traceback.format_exc())
