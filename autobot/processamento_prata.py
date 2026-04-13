import os
import pandas as pd
import boto3
import traceback
import io
import hashlib
import h3
import unicodedata
from .comunicador import RoboComunicador

class ProcessamentoPrata:
    def __init__(self, robo: RoboComunicador):
        self.robo = robo
        self.s3 = boto3.client('s3',
                                endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                                aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                                aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))
        self.bucket = os.environ.get("R2_BUCKET_NAME")
        self.sal_lgpd = os.environ.get("LGPD_SALT", "seguranca_safedriver_2026")

    def normalizar(self, texto):
        if not isinstance(texto, str):
            return ""
        texto = unicodedata.normalize("NFKD", texto)
        texto = "".join(c for c in texto if not unicodedata.combining(c))
        return texto.upper().strip()

    def carregar_base_geografica(self):
        caminho_geo = "safedriver/datalake/base_geografica/safedriver_geo_base_sp_h3_8.parquet"
        try:
            resposta = self.s3.get_object(Bucket=self.bucket, Key=caminho_geo)
            df_geo = pd.read_parquet(io.BytesIO(resposta['Body'].read()))
            
            df_geo['CHAVE_LOCALIDADE'] = (
                df_geo['NM_MUN'].apply(self.normalizar) + "|" +
                df_geo['NM_BAIRRO'].apply(self.normalizar) + "|" +
                df_geo['LOGRADOURO_NORMALIZADO'].apply(self.normalizar)
            )
            return df_geo
        except Exception:
            raise Exception("Falha ao carregar Master Geo Table")

    def executar(self, ano):
        caminho_bruta = f"safedriver/datalake/bruta/ssp_{ano}_bronze.parquet"
        caminho_prata = f"safedriver/datalake/prata/ssp_{ano}_prata.parquet"

        try:
            self.robo.enviar_relatorio_operacional(f"Inicio Prata Profunda {ano}")

            resposta_bruta = self.s3.get_object(Bucket=self.bucket, Key=caminho_bruta)
            df = pd.read_parquet(io.BytesIO(resposta_bruta['Body'].read()))

            df['MUNICIPIO_NORM'] = df['MUNICIPIO'].apply(self.normalizar)
            df['BAIRRO_NORM'] = df['BAIRRO'].apply(self.normalizar)
            df['LOGRADOURO_NORM'] = df['LOGRADOURO'].apply(self.normalizar)
            df['CHAVE_LOCALIDADE'] = df['MUNICIPIO_NORM'] + "|" + df['BAIRRO_NORM'] + "|" + df['LOGRADOURO_NORM']

            df['LATITUDE'] = pd.to_numeric(df['LATITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
            df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'].astype(str).str.replace(',', '.'), errors='coerce')

            mask_gps = (df['LATITUDE'].notnull()) & (df['LATITUDE'] != 0) & (df['LONGITUDE'] != 0)
            df.loc[mask_gps, 'H3_INDEX'] = [
                h3.latlng_to_cell(lat, lon, 8) 
                for lat, lon in zip(df.loc[mask_gps, 'LATITUDE'], df.loc[mask_gps, 'LONGITUDE'])
            ]

            df_geo = self.carregar_base_geografica()

            mask_recuperacao = df['H3_INDEX'].isnull()
            if mask_recuperacao.any():
                df_recuperado = pd.merge(
                    df.loc[mask_recuperacao].drop(columns=['H3_INDEX']),
                    df_geo[['CHAVE_LOCALIDADE', 'H3_INDEX']],
                    on='CHAVE_LOCALIDADE',
                    how='inner'
                )
                df = pd.concat([df.loc[~mask_recuperacao], df_recuperado], ignore_index=True)

            colunas_geo = [
                'H3_INDEX', 'DENSIDADE_LOGRADOUROS', 'TOTAL_RESIDENCIAS_H3', 
                'TOTAL_EDIFICACOES_H3', 'TOTAL_NAO_RESIDENCIAIS_H3', 
                'PROPORCAO_RESIDENCIAL_H3', 'DIVERSIDADE_LOGRADOUROS_H3'
            ]
            
            df = pd.merge(df, df_geo[colunas_geo], on='H3_INDEX', how='inner')

            df['ID_ANONIMO'] = (df['NUM_BO'].astype(str) + df['ANO_BO'].astype(str) + self.sal_lgpd).apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
            )

            colunas_remover = [
                'NUM_BO', 'LATITUDE', 'LONGITUDE', 'CHAVE_LOCALIDADE', 
                'MUNICIPIO_NORM', 'BAIRRO_NORM', 'LOGRADOURO_NORM'
            ]
            df_final = df.drop(columns=[c for c in colunas_remover if c in df.columns])

            buffer = io.BytesIO()
            df_final.to_parquet(buffer, index=False)
            self.s3.put_object(Bucket=self.bucket, Key=caminho_prata, Body=buffer.getvalue())

            self.robo.enviar_relatorio_operacional(f"Sucesso Prata {ano}", {"Registros": len(df_final)})
            return True

        except Exception:
            self.robo.enviar_alerta_tecnico(f"Erro Prata {ano}", traceback.format_exc())
            return False

def iniciar_processamento_prata(robo, ano):
    processador = ProcessamentoPrata(robo)
    return processador.executar(ano)
