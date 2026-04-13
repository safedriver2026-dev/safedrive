import os
import pandas as pd
import boto3
import traceback
import io
import hashlib
import h3
import unicodedata

# Importação Absoluta para evitar ModuleNotFoundError no CI/CD
from autobot.comunicador import RoboComunicador

class ProcessamentoPrata:
    def __init__(self, robo: RoboComunicador):
        self.robo = robo
        self.s3 = boto3.client('s3',
                                endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                                aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                                aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))
        self.bucket = os.environ.get("R2_BUCKET_NAME")
        # Sal para garantir que o Hash da LGPD seja irreversível para terceiros
        self.sal_lgpd = os.environ.get("LGPD_SALT", "safedriver_secret_2026")

    def normalizar(self, texto):
        if not isinstance(texto, str):
            return ""
        texto = unicodedata.normalize("NFKD", texto)
        texto = "".join(c for c in texto if not unicodedata.combining(c))
        return texto.upper().strip()

    def carregar_base_geografica(self):
        """Carrega a Master Geo Table do R2 para recuperação e enriquecimento"""
        caminho_geo = "datalake/base_geografica/safedriver_geo_base_sp_h3_8.parquet"
        try:
            resposta = self.s3.get_object(Bucket=self.bucket, Key=caminho_geo)
            df_geo = pd.read_parquet(io.BytesIO(resposta['Body'].read()))
            
            # Chave de cruzamento para registros sem GPS (Município|Bairro|Logradouro)
            df_geo['CHAVE_LOCALIDADE'] = (
                df_geo['NM_MUN'].apply(self.normalizar) + "|" +
                df_geo['NM_BAIRRO'].apply(self.normalizar) + "|" +
                df_geo['LOGRADOURO_NORMALIZADO'].apply(self.normalizar)
            )
            return df_geo
        except Exception:
            raise Exception("Falha Crítica: Base Geográfica não encontrada no R2.")

    def executar(self, ano):
        origem = f"datalake/bruta/ssp_{ano}_bronze.parquet"
        destino = f"datalake/prata/ssp_{ano}_prata.parquet"

        try:
            self.robo.enviar_relatorio_operacional(f"Iniciando Processamento Prata {ano}")

            # Leitura da Bronze (Raw)
            resp_s3 = self.s3.get_object(Bucket=self.bucket, Key=origem)
            df = pd.read_parquet(io.BytesIO(resp_s3['Body'].read()))

            # 1. Normalização de Strings para Join e Auditoria
            df['MUN_NORM'] = df['MUNICIPIO'].apply(self.normalizar)
            df['BAI_NORM'] = df['BAIRRO'].apply(self.normalizar)
            df['LOG_NORM'] = df['LOGRADOURO'].apply(self.normalizar)
            df['CHAVE_LOCALIDADE'] = df['MUN_NORM'] + "|" + df['BAI_NORM'] + "|" + df['LOG_NORM']

            # 2. Tratamento de Coordenadas GPS
            df['LATITUDE'] = pd.to_numeric(df['LATITUDE'].astype(str).str.replace(',', '.'), errors='coerce')
            df['LONGITUDE'] = pd.to_numeric(df['LONGITUDE'].astype(str).str.replace(',', '.'), errors='coerce')

            # 3. Indexação Geo-Espacial (H3) via GPS
            mask_gps = (df['LATITUDE'].notnull()) & (df['LATITUDE'] != 0) & (df['LONGITUDE'] != 0)
            df.loc[mask_gps, 'H3_INDEX'] = [
                h3.latlng_to_cell(lat, lon, 8) 
                for lat, lon in zip(df.loc[mask_gps, 'LATITUDE'], df.loc[mask_gps, 'LONGITUDE'])
            ]

            # 4. Recuperação via Logradouro (Para casos sem GPS válido)
            df_geo = self.carregar_base_geografica()
            mask_sem_h3 = df['H3_INDEX'].isnull()
            
            if mask_sem_h3.any():
                df_recuperado = pd.merge(
                    df.loc[mask_sem_h3].drop(columns=['H3_INDEX']),
                    df_geo[['CHAVE_LOCALIDADE', 'H3_INDEX']],
                    on='CHAVE_LOCALIDADE',
                    how='inner'
                )
                df = pd.concat([df.loc[~mask_sem_h3], df_recuperado], ignore_index=True)

            # 5. Enriquecimento Profundo (Uso de solo e densidade para a IA)
            colunas_geo_profundas = [
                'H3_INDEX', 
                'DENSIDADE_LOGRADOUROS', 
                'TOTAL_RESIDENCIAS_H3',      # Adicionado conforme solicitado
                'TOTAL_EDIFICACOES_H3', 
                'TOTAL_NAO_RESIDENCIAIS_H3', 
                'PROPORCAO_RESIDENCIAL_H3', 
                'DIVERSIDADE_LOGRADOUROS_H3'
            ]
            df = pd.merge(df, df_geo[colunas_geo_profundas], on='H3_INDEX', how='inner')

            # 6. Atribuição de Pesos de Crime (Arredondamento para 1 Casa Decimal)
            # Ex: Roubo = 5.5, Furto = 2.0
            pesos_referencia = {
                'HOMICIDIO': 10.0, 'LATROCINIO': 10.0, 'ROUBO': 5.5, 
                'ESTUPRO': 8.5, 'FURTO': 2.0, 'TRAFICO': 4.5
            }
            
            def atribuir_peso(natureza):
                natureza_norm = self.normalizar(natureza)
                for crime, peso in pesos_referencia.items():
                    if crime in natureza_norm: return peso
                return 1.0

            df['PESO_CRIME'] = df['NATUREZA'].apply(atribuir_peso).astype(float).round(1)

            # 7. Anonimização LGPD (ID Irreversível)
            df['ID_ANONIMO'] = (df['NUM_BO'].astype(str) + df['ANO_BO'].astype(str) + self.sal_lgpd).apply(
                lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
            )

            # 8. Limpeza de Colunas Sensíveis e Temporárias
            colunas_finais = [
                'ID_ANONIMO', 'ANO_BO', 'DATA_OCORRENCIA', 'HORA_OCORRENCIA', 
                'NATUREZA', 'PESO_CRIME', 'H3_INDEX', 'DENSIDADE_LOGRADOUROS',
                'TOTAL_RESIDENCIAS_H3', 'TOTAL_EDIFICACOES_H3', 'TOTAL_NAO_RESIDENCIAIS_H3',
                'PROPORCAO_RESIDENCIAL_H3', 'DIVERSIDADE_LOGRADOUROS_H3'
            ]
            
            df_final = df[colunas_finais].copy()

            # 9. Salvamento Incremental (Delta Sync)
            buffer = io.BytesIO()
            df_final.to_parquet(buffer, index=False)
            self.s3.put_object(Bucket=self.bucket, Key=destino, Body=buffer.getvalue())

            self.robo.enviar_relatorio_operacional(f"Sucesso Prata {ano}", {"Registros": len(df_final)})
            return True

        except Exception:
            self.robo.enviar_alerta_tecnico(f"Falha na Camada Prata {ano}", traceback.format_exc())
            return False
