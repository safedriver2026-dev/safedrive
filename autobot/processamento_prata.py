import polars as pl
import boto3
import io
import os
import h3
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
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
        self.h3_resolution = 8
        
        # Dicionario Executivo de Personas baseado na Natureza Apurada
        self.mapa_personas = {
            "MOTORISTA": [
                "FURTO DE VEÍCULO", "ROUBO DE VEÍCULO", 
                "HOMICÍDIO CULPOSO POR ACIDENTE DE TRÂNSITO",
                "LESÃO CORPORAL CULPOSA POR ACIDENTE DE TRÂNSITO",
                "HOMICÍDIO DOLOSO POR ACIDENTE DE TRÂNSITO"
            ],
            "PEDESTRE": [
                "FURTO - OUTROS", "ROUBO - OUTROS", "LATROCÍNIO",
                "ESTUPRO", "ESTUPRO DE VULNERÁVEL", "HOMICÍDIO DOLOSO",
                "LESÃO CORPORAL DOLOSA", "TENTATIVA DE HOMICÍDIO"
            ]
        }

    def executar_prata(self, ano):
        path_bronze = f"datalake/bronze/ssp_raw_{ano}.xlsx"
        path_geo = "datalake/base_geografica/safedriver_geo_base_sp_h3_8.parquet"
        path_prata = f"datalake/prata/ssp_consolidada_{ano}.parquet"

        try:
            logger.info(f"PRATA: Iniciando processamento do ano {ano}...")
            resp_ssp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            conteudo_excel = resp_ssp['Body'].read()

            # 1. Leitura do Dicionario da Capa (Metadados)
            self._processar_dicionario_capa(conteudo_excel)

            # 2. Extracao de Dados Reais
            lf_ssp = self._extrair_dados_consolidados(conteudo_excel)
            
            # 3. Join com Base Geografica
            resp_geo = self.s3.get_object(Bucket=self.bucket, Key=path_geo)
            lf_geo = pl.read_parquet(io.BytesIO(resp_geo['Body'].read()))

            # 4. Geocodificacao H3
            lf_ssp = self._geocodificar_h3(lf_ssp)

            # 5. Agregacao por Persona (Usando Natureza Apurada)
            lf_crimes = self._agregar_por_persona(lf_ssp)

            # 6. Consolidacao Final
            lf_final = lf_crimes.join(lf_geo, on="H3_INDEX", how="inner")
            lf_final = self._aplicar_indicadores(lf_final, ano)

            # Salvar no R2
            buffer = io.BytesIO()
            lf_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            logger.info(f"PRATA: Dados de {ano} consolidados com sucesso.")
            return True

        except Exception as e:
            logger.error(f"Erro no processamento Prata {ano}: {e}")
            raise e

    def _processar_dicionario_capa(self, conteudo):
        try:
            # Tenta ler a aba que geralmente contem as definicoes de campos
            df_capa = pl.read_excel(io.BytesIO(conteudo), sheet_id=0)
            logger.info(f"PRATA: Dicionario de campos detectado com {len(df_capa)} definicoes.")
        except:
            logger.warning("PRATA: Nao foi possivel ler a aba de metadados (Capa).")

    def _extrair_dados_consolidados(self, conteudo):
        lista_dfs = []
        for i in range(1, 15): # Percorre meses e abas extras
            try:
                df = pl.read_excel(io.BytesIO(conteudo), sheet_id=i)
                df.columns = [c.upper().strip() for c in df.columns]
                
                if "LATITUDE" in df.columns and "NATUREZA_APURADA" in df.columns:
                    # Filtro de qualidade de coordenadas
                    df = df.filter(
                        (pl.col("LATITUDE").is_not_null()) & 
                        (pl.col("LATITUDE").cast(pl.Utf8) != "0")
                    )
                    df = df.with_columns(pl.all().cast(pl.Utf8))
                    lista_dfs.append(df)
            except:
                continue
        
        return pl.concat(lista_dfs, how="diagonal")

    def _geocodificar_h3(self, lf):
        df_pd = lf.to_pandas()
        df_pd['H3_INDEX'] = df_pd.apply(
            lambda r: h3.latlng_to_cell(float(r['LATITUDE']), float(r['LONGITUDE']), self.h3_resolution),
            axis=1
        )
        return pl.from_pandas(df_pd)

    def _agregar_por_persona(self, lf):
        # Identificacao de Motociclistas (Busca por palavras-chave na Rubrica e Conduta)
        lf = lf.with_columns([
            pl.when(
                (pl.col("RUBRICA").str.contains("(?i)Moto|Motocicleta")) | 
                (pl.col("DESCR_CONDUTA").str.contains("(?i)Moto|Motocicleta"))
            ).then(pl.lit("MOTOCICLISTA"))
            .when(pl.col("NATUREZA_APURADA").is_in(self.mapa_personas["MOTORISTA"]))
            .then(pl.lit("MOTORISTA"))
            .when(pl.col("NATUREZA_APURADA").is_in(self.mapa_personas["PEDESTRE"]))
            .then(pl.lit("PEDESTRE"))
            .otherwise(pl.lit("OUTROS"))
            .alias("CATEGORIA_PERSONA")
        ])

        return lf.group_by("H3_INDEX").agg([
            pl.col("H3_INDEX").filter(pl.col("CATEGORIA_PERSONA") == "MOTORISTA").count().alias("TOTAL_CRIMES_MOTORISTA"),
            pl.col("H3_INDEX").filter(pl.col("CATEGORIA_PERSONA") == "PEDESTRE").count().alias("TOTAL_CRIMES_PEDESTRE"),
            pl.col("H3_INDEX").filter(pl.col("CATEGORIA_PERSONA") == "MOTOCICLISTA").count().alias("TOTAL_CRIMES_MOTOCICLISTA")
        ])

    def _aplicar_indicadores(self, lf, ano):
        return lf.with_columns([
            pl.lit(ano).alias("ANO_REFERENCIA"),
            pl.col("TOTAL_NAO_RESIDENCIAIS_H3").alias("TOTAL_NAO_RES_H3"),
            pl.col("PROPORCAO_RESIDENCIAL_H3").alias("INDICE_RESIDENCIAL"),
            pl.col("DENSIDADE_LOGRADOUROS").alias("DENSIDADE_ENDERECOS")
        ]).fill_null(0)

if __name__ == "__main__":
    ProcessamentoPrata().executar_prata(2024)
