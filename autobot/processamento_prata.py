import polars as pl
import boto3
import json
import os
import io
import h3
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
       
        try:
            self.s3 = boto3.client('s3',
                                    endpoint_url=os.getenv("R2_ENDPOINT_URL"),
                                    aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID"),
                                    aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY"))
            self.bucket = os.getenv("R2_BUCKET_NAME")
        except Exception as e:
            logger.error(f"Erro ao conectar ao R2: {e}")
            raise

        self.colunas_canonicas = [
            'NUM_BO', 'DATA_OCORRENCIA', 'LATITUDE', 'LONGITUDE', 
            'LOGRADOURO', 'NATUREZA_APURADA', 'DESCRICAO_LOCAL'
        ]

    def _converter_para_h3(self, lat, lon):
        try:
            return h3.latlng_to_cell(float(lat), float(lon), 8)
        except:
            return None

    def executar_prata(self, ano_ref):
        logger.info(f" INICIANDO REFINARIA PRATA - ANO: {ano_ref}")

       
        try:
         
            resp_dic = self.s3.get_object(Bucket=self.bucket, Key="datalake/bronze/mapa_capas_ssp.json")
            mapa_ano = json.loads(resp_dic['Body'].read().decode('utf-8')).get(str(ano_ref), {})
            de_para = self._gerar_mapeamento_canonico(mapa_ano)

        
            resp_geo = self.s3.get_object(Bucket=self.bucket, Key="datalake/base_geografica/safedriver_geo_base_sp_h3_8.parquet")
      
            lf_geo = pl.scan_parquet(io.BytesIO(resp_geo['Body'].read()))

         
            resp_raw = self.s3.get_object(Bucket=self.bucket, Key=f"datalake/bronze/ssp_raw_{ano_ref}.parquet")
            lf_ssp = pl.scan_parquet(io.BytesIO(resp_raw['Body'].read()))
        except Exception as e:
            logger.error(f"Falha no carregamento de arquivos: {e}")
            return

       
        pipeline = (
            lf_ssp.rename(de_para)
            .select([pl.col(c) for c in self.colunas_canonicas if c in lf_ssp.columns])
            .with_columns([
                pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False),
                pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False),
                pl.col("NATUREZA_APURADA").str.to_uppercase(),
                pl.col("LOGRADOURO").str.to_uppercase().str.strip_chars()
            ])
       
            .filter(
                (pl.col("LATITUDE").is_between(-25.31, -19.77)) & 
                (pl.col("LONGITUDE").is_between(-53.11, -44.16))
            )
        )

     
        df_ssp = pipeline.collect()
        
        df_ssp = df_ssp.with_columns(
            pl.struct(["LATITUDE", "LONGITUDE"])
            .map_elements(lambda x: self._converter_para_h3(x["LATITUDE"], x["LONGITUDE"]), return_dtype=pl.Utf8)
            .alias("H3_INDEX")
        )

   
        df_geo = lf_geo.collect()
        base_busca = df_geo.select(["LOGRADOURO", "H3_INDEX"]).drop_nulls().unique(subset=["LOGRADOURO"])

        com_gps = df_ssp.filter(pl.col("H3_INDEX").is_not_null())
        sem_gps = df_ssp.filter(pl.col("H3_INDEX").is_null()).drop("H3_INDEX")
        
        sem_gps_recuperado = sem_gps.join(base_busca, on="LOGRADOURO", how="left")
        
        df_final = pl.concat([com_gps, sem_gps_recuperado], how="diagonal")

       
        df_final = df_final.filter(pl.col("H3_INDEX").is_not_null())
        
       
        df_final = df_final.join(
            df_geo.select(["H3_INDEX", "GEO_CD_SETOR", "GEO_NM_BAIRRO", "GEO_NM_MUN"]).unique(subset=["H3_INDEX"]),
            on="H3_INDEX",
            how="left"
        )

      
        df_final = df_final.with_columns(
            pl.when(pl.col("NATUREZA_APURADA").str.contains("VEICULO|CARGA|ROUBO DE BENS"))
            .then(pl.lit("MOTORISTA"))
            .when(pl.col("NATUREZA_APURADA").str.contains("MOTOCICLETA|MOTO"))
            .then(pl.lit("MOTOCICLISTA"))
            .when(pl.col("NATUREZA_APURADA").str.contains("CELULAR|TRANSEUNTE|PEDESTRE"))
            .then(pl.lit("PEDESTRE"))
            .otherwise(pl.lit("GERAL"))
            .alias("PERFIL_PERSONA")
        )

     
        buffer = io.BytesIO()
        df_final.write_parquet(buffer)
        
        self.s3.put_object(
            Bucket=self.bucket, 
            Key=f"datalake/prata/ssp_consolidada_{ano_ref}.parquet", 
            Body=buffer.getvalue()
        )
        
        logger.info(f"✅ Camada Prata Finalizada. Retenção: {(df_final.height / lf_ssp.collect().height)*100:.2f}%")

    def _gerar_mapeamento_canonico(self, mapa_ano):
        de_para = {}
        for col, desc in mapa_ano.items():
            d = str(desc).upper()
            if "LATITUDE" in d: de_para[col] = "LATITUDE"
            elif "LONGITUDE" in d: de_para[col] = "LONGITUDE"
            elif "BOLETIM" in d: de_para[col] = "NUM_BO"
            elif "DATA" in d: de_para[col] = "DATA_OCORRENCIA"
            elif "NATUREZA" in d: de_para[col] = "NATUREZA_APURADA"
            elif "LOGRADOURO" in d: de_para[col] = "LOGRADOURO"
        return de_para
