import polars as pl
import h3
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import io
import os
import json
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )
        
        self.tracker_path = "safedriver/datalake/prata/tracker_estado_bronze.json"

    def _carregar_tracker(self):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.tracker_path)
            return json.loads(resp['Body'].read())
        except ClientError: return {}

    def _salvar_tracker(self, estado):
        self.s3.put_object(Bucket=self.bucket, Key=self.tracker_path, Body=json.dumps(estado))

    def executar_todos_os_anos(self):
        logger.info("PRATA: Iniciando consolidação com Inteligência de Mapeamento SSP...")
        ano_atual = datetime.now().year
        estado_atual = self._carregar_tracker()

        for ano in range(2022, ano_atual + 1):
            if self.processar_ano_com_delta(ano, estado_atual):
                self._salvar_tracker(estado_atual)
                logger.info(f"PRATA: Ciclo {ano} finalizado com sucesso.")

    def processar_ano_com_delta(self, ano, estado):
        path_bronze = f"safedriver/datalake/bronze/ssp_raw_{ano}.xlsx"
        path_prata_atual = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
        path_prata_anterior = f"safedriver/datalake/prata/ssp_consolidada_{ano - 1}.parquet"

        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_bronze)
            tamanho_atual = meta['ContentLength']
        except ClientError: return False

        if estado.get(str(ano)) == tamanho_atual:
            logger.info(f"PRATA: [{ano}] Sem alterações.")
            return False

        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            excel_data = resp['Body'].read()

            # 1. BUSCA DE ABA (Descarta abas de legenda/capa)
            df = None
            for sheet_idx in [1, 2]:
                try:
                    temp_df = pl.read_excel(io.BytesIO(excel_data), sheet_id=sheet_idx, engine="calamine")
                    temp_cols = [str(c).upper() for c in temp_df.columns]
                    if "RUBRICA" in temp_cols or "LATITUDE" in temp_cols:
                        df = temp_df
                        break
                except: continue

            if df is None: return False

            # 2. NORMALIZAÇÃO CANÔNICA (Unifica as variações de 2022-2026)
            # Removemos acentos e espaços para comparação
            def clean_col(c): return str(c).upper().replace("Ç", "C").replace("Ã", "A").strip()
            df = df.rename({c: clean_col(c) for c in df.columns})

            # Mapeamento de De/Para para unificar o Schema
            mapeamento = {
                "DESCR_TIPOLOCAL": "LOCAL_TIPO",
                "DESCR_SUBTIPOLOCAL": "LOCAL_TIPO",
                "CIDADE": "MUNICIPIO",
                "NOME_MUNICIPIO": "MUNICIPIO",
                "DESCR_PERIODO": "PERIODO",
                "DESC_PERIODO": "PERIODO"
            }
            
            # Aplica o de/para apenas nas colunas que existem no arquivo atual
            cols_atuais = df.columns
            rename_dict = {k: v for k, v in mapeamento.items() if k in cols_atuais}
            df = df.rename(rename_dict)

            # 3. LIMPEZA E H3 (latlng_to_cell para v4)
            df = df.with_columns([
                pl.col("LATITUDE").cast(pl.Float64, strict=False),
                pl.col("LONGITUDE").cast(pl.Float64, strict=False)
            ]).filter(pl.col("LATITUDE").is_not_null() & (pl.col("LATITUDE") != 0))

            if df.is_empty(): return False

            h3_list = [h3.latlng_to_cell(lat, lon, 8) for lat, lon in zip(df["LATITUDE"], df["LONGITUDE"])]
            df = df.with_columns(pl.Series("H3_INDEX", h3_list))

            # 4. AGREGAÇÃO (Agora com nomes unificados)
            df_atual = df.group_by(["H3_INDEX"]).agg([
                pl.col("RUBRICA").str.to_uppercase().filter(pl.col("RUBRICA").str.contains("ROUBO|FURTO|VEICULO|CARGA")).count().alias("TOTAL_CRIMES_MOTORISTA"),
                pl.col("RUBRICA").str.to_uppercase().filter(pl.col("RUBRICA").str.contains("CELULAR|TRANSEUNTE|PESSOA")).count().alias("TOTAL_CRIMES_PEDESTRE"),
                pl.col("RUBRICA").str.to_uppercase().filter(pl.col("RUBRICA").str.contains("MOTOCICLETA|MOTO")).count().alias("TOTAL_CRIMES_MOTOCICLISTA"),
                pl.col("LOCAL_TIPO").str.to_uppercase().filter(pl.col("LOCAL_TIPO").str.contains("RESIDENCIA|CONDOMINIO")).count().alias("INDICE_RESIDENCIAL"),
                pl.col("LOCAL_TIPO").str.to_uppercase().filter(~pl.col("LOCAL_TIPO").str.contains("RESIDENCIA")).count().alias("TOTAL_NAO_RES_H3"),
                pl.col("LOGRADOURO").n_unique().alias("DENSIDADE_ENDERECOS")
            ]).with_columns(pl.lit(ano).alias("ANO_REFERENCIA"))

            # 5. INJEÇÃO DE DELTAS E PERSISTÊNCIA
            df_final = self._injetar_deltas(df_atual, path_prata_anterior)

            buffer = io.BytesIO()
            df_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata_atual, Body=buffer.getvalue())
            
            estado[str(ano)] = tamanho_atual
            return True

        except Exception as e:
            logger.error(f"PRATA: Falha no ciclo de {ano}: {e}")
            return False

    def _injetar_deltas(self, df_atual, path_anterior):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_anterior)
            df_passado = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            df_join = df_atual.join(
                df_passado.select(["H3_INDEX", "TOTAL_CRIMES_MOTORISTA", "TOTAL_CRIMES_PEDESTRE", "TOTAL_CRIMES_MOTOCICLISTA"]),
                on="H3_INDEX", how="left", suffix="_PASSADO"
            ).fill_null(0)
            return df_join.with_columns([
                (pl.col("TOTAL_CRIMES_MOTORISTA") - pl.col("TOTAL_CRIMES_MOTORISTA_PASSADO")).alias("DELTA_MOTORISTA"),
                (pl.col("TOTAL_CRIMES_PEDESTRE") - pl.col("TOTAL_CRIMES_PEDESTRE_PASSADO")).alias("DELTA_PEDESTRE"),
                (pl.col("TOTAL_CRIMES_MOTOCICLISTA") - pl.col("TOTAL_CRIMES_MOTOCICLISTA_PASSADO")).alias("DELTA_MOTOCICLISTA")
            ]).drop([c for c in df_join.columns if "_PASSADO" in c])
        except:
            return df_atual.with_columns([pl.lit(0).alias("DELTA_MOTORISTA"), pl.lit(0).alias("DELTA_PEDESTRE"), pl.lit(0).alias("DELTA_MOTOCICLISTA")])

if __name__ == "__main__":
    ProcessamentoPrata().executar_todos_os_anos()
