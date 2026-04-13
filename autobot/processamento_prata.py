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
        logger.info("PRATA: Iniciando consolidação com Scanner de Cabeçalhos...")
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

            # 1. SCANNER DE ABAS (Procura a aba que contém os cabeçalhos reais)
            df = None
            # Varremos até 5 abas (alguns anos podem ter capas e legendas extras)
            for sheet_idx in range(1, 6):
                try:
                    # Leitura rápida para checar colunas
                    temp_df = pl.read_excel(io.BytesIO(excel_data), sheet_id=sheet_idx, engine="calamine")
                    
                    # Padronização agressiva temporária para checagem
                    cols_check = [str(c).upper().replace("Ç", "C").replace("Ã", "A").strip() for c in temp_df.columns]
                    
                    # Se tiver as 3 colunas vitais, é a aba de dados
                    if all(k in cols_check for k in ["LATITUDE", "LONGITUDE", "RUBRICA"]):
                        df = temp_df
                        logger.info(f"PRATA: [{ano}] Dados reais localizados na Aba {sheet_idx}.")
                        break
                    else:
                        logger.warning(f"PRATA: [{ano}] Aba {sheet_idx} identificada como 'Capa' ou 'Legenda'. Ignorando...")
                except Exception:
                    continue

            if df is None:
                logger.error(f"PRATA: [{ano}] Nenhuma aba de dados válida encontrada no Excel.")
                return False

            # 2. NORMALIZAÇÃO DE NOMES E LIMPEZA DE STRINGS
            df = df.rename({c: str(c).upper().replace("Ç", "C").replace("Ã", "A").strip() for c in df.columns})
            
            # Sanitização total de campos Utf8 (Remoção de espaços e Upper)
            df = df.with_columns([
                pl.col(pl.Utf8).str.strip_chars().str.to_uppercase()
            ])

            # 3. RESOLUÇÃO DE DUPLICIDADE (TIPOLOCAL vs SUBTIPOLOCAL)
            if "DESCR_SUBTIPOLOCAL" in df.columns and "DESCR_TIPOLOCAL" in df.columns:
                df = df.drop("DESCR_TIPOLOCAL") # Mantém a mais específica
            
            mapeamento = {
                "DESCR_TIPOLOCAL": "LOCAL_TIPO", "DESCR_SUBTIPOLOCAL": "LOCAL_TIPO",
                "CIDADE": "MUNICIPIO", "NOME_MUNICIPIO": "MUNICIPIO",
                "DESCR_PERIODO": "PERIODO", "DESC_PERIODO": "PERIODO"
            }
            df = df.rename({k: v for k, v in mapeamento.items() if k in df.columns})

            # 4. CONVERSÃO GEOGRÁFICA
            df = df.with_columns([
                pl.col("LATITUDE").cast(pl.Float64, strict=False),
                pl.col("LONGITUDE").cast(pl.Float64, strict=False)
            ]).filter(pl.col("LATITUDE").is_not_null() & (pl.col("LATITUDE") != 0))

            if df.is_empty(): return False

            h3_list = [h3.latlng_to_cell(lat, lon, 8) for lat, lon in zip(df["LATITUDE"], df["LONGITUDE"])]
            df = df.with_columns(pl.Series("H3_INDEX", h3_list))

            # 5. AGREGAÇÃO COM CASTING RIGOROSO (INT32)
            # Forçamos Int32 em tudo para evitar erros de UInt32 (Unsigned) no Treinador
            df_atual = df.group_by(["H3_INDEX"]).agg([
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("ROUBO|FURTO|VEICULO|CARGA")).count().cast(pl.Int32).alias("TOTAL_CRIMES_MOTORISTA"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("CELULAR|TRANSEUNTE|PESSOA")).count().cast(pl.Int32).alias("TOTAL_CRIMES_PEDESTRE"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("MOTOCICLETA|MOTO")).count().cast(pl.Int32).alias("TOTAL_CRIMES_MOTOCICLISTA"),
                pl.col("LOCAL_TIPO").filter(pl.col("LOCAL_TIPO").str.contains("RESIDENCIA|CONDOMINIO")).count().cast(pl.Int32).alias("INDICE_RESIDENCIAL"),
                pl.col("LOCAL_TIPO").filter(~pl.col("LOCAL_TIPO").str.contains("RESIDENCIA")).count().cast(pl.Int32).alias("TOTAL_NAO_RES_H3"),
                pl.col("LOGRADOURO").n_unique().cast(pl.Int32).alias("DENSIDADE_ENDERECOS")
            ]).with_columns(pl.lit(ano).cast(pl.Int32).alias("ANO_REFERENCIA"))

            # 6. DELTAS E PERSISTÊNCIA
            df_final = self._injetar_deltas(df_atual, path_prata_anterior)

            buffer = io.BytesIO()
            df_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata_atual, Body=buffer.getvalue())
            
            estado[str(ano)] = tamanho_atual
            return True

        except Exception as e:
            logger.error(f"PRATA: Falha crítica no ciclo de {ano}: {e}")
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
                (pl.col("TOTAL_CRIMES_MOTORISTA") - pl.col("TOTAL_CRIMES_MOTORISTA_PASSADO")).cast(pl.Int32).alias("DELTA_MOTORISTA"),
                (pl.col("TOTAL_CRIMES_PEDESTRE") - pl.col("TOTAL_CRIMES_PEDESTRE_PASSADO")).cast(pl.Int32).alias("DELTA_PEDESTRE"),
                (pl.col("TOTAL_CRIMES_MOTOCICLISTA") - pl.col("TOTAL_CRIMES_MOTOCICLISTA_PASSADO")).cast(pl.Int32).alias("DELTA_MOTOCICLISTA")
            ]).drop([c for c in df_join.columns if "_PASSADO" in c])
        except:
            return df_atual.with_columns([
                pl.lit(0).cast(pl.Int32).alias("DELTA_MOTORISTA"),
                pl.lit(0).cast(pl.Int32).alias("DELTA_PEDESTRE"),
                pl.lit(0).cast(pl.Int32).alias("DELTA_MOTOCICLISTA")
            ])

if __name__ == "__main__":
    ProcessamentoPrata().executar_todos_os_anos()
