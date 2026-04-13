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
        logger.info("PRATA: Iniciando consolidação com Dicionário de Sinônimos Resiliente...")
        ano_atual = datetime.now().year
        estado_atual = self._carregar_tracker()

        for ano in range(2022, ano_atual + 1):
            if self.processar_ano_com_delta(ano, estado_atual):
                self._salvar_tracker(estado_atual)
                logger.info(f"PRATA: Ciclo {ano} consolidado com sucesso.")

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
            dfs_abas = []

            for sheet_idx in range(1, 6):
                try:
                    # Estratégia String-First: Evita conflitos de tipos entre anos/abas
                    temp_df = pl.read_excel(io.BytesIO(excel_data), sheet_id=sheet_idx, engine="calamine")
                    temp_df = temp_df.with_columns(pl.all().cast(pl.String))
                    
                    # Normalização de nomes: Remove acentos, espaços e força Upper
                    temp_df = temp_df.rename({
                        c: str(c).upper().replace("Ç", "C").replace("Ã", "A").strip() 
                        for c in temp_df.columns
                    })
                    
                    if any(k in temp_df.columns for k in ["LATITUDE", "RUBRICA"]):
                        dfs_abas.append(temp_df)
                except: continue

            if not dfs_abas: return False

            # Concatena abas do mesmo ano (diagonal permite colunas extras como CMD/BTL)
            df = pl.concat(dfs_abas, how="diagonal")

            # --- DICIONÁRIO DE SINÔNIMOS CANÔNICO ---
            # Resolve as variações que você mapeou de 2022 a 2026
            sinonimos = {
                "MUNICIPIO": ["NOME_MUNICIPIO", "CIDADE"],
                "LOCAL_TIPO": ["DESCR_SUBTIPOLOCAL", "DESCR_TIPOLOCAL"],
                "PERIODO": ["DESC_PERIODO", "DESCR_PERIODO"],
                "SEC_CIRC": ["NOME_SECCIONAL_CIRCUNSCRICAO", "NOME_SECCIONAL_CIRCUNSCRIÇÃO"],
                "DEP_CIRC": ["NOME_DEPARTAMENTO_CIRCUNSCRICAO", "NOME_DEPARTAMENTO_CIRCUNSCRIÇÃO"],
                "MUN_CIRC": ["NOME_MUNICIPIO_CIRCUNSCRICAO", "NOME_MUNICIPIO_CIRCUNSCRIÇÃO"]
            }

            for alvo, origens in sinonimos.items():
                col_encontrada = next((o for o in origens if o in df.columns), None)
                if col_encontrada:
                    df = df.rename({col_encontrada: alvo})
                    # Limpeza de duplicidade: se houver outros sinônimos, apaga-os
                    outros = [o for o in origens if o in df.columns and o != alvo]
                    if outros: df = df.drop(outros)

            # Sanitização dos dados (Strings limpas)
            df = df.with_columns(pl.col(pl.String).str.strip_chars().str.to_uppercase())

            # Conversão numérica para cálculo geográfico
            df = df.with_columns([
                pl.col("LATITUDE").cast(pl.Float64, strict=False),
                pl.col("LONGITUDE").cast(pl.Float64, strict=False)
            ]).filter(pl.col("LATITUDE").is_not_null() & (pl.col("LATITUDE") != 0))

            if df.is_empty(): return False

            # Geração do H3
            h3_list = [h3.latlng_to_cell(lat, lon, 8) for lat, lon in zip(df["LATITUDE"], df["LONGITUDE"])]
            df = df.with_columns(pl.Series("H3_INDEX", h3_list))

            # Agregação com Casting Int32 (Essencial para o Treinador IA)
            df_atual = df.group_by(["H3_INDEX"]).agg([
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("ROUBO|FURTO|VEICULO|CARGA")).count().cast(pl.Int32).alias("TOTAL_CRIMES_MOTORISTA"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("CELULAR|TRANSEUNTE|PESSOA")).count().cast(pl.Int32).alias("TOTAL_CRIMES_PEDESTRE"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("MOTOCICLETA|MOTO")).count().cast(pl.Int32).alias("TOTAL_CRIMES_MOTOCICLISTA"),
                pl.col("LOCAL_TIPO").filter(pl.col("LOCAL_TIPO").str.contains("RESIDENCIA|CONDOMINIO")).count().cast(pl.Int32).alias("INDICE_RESIDENCIAL"),
                pl.col("LOCAL_TIPO").filter(~pl.col("LOCAL_TIPO").str.contains("RESIDENCIA")).count().cast(pl.Int32).alias("TOTAL_NAO_RES_H3"),
                pl.col("LOGRADOURO").n_unique().cast(pl.Int32).alias("DENSIDADE_ENDERECOS")
            ]).with_columns(pl.lit(ano).cast(pl.Int32).alias("ANO_REFERENCIA"))

            # Cálculo de Deltas
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
