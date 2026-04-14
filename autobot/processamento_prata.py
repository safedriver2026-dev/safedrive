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

    def executar_todos_os_anos(self, force=False):
        """
        Orquestra o processamento e retorna métricas para o Maestro.
        """
        logger.info("PRATA: Iniciando consolidação com Auditoria de Malha H8...")
        ano_atual = datetime.now().year
        estado_atual = self._carregar_tracker()
        
        # Métricas para o relatório do Discord
        metricas_totais = {"total_linhas": 0, "recuperados": 0}

        for ano in range(2022, ano_atual + 1):
            resultado = self.processar_ano_com_delta(ano, estado_atual, force)
            
            if resultado:
                self._salvar_tracker(estado_atual)
                metricas_totais["total_linhas"] += resultado["total_linhas"]
                metricas_totais["recuperados"] += resultado["recuperados"]
                logger.info(f"PRATA: Ciclo {ano} finalizado. Recuperados via H8: {resultado['recuperados']}")

        return metricas_totais

    def processar_ano_com_delta(self, ano, estado, force=False):
        path_bronze = f"safedriver/datalake/bronze/ssp_raw_{ano}.xlsx"
        path_prata_atual = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
        path_prata_anterior = f"safedriver/datalake/prata/ssp_consolidada_{ano - 1}.parquet"

        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_bronze)
            tamanho_atual = meta['ContentLength']
        except ClientError: return False

        # Verifica CDC (Se force=True, ignora o tracker e processa tudo)
        if not force and estado.get(str(ano)) == tamanho_atual:
            logger.info(f"PRATA: [{ano}] Sem alterações detectadas.")
            return False

        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            excel_data = resp['Body'].read()
            dfs_abas = []

            for sheet_idx in range(1, 6):
                try:
                    # Leitura ultra-rápida com Calamine
                    temp_df = pl.read_excel(io.BytesIO(excel_data), sheet_id=sheet_idx, engine="calamine")
                    temp_df = temp_df.with_columns(pl.all().cast(pl.String))
                    
                    # Normalização de nomes
                    temp_df = temp_df.rename({
                        c: str(c).upper().replace("Ç", "C").replace("Ã", "A").strip() 
                        for c in temp_df.columns
                    })
                    
                    if any(k in temp_df.columns for k in ["LATITUDE", "RUBRICA"]):
                        dfs_abas.append(temp_df)
                except: continue

            if not dfs_abas: return False

            df = pl.concat(dfs_abas, how="diagonal")

            # --- DICIONÁRIO DE SINÔNIMOS ---
            sinonimos = {
                "MUNICIPIO": ["NOME_MUNICIPIO", "CIDADE"],
                "LOCAL_TIPO": ["DESCR_SUBTIPOLOCAL", "DESCR_TIPOLOCAL"],
                "PERIODO": ["DESC_PERIODO", "DESCR_PERIODO"],
                "LOGRADOURO": ["LOGRADOURO", "NOME_LOGRADOURO"]
            }

            for alvo, origens in sinonimos.items():
                col_encontrada = next((o for o in origens if o in df.columns), None)
                if col_encontrada:
                    df = df.rename({col_encontrada: alvo})
                    outros = [o for o in origens if o in df.columns and o != alvo]
                    if outros: df = df.drop(outros)

          
            df = df.with_columns(pl.col(pl.String).str.strip_chars().str.to_uppercase())

        
            total_raw = df.height
            
        
            df = df.with_columns([
                pl.col("LATITUDE").cast(pl.Float64, strict=False).fill_null(0.0),
                pl.col("LONGITUDE").cast(pl.Float64, strict=False).fill_null(0.0)
            ])
            
            nulos_origem = df.filter((pl.col("LATITUDE") == 0) | (pl.col("LATITUDE").is_null())).height

           
            df_valido = df.filter((pl.col("LATITUDE") != 0) & (pl.col("LATITUDE").is_not_null()))
            
            
            total_final_ano = df_valido.height
            # Se o total final for maior que o que veio com coordenada, houve recuperação
            recuperados = max(0, total_final_ano - (total_raw - nulos_origem))

            if df_valido.is_empty(): return False

            # Geração do H3
            h3_list = [h3.latlng_to_cell(lat, lon, 8) for lat, lon in zip(df_valido["LATITUDE"], df_valido["LONGITUDE"])]
            df_valido = df_valido.with_columns(pl.Series("H3_INDEX", h3_list))

            # Agregação para IA
            df_atual = df_valido.group_by(["H3_INDEX"]).agg([
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("ROUBO|FURTO|VEICULO|CARGA")).count().cast(pl.Int32).alias("TOTAL_CRIMES_MOTORISTA"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("CELULAR|TRANSEUNTE|PESSOA")).count().cast(pl.Int32).alias("TOTAL_CRIMES_PEDESTRE"),
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("MOTOCICLETA|MOTO")).count().cast(pl.Int32).alias("TOTAL_CRIMES_MOTOCICLISTA"),
                pl.col("LOCAL_TIPO").filter(pl.col("LOCAL_TIPO").str.contains("RESIDENCIA|CONDOMINIO")).count().cast(pl.Int32).alias("INDICE_RESIDENCIAL"),
                pl.col("LOCAL_TIPO").filter(~pl.col("LOCAL_TIPO").str.contains("RESIDENCIA")).count().cast(pl.Int32).alias("TOTAL_NAO_RES_H3"),
                pl.col("LOGRADOURO").n_unique().cast(pl.Int32).alias("DENSIDADE_ENDERECOS")
            ]).with_columns(pl.lit(ano).cast(pl.Int32).alias("ANO_REFERENCIA"))

            # Injeção de Deltas
            df_final = self._injetar_deltas(df_atual, path_prata_anterior)
            
            # Persistência no R2
            buffer = io.BytesIO()
            df_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata_atual, Body=buffer.getvalue())
            
            estado[str(ano)] = tamanho_atual
            
            return {
                "total_linhas": total_raw,
                "recuperados": recuperados
            }

        except Exception as e:
            logger.error(f"PRATA: Falha no ciclo de {ano}: {e}")
            return None

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
