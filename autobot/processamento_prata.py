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
        self.malha_path = "safedriver/datalake/base_geografica/safedriver_geo_base_sp_h3_9.parquet"
        self._inicializar_dependencias()

    def _inicializar_dependencias(self):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            self.df_malha = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            
            self.dens_distrito = self.df_malha.group_by(["NM_MUN", "NM_DIST"]).agg(
                pl.col("DENSIDADE_DEMOGRAFICA").mean().alias("DENS_DIST")
            )
            self.dens_cidade = self.df_malha.group_by(["NM_MUN"]).agg(
                pl.col("DENSIDADE_DEMOGRAFICA").mean().alias("DENS_CID")
            )
        except Exception as e:
            logger.error(f"PRATA: Falha ao carregar malha geográfica: {e}")
            self.df_malha = None

    def _canonizar_texto(self, coluna):
        return (
            coluna.str.replace(r"^(RUA|R|AVENIDA|AV|ALAMEDA|AL|PRACA|PRC|ESTRADA|EST|VIELA|VL)\.?\s+", "")
            .str.replace_all(r"[ÁÀÂÃÄ]", "A").str.replace_all(r"[ÉÈÊË]", "E")
            .str.replace_all(r"[ÍÌÎÏ]", "I").str.replace_all(r"[ÓÒÔÕÖ]", "O")
            .str.replace_all(r"[ÚÙÛÜ]", "U").str.replace_all(r"[Ç]", "C")
            .str.replace_all(r"[^A-Z0-9 ]", "").str.replace_all(r"\s+", " ").str.strip_chars()
        )

    def _motor_recuperacao_cruzada(self, df_bronze):
        df_ssp = df_bronze.with_columns([
            pl.col("LATITUDE").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("LONGITUDE").cast(pl.Float64, strict=False).fill_null(0.0),
            self._canonizar_texto(pl.col("LOGRADOURO")).alias("LOG_CANONICO")
        ])

        df_gps = df_ssp.filter(pl.col("LATITUDE") != 0).with_columns(
            pl.struct(["LATITUDE", "LONGITUDE"])
            .map_elements(lambda x: h3.latlng_to_cell(x["LATITUDE"], x["LONGITUDE"], 9), return_dtype=pl.String)
            .alias("H3_INDEX")
        )

        df_cegos = df_ssp.filter(pl.col("LATITUDE") == 0)
        malha_lookup = self.df_malha.with_columns(
            self._canonizar_texto(pl.col("LOGRADOURO")).alias("LOG_CANONICO")
        ).unique(subset=["NM_MUN", "LOG_CANONICO"]).select(["NM_MUN", "LOG_CANONICO", "H3_INDEX"])

        df_resgatados = df_cegos.join(
            malha_lookup, 
            left_on=["MUNICIPIO", "LOG_CANONICO"], 
            right_on=["NM_MUN", "LOG_CANONICO"], 
            how="inner"
        )

        df_unificado = pl.concat([df_gps, df_resgatados], how="diagonal")

        df_cross = df_unificado.join(
            self.df_malha.unique(subset=["H3_INDEX"]).select([
                "H3_INDEX", "LOGRADOURO", "NM_BAIRRO", "NM_MUN", "NM_DIST", "DENSIDADE_DEMOGRAFICA", "POPULACAO_H3"
            ]),
            on="H3_INDEX", how="left", suffix="_IBGE"
        ).join(self.dens_distrito, on=["NM_MUN", "NM_DIST"], how="left"
        ).join(self.dens_cidade, on=["NM_MUN"], how="left")

        novos_bairros = df_cross.filter(
            (pl.col("NM_BAIRRO") == "BAIRRO_PENDENTE") & (pl.col("BAIRRO").is_not_null())
        ).group_by("H3_INDEX").agg(pl.col("BAIRRO").first().alias("BAIRRO_CURADO"))

        if not novos_bairros.is_empty():
            self._persistir_enriquecimento_malha(novos_bairros)

        df_final = df_cross.with_columns([
            pl.when(pl.col("NM_BAIRRO") == "BAIRRO_PENDENTE").then(pl.col("BAIRRO")).otherwise(pl.col("NM_BAIRRO")).alias("BAIRRO_FINAL"),
            pl.when(pl.col("DENSIDADE_DEMOGRAFICA") > 0).then(pl.col("DENSIDADE_DEMOGRAFICA"))
            .when(pl.col("DENS_DIST") > 0).then(pl.col("DENS_DIST"))
            .when(pl.col("DENS_CID") > 0).then(pl.col("DENS_CID"))
            .otherwise(0.0).alias("DENSIDADE_FINAL")
        ])

        return df_final, novos_bairros.height, df_resgatados.height

    def _persistir_enriquecimento_malha(self, novos_dados):
        self.df_malha = self.df_malha.join(novos_dados, on="H3_INDEX", how="left")
        self.df_malha = self.df_malha.with_columns(
            pl.when(pl.col("BAIRRO_CURADO").is_not_null())
            .then(pl.col("BAIRRO_CURADO"))
            .otherwise(pl.col("NM_BAIRRO"))
            .alias("NM_BAIRRO")
        ).drop("BAIRRO_CURADO")
        
        buffer = io.BytesIO()
        self.df_malha.write_parquet(buffer)
        self.s3.put_object(Bucket=self.bucket, Key=self.malha_path, Body=buffer.getvalue())

    def _normalizar_bronze(self, data):
        dfs = []
        for i in range(1, 6):
            try:
                t = pl.read_excel(io.BytesIO(data), sheet_id=i, engine="calamine").with_columns(pl.all().cast(pl.String))
                t.columns = [c.upper().replace("Ç", "C").replace("Ã", "A").strip() for c in t.columns]
                dfs.append(t)
            except: continue
        
        if not dfs: return None

        df = pl.concat(dfs, how="diagonal")
        mapa = {"MUNICIPIO": ["NOME_MUNICIPIO", "CIDADE"], "LOGRADOURO": ["NOME_LOGRADOURO", "LOGRADOURO"], "BAIRRO": ["BAIRRO", "NOME_BAIRRO"]}
        for alvo, origens in mapa.items():
            col = next((o for o in origens if o in df.columns), None)
            if col: df = df.rename({col: alvo})
            
        return df.with_columns(pl.col(pl.String).str.strip_chars().str.to_uppercase())

    def processar_ano_com_delta(self, ano, estado, force=False):
        path_bronze = f"safedriver/datalake/bronze/ssp_raw_{ano}.xlsx"
        path_prata = f"safedriver/datalake/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_bronze)
            tamanho = meta['ContentLength']
        except: return False

        if not force and estado.get(str(ano)) == tamanho: return False

        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            df_bronze = self._normalizar_bronze(resp['Body'].read())
            
            if df_bronze is None: return None

            total_raw = df_bronze.height
            df_curado, cura_malha, resgate_h3 = self._motor_recuperacao_cruzada(df_bronze)

            df_final = df_curado.group_by(["H3_INDEX"]).agg([
                pl.col("RUBRICA").filter(pl.col("RUBRICA").str.contains("ROUBO|FURTO")).count().alias("TOTAL_CRIMES"),
                pl.col("BAIRRO_FINAL").first().alias("NM_BAIRRO"),
                pl.col("NM_MUN").first().alias("NM_MUN"),
                pl.col("DENSIDADE_FINAL").first().alias("DENSIDADE"),
                pl.col("POPULACAO_H3").first().alias("POPULACAO_H3")
            ]).with_columns(pl.lit(ano).cast(pl.Int32).alias("ANO_REFERENCIA"))

            buffer = io.BytesIO()
            df_final.write_parquet(buffer)
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            estado[str(ano)] = tamanho
            return {"ano": ano, "total_raw": total_raw, "bo_recuperados": resgate_h3, "malha_curada": cura_malha, "h3_unicos": df_final.height}
        except: return None

    def executar_todos_os_anos(self, force=False):
        estado_atual = self._carregar_tracker()
        relatorio = []
        metricas_globais = {"processados": 0, "bo_recuperados": 0, "malha_resgatada": 0}

        for ano in range(2022, datetime.now().year + 1):
            res = self.processar_ano_com_delta(ano, estado_atual, force)
            if res:
                self._salvar_tracker(estado_atual)
                metricas_globais["processados"] += res["total_raw"]
                metricas_globais["bo_recuperados"] += res["bo_recuperados"]
                metricas_globais["malha_resgatada"] += res["malha_curada"]
                relatorio.append(res)
        
        return {"metricas": metricas_globais, "detalhes": relatorio}

    def _carregar_tracker(self):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.tracker_path)
            return json.loads(resp['Body'].read())
        except ClientError: return {}

    def _salvar_tracker(self, estado):
        self.s3.put_object(Bucket=self.bucket, Key=self.tracker_path, Body=json.dumps(estado))
