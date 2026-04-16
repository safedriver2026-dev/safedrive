import polars as pl
import pandas as pd
import h3
import boto3
from botocore.config import Config
import io
import os
import json
import logging
from datetime import datetime
from zoneinfo import ZoneInfo

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}, max_pool_connections=50))
        
        self.base_path = self._localizar_datalake_real()
        self.tracker_path = f"{self.base_path}/prata/tracker_estado_bronze.json"
        self.malha_path = f"{self.base_path}/base_geografica/safedriver_geo_base_sp_h3_9.parquet"
        
        self.campos_recuperados_grade = 0
        self._inicializar_dependencias()

    def _localizar_datalake_real(self):
        try:
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, MaxKeys=100):
                for obj in page.get('Contents', []):
                    if "datalake/bronze/trusted/" in obj['Key']:
                        return obj['Key'].split("datalake/")[0] + "datalake"
            return "datalake"
        except: return "datalake"

    def _limpar_texto_extremo(self, coluna):
        return (
            pl.col(coluna)
            .cast(pl.String)
            .str.to_uppercase()
            .str.replace_all(r"[ÁÀÂÃÄ]", "A")
            .str.replace_all(r"[ÉÈÊË]", "E")
            .str.replace_all(r"[ÍÌÎÏ]", "I")
            .str.replace_all(r"[ÓÒÔÕÖ]", "O")
            .str.replace_all(r"[ÚÙÛÜ]", "U")
            .str.replace_all(r"[Ç]", "C")
            .str.replace_all(r"[Ñ]", "N")
            .str.strip_chars()
            .fill_null("INDEFINIDO")
        )

    def _inicializar_dependencias(self):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.malha_path)
            self.df_malha = pl.read_parquet(io.BytesIO(resp['Body'].read()))
            self.df_malha_lazy = self.df_malha.lazy().with_columns([
                self._limpar_texto_extremo("NM_MUN").alias("NM_MUN"),
                self._limpar_texto_extremo("NM_BAIRRO").alias("NM_BAIRRO")
            ])
            logger.info("PRATA: Malha geográfica mestre carregada.")
        except Exception as e:
            logger.error(f"PRATA: Falha ao carregar malha H3: {e}")
            self.df_malha_lazy = None

    def _curar_malha_referencia(self, df_bo_limpo):
        if self.df_malha is None: return
        df_conhecimento = (
            df_bo_limpo.filter((pl.col("H3_INDEX").is_not_null()) & (pl.col("NM_MUN_ORIGINAL") != "INDEFINIDO"))
            .group_by("H3_INDEX").agg([
                pl.col("NM_MUN_ORIGINAL").first().alias("MUN_NOVO"),
                pl.col("NM_BAIRRO_ORIGINAL").first().alias("BAIRRO_NOVO")
            ])
        )
        malha_antes = self.df_malha.filter((pl.col("NM_MUN") != "INDEFINIDO") & (pl.col("NM_MUN").is_not_null())).height
        self.df_malha = self.df_malha.join(df_conhecimento, on="H3_INDEX", how="left").with_columns([
            pl.coalesce([pl.col("MUN_NOVO"), pl.col("NM_MUN")]).fill_null("INDEFINIDO").alias("NM_MUN"),
            pl.coalesce([pl.col("BAIRRO_NOVO"), pl.col("NM_BAIRRO")]).fill_null("INDEFINIDO").alias("NM_BAIRRO")
        ]).drop(["MUN_NOVO", "BAIRRO_NOVO"])
        
        malha_depois = self.df_malha.filter((pl.col("NM_MUN") != "INDEFINIDO") & (pl.col("NM_MUN").is_not_null())).height
        self.campos_recuperados_grade += (malha_depois - malha_antes)
        self.df_malha_lazy = self.df_malha.lazy()

    def _gerar_features_espaciais_ia(self, df_pd):
        usar_grid_disk = hasattr(h3, 'grid_disk')
        df_unique = df_pd.groupby('H3_INDEX', as_index=False)['TOTAL_CRIMES'].sum()
        crimes_dit = dict(zip(df_unique['H3_INDEX'], df_unique['TOTAL_CRIMES']))
        contagio_dit = {}
        for h3_index in df_unique['H3_INDEX']:
            try:
                v1 = set(h3.grid_disk(h3_index, 1) if usar_grid_disk else h3.k_ring(h3_index, 1))
                v1.discard(h3_index)
                c1 = sum(crimes_dit.get(v, 0) for v in v1)
                v_total = set(h3.grid_disk(h3_index, 2) if usar_grid_disk else h3.k_ring(h3_index, 2))
                v2 = v_total - v1
                contagio_dit[h3_index] = (c1 * 1.0) + (sum(crimes_dit.get(v, 0) for v in v2) * 0.5)
            except: contagio_dit[h3_index] = 0.0
        df_pd['CONTAGIO_PONDERADO'] = df_pd['H3_INDEX'].map(contagio_dit)
        df_pd['PRESSAO_RISCO_LOCAL'] = df_pd['CONTAGIO_PONDERADO'] / (df_pd['DENSIDADE'] + 0.001)
        return df_pd

    def processar_ano_com_delta(self, ano, estado, force=False):
        path_trusted = f"{self.base_path}/bronze/trusted/ssp_trusted_{ano}.parquet"
        path_prata = f"{self.base_path}/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            tamanho_atual = meta['ContentLength']
            if not force and estado.get(str(ano)) == tamanho_atual: return None

            resp = self.s3.get_object(Bucket=self.bucket, Key=path_trusted)
            lf = pl.read_parquet(io.BytesIO(resp['Body'].read())).lazy()

            lf = lf.filter((pl.col("H3_INDEX").is_not_null()))
            total_in = lf.select(pl.len()).collect().item()

            # Recuperando os mapeamentos de Tipo de Local que a IA ama
            mapeamento = {
                "CIDADE": "NM_MUN_ORIGINAL", "MUNICIPIO": "NM_MUN_ORIGINAL", "BAIRRO": "NM_BAIRRO_ORIGINAL",
                "HORA_OCORRENCIA_BO": "HORA", "DATA_OCORRENCIA_BO": "DATA_BRUTA", "DATA_OCORRENCIA": "DATA_BRUTA",
                "DESCR_TIPOLOCAL": "TIPO_LOCAL", "DESCR_SUBTIPOLOCAL": "TIPO_LOCAL"
            }
            lf = lf.rename({old: new for old, new in mapeamento.items() if old in lf.collect_schema().names()})

            if "TIPO_LOCAL" not in lf.collect_schema().names():
                lf = lf.with_columns(pl.lit("INDEFINIDO").alias("TIPO_LOCAL"))

            if "DATA_BRUTA" in lf.collect_schema().names():
                lf = lf.with_columns([
                    pl.col("DATA_BRUTA").cast(pl.String).str.to_date(format="%d/%m/%Y", strict=False).alias("DATA"),
                    pl.col("HORA").cast(pl.String).str.split(":").list.first().cast(pl.Int32, strict=False).alias("HORA_INT")
                ])
                lf = lf.with_columns([
                    pl.col("DATA").dt.month().fill_null(1).alias("MES_OCORRENCIA"),
                    pl.col("DATA").dt.weekday().fill_null(1).alias("DIA_SEMANA_OCORRENCIA"),
                    pl.when(pl.col("HORA_INT") < 6).then(pl.lit("MADRUGADA"))
                      .when(pl.col("HORA_INT") < 12).then(pl.lit("MANHA"))
                      .when(pl.col("HORA_INT") < 18).then(pl.lit("TARDE"))
                      .otherwise(pl.lit("NOITE")).alias("PERIODO_DIA"),
                    pl.when(pl.col("RUBRICA").str.contains("(?i)VEICULO|AUTO|MOTO")).then(pl.lit("MOTORISTA"))
                      .otherwise(pl.lit("PEDESTRE")).alias("PERFIL_ALVO")
                ])
            else:
                lf = lf.with_columns([
                    pl.lit(1).alias("MES_OCORRENCIA"), pl.lit(1).alias("DIA_SEMANA_OCORRENCIA"),
                    pl.lit("NOITE").alias("PERIODO_DIA"), pl.lit("PEDESTRE").alias("PERFIL_ALVO")
                ])
            
            # Agregando com toda a inteligência junta
            lf_agg = lf.group_by([
                "H3_INDEX", "NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL", 
                "MES_OCORRENCIA", "DIA_SEMANA_OCORRENCIA", "PERIODO_DIA", "PERFIL_ALVO", "TIPO_LOCAL"
            ]).agg([
                pl.when(pl.col("RUBRICA").str.contains("(?i)ROUBO")).then(3).otherwise(1).sum().alias("TOTAL_CRIMES")
            ])

            lf_agg = lf_agg.with_columns([self._limpar_texto_extremo(c) for c in ["NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL", "TIPO_LOCAL"]])
            self._curar_malha_referencia(lf_agg.select(["H3_INDEX", "NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL"]).collect())

            # Trazendo a DENSIDADE e TAXA_VACANCIA da malha, e cravando o nome final exato
            lf_enriquecido = lf_agg.join(self.df_malha_lazy, on="H3_INDEX", how="left").with_columns([
                pl.coalesce([pl.col("NM_MUN"), pl.col("NM_MUN_ORIGINAL")]).alias("NM_MUN"),
                pl.coalesce([pl.col("NM_BAIRRO"), pl.col("NM_BAIRRO_ORIGINAL")]).alias("NM_BAIRRO"),
                pl.col("DENSIDADE_AJUSTADA").cast(pl.Float64).fill_null(0.0).alias("DENSIDADE"),
                pl.col("TAXA_VACANCIA").cast(pl.Float64, strict=False).fill_null(0.0).alias("TAXA_VACANCIA")
            ]).drop(["NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL"])

            df_final_pd = self._gerar_features_espaciais_ia(lf_enriquecido.collect().to_pandas())
            
            # Gerando Exposicao e o Ranking
            df_final = pl.from_pandas(df_final_pd).with_columns([
                (pl.col("TOTAL_CRIMES") / (pl.col("DENSIDADE") + 1)).alias("INDICE_EXPOSICAO"),
                (pl.col("TOTAL_CRIMES").rank().over("PERIODO_DIA") / pl.col("TOTAL_CRIMES").count().over("PERIODO_DIA")).alias("RANKING_RISCO_LOCAL"),
                pl.lit(ano).alias("ANO_REFERENCIA")
            ])

            buffer = io.BytesIO()
            df_final.write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            estado[str(ano)] = tamanho_atual 
            return {"linhas_in": total_in, "linhas_out": df_final.height}
        except Exception as e:
            logger.error(f"PRATA: Erro no ano {ano}: {e}")
            return None

    def executar_todos_os_anos(self, force=False):
        stats = {"linhas_in": 0, "linhas_out": 0}
        estado = self._carregar_tracker()
        for ano in range(2022, datetime.now().year + 1):
            res = self.processar_ano_com_delta(ano, estado, force)
            if res:
                stats["linhas_in"] += res["linhas_in"]
                stats["linhas_out"] += res["linhas_out"]
                self._salvar_tracker(estado)
        
        if self.campos_recuperados_grade > 0:
            buffer_malha = io.BytesIO()
            self.df_malha.write_parquet(buffer_malha, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=self.malha_path, Body=buffer_malha.getvalue())

        stats["recuperado_grade"] = self.campos_recuperados_grade
        stats["taxa_recuperacao"] = round((stats["linhas_out"] / stats["linhas_in"] * 100), 2) if stats["linhas_in"] > 0 else 100
        stats["status_camadas"] = {"prata": "✅ Concluido"}
        return stats

    def _carregar_tracker(self):
        try:
            return json.loads(self.s3.get_object(Bucket=self.bucket, Key=self.tracker_path)['Body'].read())
        except: return {}

    def _salvar_tracker(self, estado):
        self.s3.put_object(Bucket=self.bucket, Key=self.tracker_path, Body=json.dumps(estado))

if __name__ == "__main__":
    prata = ProcessamentoPrata()
    prata.executar_todos_os_anos(force=True)
