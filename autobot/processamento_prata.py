import polars as pl
import boto3
from botocore.config import Config
import io
import os
import json
import logging
import h3
from datetime import datetime
from zoneinfo import ZoneInfo

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
        # Credenciais Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        # Configuração de Conexão S3
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
                self._limpar_texto_extremo("NM_BAIRRO").alias("NM_BAIRRO"),
                self._limpar_texto_extremo("LOGRADOURO").alias("LOGRADOURO_GRID")
            ])
            logger.info("PRATA: Malha geográfica normalizada carregada.")
        except Exception as e:
            logger.error(f"PRATA: Falha ao carregar malha H3: {e}")
            self.df_malha_lazy = None

    # --- UPGRADE 1: CURA DA MALHA ---
    def _curar_malha_referencia(self, df_bo_limpo):
        if self.df_malha is None: return
        df_conhecimento = (
            df_bo_limpo.filter((pl.col("H3_INDEX").is_not_null()) & (pl.col("NM_MUN_ORIGINAL") != "INDEFINIDO"))
            .group_by("H3_INDEX").agg([
                pl.col("NM_MUN_ORIGINAL").first().alias("MUN_NOVO"),
                pl.col("NM_BAIRRO_ORIGINAL").first().alias("BAIRRO_NOVO")
            ])
        )
        malha_antes = self.df_malha.filter(pl.col("NM_MUN") != "INDEFINIDO").height
        self.df_malha = self.df_malha.join(df_conhecimento, on="H3_INDEX", how="left").with_columns([
            pl.when(pl.col("NM_MUN") == "INDEFINIDO").then(pl.col("MUN_NOVO")).otherwise(pl.col("NM_MUN")).alias("NM_MUN"),
            pl.when(pl.col("NM_BAIRRO") == "INDEFINIDO").then(pl.col("BAIRRO_NOVO")).otherwise(pl.col("NM_BAIRRO")).alias("NM_BAIRRO")
        ]).drop(["MUN_NOVO", "BAIRRO_NOVO"])
        
        self.campos_recuperados_grade += (self.df_malha.filter(pl.col("NM_MUN") != "INDEFINIDO").height - malha_antes)
        # Atualiza a lazy para o join que ocorre em seguida
        self.df_malha_lazy = self.df_malha.lazy()

    # --- UPGRADE 2: CONTÁGIO ESPACIAL (IA) ---
    def _gerar_features_espaciais_ia(self, df_pd):
        crimes_dit = dict(zip(df_pd['H3_INDEX'], df_pd['TOTAL_CRIMES']))
        usar_grid_disk = hasattr(h3, 'grid_disk')
        contagio = []
        for h3_index in df_pd['H3_INDEX']:
            try:
                v1 = set(h3.grid_disk(h3_index, 1) if usar_grid_disk else h3.k_ring(h3_index, 1))
                v1.discard(h3_index)
                c1 = sum(crimes_dit.get(v, 0) for v in v1)
                
                v_total = set(h3.grid_disk(h3_index, 2) if usar_grid_disk else h3.k_ring(h3_index, 2))
                v2 = v_total - v1
                v2.discard(h3_index)
                c2 = sum(crimes_dit.get(v, 0) for v in v2)
                contagio.append((c1 * 1.0) + (c2 * 0.5))
            except: contagio.append(0.0)
        
        df_pd['CONTAGIO_PONDERADO'] = contagio
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

            # --- SUA LÓGICA ANTIGA DE SCHEMA ---
            cols = lf.collect_schema().names()
            if "DESCR_SUBTIPOLOCAL" in cols:
                lf = lf.rename({"DESCR_SUBTIPOLOCAL": "TIPO_LOCAL"})
                if "DESCR_TIPOLOCAL" in cols: lf = lf.drop("DESCR_TIPOLOCAL")
            elif "DESCR_TIPOLOCAL" in cols:
                lf = lf.rename({"DESCR_TIPOLOCAL": "TIPO_LOCAL"})

            mapeamento = {
                "CIDADE": "NM_MUN_ORIGINAL", "NOME_MUNICIPIO": "NM_MUN_ORIGINAL", "MUNICIPIO": "NM_MUN_ORIGINAL",
                "BAIRRO": "NM_BAIRRO_ORIGINAL", "LOGRADOURO": "LOGRADOURO_ORIGINAL",
                "HORA_OCORRENCIA_BO": "HORA", "DESC_PERIODO": "PERIODO_TEXTO", "DESCR_PERIODO": "PERIODO_TEXTO",
                "DESCR_CONDUTA": "CONDUTA",
                "DATA_OCORRENCIA_BO": "DATA", "DATA_OCORRENCIA": "DATA" # Mapeamento da Data
            }
            rename_dict = {old: new for old, new in mapeamento.items() if old in cols}
            if rename_dict: lf = lf.rename(rename_dict)

            # --- UPGRADE 3: PARSING DE DATAS SEGURO ---
            cols_atualizadas = lf.collect_schema().names()
            if "DATA" in cols_atualizadas:
                lf = lf.with_columns(pl.col("DATA").cast(pl.String).str.to_date(format="%d/%m/%Y", strict=False).alias("DATA_PARSED"))
                lf = lf.with_columns([
                    pl.col("DATA_PARSED").dt.month().fill_null(1).alias("MES_OCORRENCIA"),
                    pl.col("DATA_PARSED").dt.weekday().fill_null(1).alias("DIA_SEMANA_OCORRENCIA")
                ])
            else:
                lf = lf.with_columns([pl.lit(1).alias("MES_OCORRENCIA"), pl.lit(1).alias("DIA_SEMANA_OCORRENCIA")])

            # --- SUA LÓGICA ANTIGA DE HIGIENE E TEXTO ---
            total_in = lf.select(pl.len()).collect().item()
            campos_texto = ["NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL", "LOGRADOURO_ORIGINAL", "RUBRICA", "CONDUTA", "TIPO_LOCAL", "PERIODO_TEXTO"]
            lf = lf.with_columns([self._limpar_texto_extremo(c) for c in campos_texto if c in cols_atualizadas])

            # Roda a Cura da Malha antes do Join
            if "H3_INDEX" in cols_atualizadas:
                self._curar_malha_referencia(lf.select(["H3_INDEX", "NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL"]).collect())

            lf_enriquecido = lf.join(self.df_malha_lazy, on="H3_INDEX", how="left")
            lf_enriquecido = lf_enriquecido.with_columns([
                pl.coalesce([pl.col("NM_MUN"), pl.col("NM_MUN_ORIGINAL")]).alias("NM_MUN_FINAL"),
                pl.coalesce([pl.col("NM_BAIRRO"), pl.col("NM_BAIRRO_ORIGINAL")]).alias("NM_BAIRRO_FINAL")
            ])

            # O FILTRO GENIAL: Remove a capa sem precisar de regex
            lf_enriquecido = lf_enriquecido.filter(
                (pl.col("H3_INDEX").is_not_null()) & 
                (pl.col("NM_MUN_FINAL") != "INDEFINIDO") & 
                (pl.col("NM_BAIRRO_FINAL") != "INDEFINIDO")
            )

            # --- SUA LÓGICA ANTIGA DE NEGÓCIO ---
            lf_enriquecido = lf_enriquecido.with_columns([
                pl.col("HORA").str.split(":").list.first().cast(pl.Int32, strict=False).alias("HORA_INT"),
                pl.when(pl.col("CONDUTA").str.contains("TRANSEUNTE|PEDESTRE")).then(pl.lit("PEDESTRE"))
                  .when(pl.col("RUBRICA").str.contains("VEICULO|AUTO|MOTO")).then(pl.lit("MOTORISTA"))
                  .otherwise(pl.lit("PEDESTRE")).alias("PERFIL_ALVO")
            ]).with_columns([
                pl.when((pl.col("HORA_INT") > 0) & (pl.col("HORA_INT") < 6)).then(pl.lit("MADRUGADA"))
                  .when((pl.col("HORA_INT") >= 6) & (pl.col("HORA_INT") < 12)).then(pl.lit("MANHA"))
                  .when((pl.col("HORA_INT") >= 12) & (pl.col("HORA_INT") < 18)).then(pl.lit("TARDE"))
                  .when((pl.col("HORA_INT") >= 18) & (pl.col("HORA_INT") <= 23)).then(pl.lit("NOITE"))
                  .when(pl.col("PERIODO_TEXTO").str.contains("MADRUGADA")).then(pl.lit("MADRUGADA"))
                  .when(pl.col("PERIODO_TEXTO").str.contains("MANHA")).then(pl.lit("MANHA"))
                  .when(pl.col("PERIODO_TEXTO").str.contains("TARDE")).then(pl.lit("TARDE"))
                  .when(pl.col("PERIODO_TEXTO").str.contains("NOITE")).then(pl.lit("NOITE"))
                  .otherwise(pl.lit("MADRUGADA")).alias("PERIODO_DIA")
            ])

            # --- AGREGAÇÃO (COM AS COLUNAS DA IA) ---
            lf_agg = lf_enriquecido.group_by(["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO", "TIPO_LOCAL", "MES_OCORRENCIA", "DIA_SEMANA_OCORRENCIA"]).agg([
                pl.when(pl.col("RUBRICA").str.contains("ROUBO")).then(3).otherwise(1).sum().alias("TOTAL_CRIMES"),
                pl.col("NM_MUN_FINAL").first().alias("NM_MUN"),
                pl.col("NM_BAIRRO_FINAL").first().alias("NM_BAIRRO"),
                pl.col("DENSIDADE_AJUSTADA").cast(pl.Float64).first().alias("DENSIDADE"),
                pl.col("TAXA_VACANCIA").cast(pl.Float64).first().alias("TAXA_VACANCIA")
            ])

            # INJETA AS FEATURES ESPACIAIS DA IA
            df_final_pd = self._gerar_features_espaciais_ia(lf_agg.collect().to_pandas())

            lf_final = pl.from_pandas(df_final_pd).with_columns([
                (pl.col("TOTAL_CRIMES").rank().over("PERIODO_DIA") / pl.col("TOTAL_CRIMES").count().over("PERIODO_DIA")).alias("RANKING_RISCO_LOCAL"),
                (pl.col("TOTAL_CRIMES") / (pl.col("DENSIDADE").fill_null(0) + 1)).alias("INDICE_EXPOSICAO"),
                pl.lit(ano).alias("ANO_REFERENCIA")
            ])

            df_final = lf_final.collect(engine="streaming") if hasattr(lf_final, "collect") else lf_final
            
            buffer = io.BytesIO()
            df_final.write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            # Salva a Malha curada no R2 no final do processo
            buffer_malha = io.BytesIO()
            self.df_malha.write_parquet(buffer_malha, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=self.malha_path, Body=buffer_malha.getvalue())

            estado[str(ano)] = tamanho_atual 
            return {"linhas_in": total_in, "linhas_out": df_final.height}

        except Exception as e:
            logger.error(f"PRATA: Erro critico no ano {ano}: {e}")
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
