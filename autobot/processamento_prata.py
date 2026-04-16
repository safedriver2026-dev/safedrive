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
    def __init__(self, raio_contagio=2):
        self.raio_contagio = raio_contagio
        self.fuso_br = ZoneInfo("America/Sao_Paulo")
        
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
            self.df_malha_lazy = (
                pl.read_parquet(io.BytesIO(resp['Body'].read()))
                .lazy()
                .with_columns([
                    self._limpar_texto_extremo("NM_MUN").alias("NM_MUN"),
                    self._limpar_texto_extremo("NM_BAIRRO").alias("NM_BAIRRO")
                ])
            )
            logger.info("PRATA: Malha geografica carregada e normalizada.")
        except Exception as e:
            logger.error(f"PRATA: Falha ao carregar malha H3: {e}")
            self.df_malha_lazy = None

    def _gerar_features_espaciais_ia(self, df_pd):
        """O UPGRADE: Contágio Espacial e Pressão de Risco."""
        crimes_dit = dict(zip(df_pd['H3_INDEX'], df_pd['TOTAL_CRIMES']))
        usar_grid_disk = hasattr(h3, 'grid_disk')
        
        contagio_final = []
        for h3_index in df_pd['H3_INDEX']:
            try:
                v1 = set(h3.grid_disk(h3_index, 1) if usar_grid_disk else h3.k_ring(h3_index, 1))
                v1.discard(h3_index)
                c1 = sum(crimes_dit.get(v, 0) for v in v1)
                
                v_total = set(h3.grid_disk(h3_index, 2) if usar_grid_disk else h3.k_ring(h3_index, 2))
                v2 = v_total - v1
                v2.discard(h3_index)
                c2 = sum(crimes_dit.get(v, 0) for v in v2)
                
                contagio_final.append((c1 * 1.0) + (c2 * 0.5))
            except: contagio_final.append(0.0)
        
        df_pd['CONTAGIO_PONDERADO'] = contagio_final
        df_pd['PRESSAO_RISCO_LOCAL'] = df_pd['CONTAGIO_PONDERADO'] / (df_pd['DENSIDADE'] + 0.001)
        return df_pd

    def processar_ano_com_delta(self, ano, estado, force=False):
        path_trusted = f"{self.base_path}/bronze/trusted/ssp_trusted_{ano}.parquet"
        path_prata = f"{self.base_path}/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            tamanho_atual = meta['ContentLength']
            
            if not force and estado.get(str(ano)) == tamanho_atual: 
                logger.info(f"PRATA: Ano {ano} em cache (sem alteracoes).")
                return None

            resp = self.s3.get_object(Bucket=self.bucket, Key=path_trusted)
            lf = pl.read_parquet(io.BytesIO(resp['Body'].read())).lazy()

            # --- 1. RESOLUÇÃO DE SCHEMA (Sua lógica antiga refinada) ---
            cols = lf.collect_schema().names()
            mapeamento = {
                "CIDADE": "NM_MUN_ORIGINAL", "MUNICIPIO": "NM_MUN_ORIGINAL", "NOME_MUNICIPIO": "NM_MUN_ORIGINAL",
                "BAIRRO": "NM_BAIRRO_ORIGINAL", "LOGRADOURO": "LOGRADOURO_ORIGINAL",
                "HORA_OCORRENCIA_BO": "HORA", "DESCR_PERIODO": "PERIODO_TEXTO", "DESC_PERIODO": "PERIODO_TEXTO",
                "DATA_OCORRENCIA_BO": "DATA_REF", "DATA_OCORRENCIA": "DATA_REF"
            }
            lf = lf.rename({old: new for old, new in mapeamento.items() if old in cols})

            # --- 2. TRATAMENTO DE DATAS (Correção para o erro 'unable to find column DATA') ---
            # Identificamos qual coluna de data sobrou e usamos ela para criar as features temporais
            cols_pos_rename = lf.collect_schema().names()
            col_data_final = "DATA_REF" if "DATA_REF" in cols_pos_rename else "DATA_OCORRENCIA_BO"
            
            if col_data_final in cols_pos_rename:
                lf = lf.with_columns([
                    pl.col(col_data_final).dt.month().alias("MES_OCORRENCIA"),
                    pl.col(col_data_final).dt.weekday().alias("DIA_SEMANA_OCORRENCIA")
                ])
            
            if "HORA" in cols_pos_rename:
                lf = lf.with_columns(
                    pl.col("HORA").str.split(":").list.first().cast(pl.Int32, strict=False).alias("HORA_INT")
                )

            # --- 3. NORMALIZAÇÃO DE TEXTO ---
            campos_texto = ["NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL", "PERIODO_TEXTO"]
            lf = lf.with_columns([self._limpar_texto_extremo(c) for c in campos_texto if c in cols_pos_rename])

            # --- 4. JOIN GEOGRÁFICO E CURA ---
            lf = lf.join(self.df_malha_lazy, on="H3_INDEX", how="left")
            lf = lf.with_columns([
                pl.coalesce([pl.col("NM_MUN"), pl.col("NM_MUN_ORIGINAL")]).alias("NM_MUN"),
                pl.coalesce([pl.col("NM_BAIRRO"), pl.col("NM_BAIRRO_ORIGINAL")]).alias("NM_BAIRRO"),
                pl.when(pl.col("HORA_INT").is_null()).then(pl.lit("MADRUGADA")) # Default seguro
                  .when(pl.col("HORA_INT") < 6).then(pl.lit("MADRUGADA"))
                  .when(pl.col("HORA_INT") < 12).then(pl.lit("MANHA"))
                  .when(pl.col("HORA_INT") < 18).then(pl.lit("TARDE"))
                  .otherwise(pl.lit("NOITE")).alias("PERIODO_DIA")
            ])

            # --- 5. AGREGAÇÃO IA ---
            # Agrupamos por H3 e Período para o Treinador IA
            lf_agg = lf.group_by(["H3_INDEX", "PERIODO_DIA", "NM_MUN", "NM_BAIRRO", "MES_OCORRENCIA", "DIA_SEMANA_OCORRENCIA"]).agg([
                pl.len().alias("TOTAL_CRIMES"),
                pl.col("DENSIDADE_AJUSTADA").first().alias("DENSIDADE"),
                pl.col("TAXA_VACANCIA").first().alias("TAXA_VACANCIA")
            ])

            # --- 6. UPGRADE DE IA (Contágio no Pandas) ---
            df_final_pd = self._gerar_features_espaciais_ia(lf_agg.collect().to_pandas())
            
            df_final = pl.from_pandas(df_final_pd).with_columns([
                (pl.col("TOTAL_CRIMES").rank() / pl.len()).alias("RANKING_RISCO_LOCAL"),
                (pl.col("TOTAL_CRIMES") / (pl.col("DENSIDADE").fill_null(0) + 1)).alias("INDICE_EXPOSICAO"),
                pl.lit(ano).alias("ANO_REFERENCIA")
            ])

            # Persistência
            buffer = io.BytesIO()
            df_final.write_parquet(buffer, compression="lz4")
            self.s3.put_object(Bucket=self.bucket, Key=path_prata, Body=buffer.getvalue())

            estado[str(ano)] = tamanho_atual 
            return {"linhas_in": lf.select(pl.len()).collect().item(), "linhas_out": df_final.height}

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
        
        # Taxa de Recuperação para o Comunicador
        stats["taxa_recuperacao"] = round((stats["linhas_out"] / stats["linhas_in"] * 100), 2) if stats["linhas_in"] > 0 else 100
        stats["status_camadas"] = {"prata": "✅ Concluido"}
        return stats

    def _carregar_tracker(self):
        try:
            resp = self.s3.get_object(Bucket=self.bucket, Key=self.tracker_path)
            return json.loads(resp['Body'].read())
        except: return {}

    def _salvar_tracker(self, estado):
        self.s3.put_object(Bucket=self.bucket, Key=self.tracker_path, Body=json.dumps(estado))

if __name__ == "__main__":
    prata = ProcessamentoPrata()
    prata.executar_todos_os_anos(force=True)
