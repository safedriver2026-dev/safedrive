import polars as pl
import pandas as pd
import h3
import boto3
import io
import os
import json
import logging
import traceback
from botocore.config import Config
from datetime import datetime
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfiguracaoProcessamento:
    RESOLUCAO_H3 = 9
    TAMANHO_JANELA = 3
    NOME_BUCKET = os.getenv("R2_BUCKET_NAME", "").strip()
    CAMINHO_BASE = "datalake"
    
    MAPA_DE_COLUNAS = {
        "NM_MUN_ORIGINAL": ["NOME_MUNICIPIO", "CIDADE"],
        "DATA_BRUTA": ["DATA_OCORRENCIA_BO", "DATA_OCORRENCIA"],
        "HORA": ["HORA_OCORRENCIA_BO", "HORA"],
        "PERIODO_SSP": ["DESC_PERIODO", "DESCR_PERIODO"],
        "TIPO_LOCAL": ["DESCR_SUBTIPOLOCAL", "DESCR_TIPOLOCAL"],
        "NM_BAIRRO_ORIGINAL": ["BAIRRO"]
    }

class ProcessadorPrata:
    def __init__(self):
        self.configuracao = ConfiguracaoProcessamento()
        self.cliente_armazenamento = self._inicializar_cliente_armazenamento()
        self.caminho_raiz = self._descobrir_raiz_datalake()
        self.caminho_malha = f"{self.caminho_raiz}/base_geografica/safedriver_geo_base_sp_h3_{self.configuracao.RESOLUCAO_H3}.parquet"
        self.dataframe_malha = self._carregar_malha_geografica()

    def _inicializar_cliente_armazenamento(self):
        return boto3.client(
            's3',
            endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(
                signature_version='s3v4',
                retries={'max_attempts': 5, 'mode': 'standard'},
                max_pool_connections=50
            )
        )

    def _descobrir_raiz_datalake(self) -> str:
        try:
            resposta = self.cliente_armazenamento.list_objects_v2(Bucket=self.configuracao.NOME_BUCKET, MaxKeys=50)
            for objeto in resposta.get('Contents', []):
                if "datalake/" in objeto['Key']:
                    return objeto['Key'].split("datalake/")[0] + "datalake"
            return "datalake"
        except Exception:
            return "datalake"

    def _carregar_malha_geografica(self) -> Optional[pl.DataFrame]:
        try:
            resposta = self.cliente_armazenamento.get_object(Bucket=self.configuracao.NOME_BUCKET, Key=self.caminho_malha)
            return pl.read_parquet(io.BytesIO(resposta['Body'].read())).unique(subset=["H3_INDEX"])
        except Exception as erro:
            logger.error(f"Erro ao carregar malha base: {erro}")
            return None

    def _validar_integridade_dados(self, dataframe: pl.DataFrame):
        if dataframe.height == 0:
            raise ValueError("O DataFrame processado esta vazio.")
        
        contagem_nulos = dataframe.filter(pl.col("H3_INDEX").is_null()).height
        if contagem_nulos > 0:
            logger.warning(f"Detectados {contagem_nulos} registros sem H3_INDEX. Removendo.")
            return dataframe.drop_nulls(subset=["H3_INDEX"])
        
        return dataframe

    def _aplicar_transformacoes_temporais(self, estrutura_preguicosa: pl.LazyFrame) -> pl.LazyFrame:
        colunas_atuais = estrutura_preguicosa.collect_schema().names()
        
        expressoes_mapeamento = []
        colunas_para_remover = []
        
        for destino, origens in self.configuracao.MAPA_DE_COLUNAS.items():
            presentes = [o for o in origens if o in colunas_atuais]
            if presentes:
                expressoes_mapeamento.append(pl.coalesce(presentes).alias(destino))
                colunas_para_remover.extend([o for o in presentes if o != destino])

        return estrutura_preguicosa.with_columns(expressoes_mapeamento).drop(colunas_para_remover).with_columns([
            pl.col("DATA_BRUTA").str.to_date(format="%d/%m/%Y", strict=False).alias("DATA"),
            pl.col("HORA").str.split(":").list.first().cast(pl.Int8, strict=False).alias("HORA_INT"),
            
            pl.when(pl.col("RUBRICA").str.contains("(?i)HOMICIDIO|LATROCINIO|ESTUPRO|MORTE")).then(10.0)
              .when(pl.col("RUBRICA").str.contains("(?i)ROUBO")).then(5.0)
              .when(pl.col("RUBRICA").str.contains("(?i)VEICULO|AUTO|MOTO")).then(3.0)
              .otherwise(1.0).alias("PESO_CRIME"),
            
            pl.when(pl.col("RUBRICA").str.contains("(?i)VEICULO|AUTO|MOTO")).then(pl.lit("MOTORISTA"))
              .otherwise(pl.lit("PEDESTRE")).alias("PERFIL_ALVO")
        ]).with_columns([
            pl.when(pl.col("HORA_INT").is_between(0, 5)).then(pl.lit("MADRUGADA"))
              .when(pl.col("HORA_INT").is_between(6, 11)).then(pl.lit("MANHA"))
              .when(pl.col("HORA_INT").is_between(12, 17)).then(pl.lit("TARDE"))
              .otherwise(pl.lit("NOITE")).alias("PERIODO_DIA")
        ])

    def processar_ciclo(self, ano: int, estado: dict, forcar_execucao: bool = False) -> bool:
        caminho_entrada = f"{self.caminho_raiz}/bronze/trusted/ssp_trusted_{ano}.parquet"
        caminho_saida = f"{self.caminho_raiz}/prata/ssp_consolidada_{ano}.parquet"
        
        try:
            metadados = self.cliente_armazenamento.head_object(Bucket=self.configuracao.NOME_BUCKET, Key=caminho_entrada)
            if not forcar_execucao and estado.get(str(ano)) == metadados['ContentLength']: 
                return False

            logger.info(f"[{ano}] Iniciando processamento de serie temporal.")
            resposta = self.cliente_armazenamento.get_object(Bucket=self.configuracao.NOME_BUCKET, Key=caminho_entrada)
            estrutura_preguicosa = pl.read_parquet(io.BytesIO(resposta['Body'].read())).lazy()

            estrutura_preguicosa = self._aplicar_transformacoes_temporais(estrutura_preguicosa)
            
            dataframe_agregado = estrutura_preguicosa.group_by([
                "H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO", "TIPO_LOCAL", 
                "NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL",
                pl.col("DATA").dt.month().alias("MES_OCORRENCIA")
            ]).agg([
                pl.len().alias("TOTAL_CRIMES"),
                pl.col("PESO_CRIME").sum().alias("INDICE_GRAVIDADE"),
                pl.col("DATA").dt.weekday().first().alias("DIA_SEMANA")
            ])

            dataframe_final = dataframe_agregado.sort(["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO", "MES_OCORRENCIA"]).with_columns([
                pl.col("INDICE_GRAVIDADE").shift(1).over(["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO"])
                  .rolling_mean(window_size=self.configuracao.TAMANHO_JANELA).fill_null(0.0).alias("GRAVIDADE_HISTORICA"),
                
                pl.col("TOTAL_CRIMES").shift(1).over(["H3_INDEX", "PERIODO_DIA", "PERFIL_ALVO"])
                  .rolling_mean(window_size=self.configuracao.TAMANHO_JANELA).fill_null(0.0).alias("VOLUME_HISTORICO")
            ]).collect()

            dataframe_final = self._validar_integridade_dados(dataframe_final)
            dataframe_pandas = dataframe_final.to_pandas()
            
            gravidade_historica_por_h3 = dataframe_pandas.groupby("H3_INDEX")["GRAVIDADE_HISTORICA"].mean().to_dict()
            mapa_contagio = {}
            
            for indice_hexagonal in dataframe_pandas['H3_INDEX'].unique():
                try:
                    vizinhos = h3.k_ring(indice_hexagonal, 1)
                    mapa_contagio[indice_hexagonal] = sum(gravidade_historica_por_h3.get(vizinho, 0.0) for vizinho in vizinhos if vizinho != indice_hexagonal)
                except: 
                    mapa_contagio[indice_hexagonal] = 0.0

            dataframe_pandas['CONTAGIO_PONDERADO'] = dataframe_pandas['H3_INDEX'].map(mapa_contagio).fillna(0.0)
            dataframe_pandas['PESO_TOTAL_H3'] = dataframe_pandas['H3_INDEX'].map(gravidade_historica_por_h3).fillna(0.0) 
            dataframe_pandas['PRESSAO_RISCO_LOCAL'] = dataframe_pandas['CONTAGIO_PONDERADO']
            
            dataframe_pandas['ANO_REF'] = ano
            dataframe_pandas['IS_FDS'] = (dataframe_pandas['DIA_SEMANA'] >= 5).astype(int)
            dataframe_pandas['IS_PAGAMENTO'] = 0 

            dataframe_final = pl.from_pandas(dataframe_pandas)

            if self.dataframe_malha is not None:
                dataframe_final = dataframe_final.join(
                    self.dataframe_malha.select(["H3_INDEX", "DENSIDADE_AJUSTADA", "TAXA_VACANCIA", "NM_MUN", "NM_BAIRRO"]), 
                    on="H3_INDEX", how="left"
                ).with_columns([
                    pl.col("DENSIDADE_AJUSTADA").alias("DENSIDADE"),
                    pl.coalesce(["NM_MUN", "NM_MUN_ORIGINAL"]).alias("NM_MUN"),
                    pl.coalesce(["NM_BAIRRO", "NM_BAIRRO_ORIGINAL"]).alias("NM_BAIRRO"),
                    pl.col("TIPO_LOCAL").fill_null("VIA PUBLICA")
                ]).drop(["NM_MUN_ORIGINAL", "NM_BAIRRO_ORIGINAL", "DENSIDADE_AJUSTADA"])

            buffer = io.BytesIO()
            dataframe_final.write_parquet(buffer, compression="lz4")
            self.cliente_armazenamento.put_object(Bucket=self.configuracao.NOME_BUCKET, Key=caminho_saida, Body=buffer.getvalue())

            estado[str(ano)] = metadados['ContentLength']
            return True
            
        except Exception as erro:
            logger.error(f"Falha de processamento no ano {ano}: {erro}")
            traceback.print_exc()
            return False

    def executar_completo(self, forcar_execucao: bool = False):
        logger.info("Iniciando consolidacao com media movel temporal.")
        estado = self._carregar_rastreio()
        ano_atual = datetime.now().year
        for ano in range(2022, ano_atual + 1):
            if self.processar_ciclo(ano, estado, forcar_execucao):
                self._salvar_rastreio(estado)
        return True

    def _carregar_rastreio(self):
        try:
            resposta = self.cliente_armazenamento.get_object(Bucket=self.configuracao.NOME_BUCKET, Key=f"{self.caminho_raiz}/prata/tracker.json")
            return json.loads(resposta['Body'].read())
        except: 
            return {}

    def _salvar_rastreio(self, estado):
        self.cliente_armazenamento.put_object(
            Bucket=self.configuracao.NOME_BUCKET, 
            Key=f"{self.caminho_raiz}/prata/tracker.json", 
            Body=json.dumps(estado)
        )

if __name__ == "__main__":
    processador = ProcessadorPrata()
    processador.executar_completo(forcar_execucao=True)
