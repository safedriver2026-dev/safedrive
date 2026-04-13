import polars as pl
import boto3
import io
import os
import h3
import logging
from datetime import datetime
from botocore.exceptions import ClientError

# Configuração de Logs para monitorização no GitHub Actions
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ProcessamentoPrata:
    def __init__(self):
        # 🛡️ SANITIZAÇÃO: .strip() remove espaços ou quebras de linha nas credenciais
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip()
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()

        self.s3 = boto3.client(
            's3',
            endpoint_url=self.endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
        
        # Resolução H3 Nível 8 (Equilíbrio ideal para análise de risco urbano)
        self.h3_resolution = 8

    def executar_prata(self, ano):
        """
        Processa o ano solicitado: Bronze (CSV) -> Prata (Parquet).
        Aplica a lógica de H3 e enriquecimento de infraestrutura.
        """
        path_bronze = f"datalake/bronze/ssp_raw_{ano}.csv"
        path_prata = f"datalake/prata/ssp_consolidada_{ano}.parquet"

        try:
            # 1. Verificação de Idempotência
            try:
                self.s3.head_object(Bucket=self.bucket, Key=path_prata)
                logger.info(f"✅ [PRATA] O ano {ano} já está processado. A avançar.")
                return True
            except:
                logger.info(f"🥈 [PRATA] A iniciar refinamento do ano {ano}...")

            # 2. Carregar dados brutos da Camada Bronze
            resp = self.s3.get_object(Bucket=self.bucket, Key=path_bronze)
            lf = pl.read_csv(io.BytesIO(resp['Body'].read()), ignore_errors=True)

            # 3. Limpeza Geoespacial (Remove coordenadas inválidas ou fora de SP)
            lf = lf.filter(
                (pl.col("LATITUDE").is_not_null()) & 
                (pl.col("LONGITUDE").is_not_null()) &
                (pl.col("LATITUDE") < -20) & (pl.col("LATITUDE") > -26)
            )

            # 4. Indexação H3
            # Convertemos para Pandas para usar a biblioteca h3-py de forma vetorizada
            df_pandas = lf.to_pandas()
            df_pandas['h3_index'] = df_pandas.apply(
                lambda row: h3.geo_to_h3(row['LATITUDE'], row['LONGITUDE'], self.h3_resolution), 
                axis=1
            )
            lf = pl.from_pandas(df_pandas)

            # 5. Inteligência de Infraestrutura (Separação Residencial vs Comercial)
            logger.info(f"🏗️ A calcular métricas de ocupação de solo para {ano}...")
            lf = self._aplicar_engenharia_atributos(lf, ano)

            # 6. Gravação em formato Parquet (Otimizado para a IA)
            buffer = io.BytesIO()
            lf.write_parquet(buffer)
            buffer.seek(0)

            logger.info(f"📤 A enviar Camada Prata consolidada para o R2: {path_prata}")
            self.s3.put_object(
                Bucket=self.bucket,
                Key=path_prata,
                Body=buffer.getvalue()
            )
            return True

        except Exception as e:
            logger.error(f"❌ Erro no processamento Prata de {ano}: {e}")
            raise e

    def _aplicar_engenharia_atributos(self, lf, ano):
        """
        Lógica solicitada: Subtrai o residencial do geral para isolar o comercial/industrial.
        Calcula também a taxa de ocupação para o modelo de IA.
        """
        return lf.with_columns([
            pl.lit(ano).alias("ANO_REFERENCIA"),
            
            # Cálculo de endereços Não-Residenciais (Comerciais/Industriais/Públicos)
            (pl.col("TOTAL_GERAL") - pl.col("TOTAL_RESIDENCIAL")).alias("TOTAL_NAO_RESIDENCIAL"),
            
            # Taxa de Residencialidade (Métrica crucial para o CatBoost)
            (pl.col("TOTAL_RESIDENCIAL") / pl.col("TOTAL_GERAL")).alias("INDICE_RESIDENCIAL"),
            
            # Densidade de Ocupação
            (pl.col("TOTAL_GERAL") / pl.col("AREA_HEXAGONO")).alias("DENSIDADE_ENDERECOS")
        ])

# ==========================================
# EXECUÇÃO HISTÓRICA COMPLETA
# ==========================================
if __name__ == "__main__":
    prata = ProcessamentoPrata()
    ano_inicio = 2022
    ano_fim = datetime.now().year
    
    for a in range(ano_inicio, ano_fim + 1):
        prata.executar_prata(a)
