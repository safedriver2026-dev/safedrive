import os
import boto3
import requests
import logging
import io
import polars as pl
from botocore.config import Config
from botocore.exceptions import ClientError
from datetime import datetime

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class IngestaoBronze:
    def __init__(self):
        # Credenciais e Endpoints (Cloudflare R2)
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
        
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        }

    def executar_ingestao_continua(self):
        """Varre os anos de interesse, realiza o CDC e gera a camada Trusted limpa."""
        logger.info("BRONZE: Iniciando extração com limpeza de metadados (Smart-Skip).")
        ano_atual = datetime.now().year
        novos_dados_ingeridos = False

        for ano in range(2022, ano_atual + 1):
            if self._verificar_e_baixar(ano):
                novos_dados_ingeridos = True

        return novos_dados_ingeridos

    def _verificar_e_baixar(self, ano):
        path_raw = f"datalake/bronze/raw/ssp_raw_{ano}.xlsx"
        path_trusted = f"datalake/bronze/trusted/ssp_trusted_{ano}.parquet"
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"

        # 1. Validação de Mudança (CDC) e Existência
        tamanho_r2 = 0
        try:
            meta_r2 = self.s3.head_object(Bucket=self.bucket, Key=path_raw)
            tamanho_r2 = meta_r2.get('ContentLength', 0)
        except ClientError:
            tamanho_r2 = 0

        try:
            resp_head = requests.head(url, headers=self.headers, timeout=30)
            tamanho_ssp = int(resp_head.headers.get('Content-Length', -1)) if resp_head.status_code == 200 else -1
        except:
            tamanho_ssp = -1

        # Check se o Trusted já existe para evitar re-conversão desnecessária
        trusted_existe = False
        try:
            self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            trusted_existe = True
        except:
            trusted_existe = False

        # Se o arquivo no R2 é igual ao da SSP e o Parquet já existe, não faz nada
        if trusted_existe and tamanho_ssp > 0 and tamanho_ssp == tamanho_r2:
            logger.info(f"BRONZE: [{ano}] Sincronizado. Pulando.")
            return False

        logger.info(f"BRONZE: [{ano}] Iniciando processamento de purificação...")

        try:
            # 2. Obtenção do Binário (Download ou R2)
            if tamanho_ssp > 0 and tamanho_ssp == tamanho_r2:
                logger.info(f"BRONZE: [{ano}] Usando cache Raw do R2.")
                obj = self.s3.get_object(Bucket=self.bucket, Key=path_raw)
                excel_bytes = obj['Body'].read()
            else:
                logger.info(f"BRONZE: [{ano}] Baixando nova versão da SSP-SP.")
                response = requests.get(url, headers=self.headers, timeout=300)
                response.raise_for_status()
                excel_bytes = response.content
                self.s3.put_object(Bucket=self.bucket, Key=path_raw, Body=excel_bytes)

            # 3. Extração e Limpeza de Cabeçalhos (A "Cura")
            xlsx_io = io.BytesIO(excel_bytes)
            dfs = []
            
            # Varre até 6 abas (procurando dados reais)
            for i in range(1, 7):
                try:
                    # Leitura inicial para testar se o cabeçalho é lixo/capa
                    df_check = pl.read_excel(xlsx_io, sheet_id=i, engine="calamine", n_rows=1)
                    if df_check.is_empty(): continue
                    
                    primeira_col = str(df_check.columns[0]).upper()
                    
                    # Lógica Smart-Skip: Se a "coluna" for o texto de aviso do governo, pula a linha
                    if "PRESENTE TABELA" in primeira_col or "FINALIDADE" in primeira_col or "__UNNAMED__" in primeira_col:
                        logger.info(f"BRONZE: [{ano}] Lixo detectado na aba {i}. Aplicando skip_rows...")
                        df = pl.read_excel(xlsx_io, sheet_id=i, engine="calamine", read_options={"skip_rows": 1})
                    else:
                        df = pl.read_excel(xlsx_io, sheet_id=i, engine="calamine")

                    # Normalização de tipos e nomes
                    df = df.with_columns(pl.all().cast(pl.String))
                    df.columns = [c.upper().replace("Ç", "C").replace("Ã", "A").strip() for c in df.columns]

                    # Mapeamento de sobrevivência para MUNICIPIO
                    mapeamento = {
                        "CIDADE": "MUNICIPIO",
                        "NOME_MUNICIPIO": "MUNICIPIO",
                        "NOME_MUNICIPIO_CIRCUNSCRICAO": "MUNICIPIO"
                    }
                    for original, novo in mapeamento.items():
                        if original in df.columns:
                            df = df.rename({original: novo})
                    
                    # Só aceita a aba se tiver as colunas mínimas de um BO
                    if "MUNICIPIO" in df.columns or "LOGRADOURO" in df.columns:
                        dfs.append(df)
                        
                except Exception:
                    continue

            if dfs:
                # Consolidação final
                df_trusted = pl.concat(dfs, how="diagonal")
                
                # Double-check para colunas que podem ter vindo com nomes sujos
                for col in df_trusted.columns:
                    if "MUNICIPIO" in col and col != "MUNICIPIO":
                         df_trusted = df_trusted.rename({col: "MUNICIPIO"})

                # Persistência na camada Trusted
                parquet_buffer = io.BytesIO()
                df_trusted.write_parquet(parquet_buffer)
                
                self.s3.put_object(
                    Bucket=self.bucket, 
                    Key=path_trusted, 
                    Body=parquet_buffer.getvalue()
                )
                
                logger.info(f"BRONZE: [{ano}] Camada Trusted Parquet gerada com sucesso.")
                return True
            
            return False

        except Exception as e:
            logger.error(f"BRONZE: [{ano}] Erro crítico: {e}")
            return False
