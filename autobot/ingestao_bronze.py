import os
import boto3
import requests
import logging
import io
import polars as pl
from botocore.config import Config
from botocore.exceptions import ClientError
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class IngestaoBronze:
    def __init__(self):
        # Configurações do Cloudflare R2 / S3
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
        """Varre os anos de interesse e garante a sincronia das camadas Raw e Trusted."""
        logger.info("BRONZE: Iniciando rotina de extração (Raw) e Padronização (Trusted).")
        ano_atual = datetime.now().year
        novos_dados_ingeridos = False

        # Processa de 2022 até o ano atual
        for ano in range(2022, ano_atual + 1):
            if self._verificar_e_baixar(ano):
                novos_dados_ingeridos = True

        return novos_dados_ingeridos

    def _verificar_e_baixar(self, ano):
        path_raw = f"datalake/bronze/raw/ssp_raw_{ano}.xlsx"
        path_trusted = f"datalake/bronze/trusted/ssp_trusted_{ano}.parquet"
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"

        # 1. Verificar se o Trusted (Parquet) já existe
        trusted_existe = False
        try:
            self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            trusted_existe = True
        except ClientError:
            trusted_existe = False

        # 2. Verificar o tamanho do Raw no R2 (se existir)
        tamanho_r2_raw = 0
        try:
            meta_r2 = self.s3.head_object(Bucket=self.bucket, Key=path_raw)
            tamanho_r2_raw = meta_r2.get('ContentLength', 0)
        except ClientError:
            tamanho_r2_raw = 0

        # 3. Verificar o tamanho do arquivo no site da SSP (CDC)
        try:
            resp_head = requests.head(url, headers=self.headers, timeout=30)
            tamanho_ssp = int(resp_head.headers.get('Content-Length', -1)) if resp_head.status_code == 200 else -1
        except:
            tamanho_ssp = -1

        # LÓGICA DE DECISÃO:
        # Se o Parquet existe E o Excel no R2 tem o mesmo tamanho do site, não faz nada.
        if trusted_existe and tamanho_ssp > 0 and tamanho_ssp == tamanho_r2_raw:
            logger.info(f"BRONZE: [{ano}] Camadas Raw e Trusted já estão atualizadas. Pulando.")
            return False

        logger.info(f"BRONZE: [{ano}] Necessário processar (Trusted_faltante={not trusted_existe} ou Mudança_detectada={tamanho_ssp != tamanho_r2_raw}).")

        try:
            # 4. Obter o conteúdo do Excel (do R2 se for igual ao site, ou download se for novo)
            if tamanho_ssp > 0 and tamanho_ssp == tamanho_r2_raw:
                logger.info(f"BRONZE: [{ano}] Recuperando Excel do R2 para gerar Trusted...")
                obj = self.s3.get_object(Bucket=self.bucket, Key=path_raw)
                excel_bytes = obj['Body'].read()
            else:
                logger.info(f"BRONZE: [{ano}] Baixando nova versão do Excel da SSP-SP...")
                response = requests.get(url, headers=self.headers, timeout=300)
                response.raise_for_status()
                excel_bytes = response.content
                # Salva a nova "Fonte da Verdade" no Raw
                self.s3.put_object(Bucket=self.bucket, Key=path_raw, Body=excel_bytes)

            # 5. REFINAMENTO E PADRONIZAÇÃO (A "Cura" do Schema)
            logger.info(f"BRONZE: [{ano}] Convertendo e normalizando colunas para Parquet...")
            dfs = []
            
            # Percorre as abas (o Polars com calamine é excelente para isso)
            for i in range(1, 6):
                try:
                    df = pl.read_excel(io.BytesIO(excel_bytes), sheet_id=i, engine="calamine")
                    # Força tudo para String para evitar erros de tipo na concatenação
                    df = df.with_columns(pl.all().cast(pl.String))
                    # Limpeza básica de nomes de colunas
                    df.columns = [c.upper().replace("Ç", "C").replace("Ã", "A").strip() for c in df.columns]
                    
                    # --- MAPEAMENTO DE SINÔNIMOS (Resolve o erro de MUNICIPIO) ---
                    mapeamento = {
                        "CIDADE": "MUNICIPIO",
                        "NOME_MUNICIPIO": "MUNICIPIO"
                    }
                    for original, novo in mapeamento.items():
                        if original in df.columns:
                            df = df.rename({original: novo})
                            
                    dfs.append(df)
                except Exception:
                    continue # Pula abas vazias ou com erro
            
            if dfs:
                # Une as abas e salva como Parquet Trusted
                df_trusted = pl.concat(dfs, how="diagonal")
                buffer = io.BytesIO()
                df_trusted.write_parquet(buffer)
                
                self.s3.put_object(
                    Bucket=self.bucket, 
                    Key=path_trusted, 
                    Body=buffer.getvalue()
                )
                
                logger.info(f"BRONZE: [{ano}] Camada Trusted gerada com sucesso.")
                return True
            
            return False

        except Exception as e:
            logger.error(f"BRONZE: [{ano}] Falha crítica no processamento: {e}")
            return False
