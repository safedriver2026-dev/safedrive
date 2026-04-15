import os, boto3, requests, logging, io
import polars as pl
from botocore.config import Config
from botocore.exceptions import ClientError
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class IngestaoBronze:
    def __init__(self):
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}))
        
        self.headers = {"User-Agent": "Mozilla/5.0"}
        
        # --- LÓGICA DE LOCALIZAÇÃO AUTOMÁTICA ---
        self.base_path = self._descobrir_prefixo_datalake()
        logger.info(f"BRONZE: Raiz do Data Lake detectada em: '{self.base_path}'")

    def _descobrir_prefixo_datalake(self):
        """
        Procura dinamicamente se a pasta 'datalake' está na raiz ou dentro de 'safedriver/'.
        """
        try:
            # Lista os primeiros objetos do bucket para 'farejar' o caminho
            response = self.s3.list_objects_v2(Bucket=self.bucket, MaxKeys=10)
            if 'Contents' in response:
                keys = [obj['Key'] for obj in response['Contents']]
                for key in keys:
                    if "safedriver/datalake" in key:
                        return "safedriver/datalake"
                    if "datalake" in key and not key.startswith("safedriver"):
                        return "datalake"
            return "datalake" # Default caso o bucket esteja vazio
        except Exception as e:
            logger.warning(f"Não foi possível listar o bucket para auto-discovery: {e}")
            return "datalake"

    def _get_path(self, camada, subpasta, filename):
        """Helper para montar caminhos sem erro de barra ou prefixo."""
        return f"{self.base_path}/{camada}/{subpasta}/{filename}".replace("//", "/")

    def executar_ingestao_continua(self, force=False):
        logger.info(f"BRONZE: Iniciando rotina (Force={force}).")
        novos_dados = False
        for ano in range(2022, datetime.now().year + 1):
            if self._verificar_e_baixar(ano, force): novos_dados = True
        return novos_dados

    def _verificar_e_baixar(self, ano, force=False):
        # Caminhos montados dinamicamente
        path_raw = self._get_path("bronze", "raw", f"ssp_raw_{ano}.xlsx")
        path_trusted = self._get_path("bronze", "trusted", f"ssp_trusted_{ano}.parquet")
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"

        try:
            # 1. Tenta recuperar o Excel (Cache R2 ou Download)
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path_raw)
                excel_bytes = obj['Body'].read()
                logger.info(f"BRONZE: [{ano}] Usando cache Raw do R2.")
            except:
                logger.info(f"BRONZE: [{ano}] Baixando da SSP-SP...")
                resp = requests.get(url, headers=self.headers, timeout=300)
                excel_bytes = resp.content
                self.s3.put_object(Bucket=self.bucket, Key=path_raw, Body=excel_bytes)

            # 2. Processamento com Header-Hunter
            xlsx_io = io.BytesIO(excel_bytes)
            dfs = []
            for i in range(1, 7):
                try:
                    df_raw = pl.read_excel(xlsx_io, sheet_id=i, engine="calamine", has_header=False)
                    header_idx = None
                    for idx, row in enumerate(df_raw.slice(0, 30).iter_rows()):
                        row_txt = " ".join([str(v).upper() for v in row if v is not None])
                        if any(x in row_txt for x in ["LOGRADOURO", "MUNICIPIO", "CIDADE", "RUBRICA"]):
                            header_idx = idx
                            break
                    
                    if header_idx is not None:
                        df = pl.read_excel(xlsx_io, sheet_id=i, engine="calamine", read_options={"skip_rows": header_idx})
                        df = df.with_columns(pl.all().cast(pl.String))
                        df.columns = [c.upper().replace("Ç", "C").replace("Ã", "A").strip() for c in df.columns]
                        
                        mapeamento = {"CIDADE": "MUNICIPIO", "NOME_MUNICIPIO": "MUNICIPIO"}
                        for old, new in mapeamento.items():
                            if old in df.columns: df = df.rename({old: new})
                        
                        if "MUNICIPIO" in df.columns:
                            dfs.append(df)
                except: continue

            if dfs:
                df_trusted = pl.concat(dfs, how="diagonal")
                buffer = io.BytesIO()
                df_trusted.write_parquet(buffer)
                
                # Salvamento dinâmico
                self.s3.put_object(Bucket=self.bucket, Key=path_trusted, Body=buffer.getvalue())
                logger.info(f"BRONZE: [{ano}] Trusted salvo com sucesso em: {path_trusted}")
                return True
            
        except Exception as e:
            logger.error(f"BRONZE: Erro no ano {ano}: {e}")
        return False
