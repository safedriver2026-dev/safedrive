import os, boto3, requests, logging, io
import polars as pl
import h3
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
        
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0"}
        
        # Localização automática do Data Lake
        self.base_path = self._descobrir_prefixo_datalake()
        logger.info(f"BRONZE: Raiz do Data Lake detectada em: '{self.base_path}'")

    def _descobrir_prefixo_datalake(self):
        """Busca dinamicamente onde a pasta 'datalake' reside no bucket."""
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, MaxKeys=15)
            if 'Contents' in response:
                keys = [obj['Key'] for obj in response['Contents']]
                for key in keys:
                    if "safedriver/datalake" in key: return "safedriver/datalake"
                    if "datalake" in key: return "datalake"
            return "datalake"
        except: return "datalake"

    def _get_path(self, camada, subpasta, filename):
        return f"{self.base_path}/{camada}/{subpasta}/{filename}".replace("//", "/")

    def _motor_h3(self, lat, lng):
        """Converte coordenadas para H3 Resolução 9 com tratamento de erro."""
        try:
            l1, l2 = float(lat), float(lng)
            # Filtro básico de coordenadas válidas (evita o meio do oceano ou valores nulos)
            if l1 == 0 or l2 == 0 or abs(l1) > 90 or abs(l2) > 180:
                return None
            return h3.latlng_to_cell(l1, l2, 9)
        except:
            return None

    def executar_ingestao_continua(self, force=False):
        logger.info(f"BRONZE: Iniciando rotina de purificação com H3 (Force={force}).")
        novos_dados = False
        for ano in range(2022, datetime.now().year + 1):
            if self._verificar_e_baixar(ano, force):
                novos_dados = True
        return novos_dados

    def _verificar_e_baixar(self, ano, force=False):
        path_raw = self._get_path("bronze", "raw", f"ssp_raw_{ano}.xlsx")
        path_trusted = self._get_path("bronze", "trusted", f"ssp_trusted_{ano}.parquet")
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"

        # Check de existência para evitar re-processamento
        try:
            self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
            if not force:
                logger.info(f"BRONZE: [{ano}] Camada Trusted já existe. Pulando.")
                return False
        except: pass

        logger.info(f"BRONZE: [{ano}] Processando dados e calculando índices H3...")
        
        try:
            # Recupera o binário do Excel (Cache R2 ou Download SSP)
            try:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path_raw)
                excel_bytes = obj['Body'].read()
                logger.info(f"BRONZE: [{ano}] Usando cache Raw do R2.")
            except:
                logger.info(f"BRONZE: [{ano}] Baixando da SSP-SP...")
                resp = requests.get(url, headers=self.headers, timeout=300)
                excel_bytes = resp.content
                self.s3.put_object(Bucket=self.bucket, Key=path_raw, Body=excel_bytes)

            xlsx_io = io.BytesIO(excel_bytes)
            dfs = []
            
            # Header-Hunter para as abas do Excel
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
                        
                        # Padronização de Colunas
                        mapeamento = {"CIDADE": "MUNICIPIO", "NOME_MUNICIPIO": "MUNICIPIO"}
                        for old, new in mapeamento.items():
                            if old in df.columns: df = df.rename({old: new})
                        
                        dfs.append(df)
                except: continue

            if dfs:
                df_trusted = pl.concat(dfs, how="diagonal")
                
                # --- INTEGRAÇÃO H3 ---
                logger.info(f"BRONZE: [{ano}] Calculando hexágonos H9 para {df_trusted.height} registros...")
                
                df_trusted = df_trusted.with_columns([
                    pl.col("LATITUDE").cast(pl.Float64, strict=False).fill_null(0.0),
                    pl.col("LONGITUDE").cast(pl.Float64, strict=False).fill_null(0.0)
                ]).with_columns(
                    pl.struct(["LATITUDE", "LONGITUDE"])
                    .map_elements(lambda x: self._motor_h3(x["LATITUDE"], x["LONGITUDE"]), return_dtype=pl.String)
                    .alias("H3_INDEX")
                ).filter(pl.col("H3_INDEX").is_not_null())

                # Salvamento em Parquet (Trusted)
                buffer = io.BytesIO()
                df_trusted.write_parquet(buffer)
                self.s3.put_object(Bucket=self.bucket, Key=path_trusted, Body=buffer.getvalue())
                
                logger.info(f"BRONZE: [{ano}] Trusted gerado com H3_INDEX integrado.")
                return True
                
        except Exception as e:
            logger.error(f"BRONZE: Falha crítica no ano {ano}: {e}")
        return False
