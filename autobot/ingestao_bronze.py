import os, boto3, requests, logging, io
import polars as pl
import h3
from botocore.config import Config
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuração de Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class IngestaoBronze:
    def __init__(self):
        # Configurações de Conectividade Cloudflare R2
        self.endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
        self.access_key = os.getenv("R2_ACCESS_KEY_ID", "").strip()
        self.secret_key = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        self.bucket = os.getenv("R2_BUCKET_NAME", "").strip()
        
        self.s3 = boto3.client('s3', endpoint_url=self.endpoint, 
                              aws_access_key_id=self.access_key,
                              aws_secret_access_key=self.secret_key, 
                              config=Config(signature_version='s3v4', s3={'addressing_style': 'path'}))
        
        # 🛡️ PROTEÇÃO DE REDE: Sessão com Retry Inteligente
        self.session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        self.headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SafeDriverBot/1.0"}
        self.base_path = self._descobrir_prefixo_datalake()

    def _descobrir_prefixo_datalake(self):
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket, MaxKeys=10)
            if 'Contents' in response:
                for obj in response['Contents']:
                    if "datalake" in obj['Key']: return obj['Key'].split("datalake")[0] + "datalake"
            return "datalake"
        except: return "datalake"

    def _get_path(self, camada, subpasta, filename):
        return f"{self.base_path}/{camada}/{subpasta}/{filename}".replace("//", "/")

    def _motor_h3(self, lat, lng):
        try:
            l1, l2 = float(lat), float(lng)
            if l1 == 0 or l2 == 0 or abs(l1) > 90 or abs(l2) > 180:
                return None
            return h3.latlng_to_cell(l1, l2, 9)
        except:
            return None

    def _obter_tamanho_remoto(self, url):
        try:
            resp = self.session.head(url, headers=self.headers, timeout=20)
            return int(resp.headers.get('Content-Length', 0)) if resp.status_code == 200 else 0
        except:
            return 0

    def executar_ingestao_continua(self, force=False):
        logger.info(f"BRONZE: Iniciando ciclo de purificação (Force={force}).")
        novos_dados = False
        for ano in range(2022, datetime.now().year + 1):
            if self._verificar_e_baixar(ano, force):
                novos_dados = True
        return novos_dados

    def _verificar_e_baixar(self, ano, force=False):
        path_raw = self._get_path("bronze", "raw", f"ssp_raw_{ano}.xlsx")
        path_trusted = self._get_path("bronze", "trusted", f"ssp_trusted_{ano}.parquet")
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"

        tamanho_remoto = self._obter_tamanho_remoto(url)
        tamanho_local = 0
        try:
            meta = self.s3.head_object(Bucket=self.bucket, Key=path_raw)
            tamanho_local = meta['ContentLength']
        except: pass

        if (tamanho_remoto == tamanho_local) and not force and tamanho_remoto > 0:
            try:
                self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
                logger.info(f"BRONZE: [{ano}] Cache validado.")
                return False
            except: pass

        try:
            if tamanho_remoto != tamanho_local or force or tamanho_local == 0:
                logger.info(f"BRONZE: [{ano}] Descarregando ficheiro original...")
                resp = self.session.get(url, headers=self.headers, timeout=300)
                excel_bytes = resp.content
                self.s3.put_object(Bucket=self.bucket, Key=path_raw, Body=excel_bytes)
            else:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path_raw)
                excel_bytes = obj['Body'].read()

            xlsx_io = io.BytesIO(excel_bytes)
            dfs_acumulados = []
            
            # 🎯 ALINHAMENTO: Incluindo colunas vitais para a Matriz de Gravidade e Período
            colunas_alvo = [
                "MUNICIPIO", "CIDADE", "NOME_MUNICIPIO", "BAIRRO", "LOGRADOURO",
                "DATA_OCORRENCIA_BO", "DATA_OCORRENCIA", "HORA_OCORRENCIA_BO", 
                "DESC_PERIODO", # <--- ESSENCIAL para evitar o viés noturno
                "RUBRICA", "LATITUDE", "LONGITUDE", "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL"
            ]
            ancoras = {"LOGRADOURO", "MUNICIPIO", "RUBRICA", "LATITUDE", "DESC_PERIODO"}
            
            # Processamento das Abas (Geralmente 1 a 3 contêm dados)
            for i in range(1, 4):
                try:
                    xlsx_io.seek(0)
                    df_scan = pl.read_excel(xlsx_io, sheet_id=i, engine="calamine", has_header=False, read_options={"n_rows": 50})
                    
                    real_header_idx = None
                    for idx, row in enumerate(df_scan.iter_rows()):
                        row_values = [str(cell).upper().strip() for cell in row if cell is not None]
                        matches = [val for val in row_values if val in ancoras]
                        if len(matches) >= 3:
                            real_header_idx = idx
                            break
                    
                    if real_header_idx is not None:
                        xlsx_io.seek(0)
                        df = pl.read_excel(xlsx_io, sheet_id=i, engine="calamine", read_options={"skip_rows": real_header_idx})
                        df.columns = [c.upper().strip() for c in df.columns]
                        
                        cols_to_keep = [c for c in df.columns if c in colunas_alvo]
                        df = df.select(cols_to_keep)
                        df = df.with_columns(pl.all().cast(pl.String))
                        
                        # Limpeza de Capas e Rodapés
                        regex_capa = r"(?i)PRESENTE TABELA|FINALIDADE ESCLARECER|CAMPOS CONTIDOS|BASE DE DADOS"
                        df = df.filter(~pl.any_horizontal(pl.all().str.contains(regex_capa)))
                        df = df.filter(pl.any_horizontal(pl.all().is_not_null()))
                        
                        dfs_acumulados.append(df)
                        logger.info(f"BRONZE: [{ano}] Aba {i} lida ({df.height} registos).")
                except:
                    continue

            if dfs_acumulados:
                df_trusted = pl.concat(dfs_acumulados, how="diagonal")
                
                # Conversão de Coordenadas
                df_trusted = df_trusted.with_columns([
                    pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0),
                    pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0)
                ])

                # 🚀 OTIMIZAÇÃO: Geocodificação de Pontos Únicos
                logger.info(f"BRONZE: [{ano}] Vetorizando motor H3...")
                df_coords = df_trusted.select(["LATITUDE", "LONGITUDE"]).unique()
                
                coords_list = df_coords.to_dicts()
                for d in coords_list:
                    d["H3_INDEX"] = self._motor_h3(d["LATITUDE"], d["LONGITUDE"])
                
                df_h3_map = pl.DataFrame(coords_list, schema={"LATITUDE": pl.Float64, "LONGITUDE": pl.Float64, "H3_INDEX": pl.String})
                df_trusted = df_trusted.join(df_h3_map, on=["LATITUDE", "LONGITUDE"], how="left")

                # Escrita no Data Lake (Parquet LZ4)
                buffer = io.BytesIO()
                df_trusted.write_parquet(buffer, compression="lz4")
                self.s3.put_object(Bucket=self.bucket, Key=path_trusted, Body=buffer.getvalue())
                
                logger.info(f"BRONZE: [{ano}] Trusted guardada. Sucesso total.")
                return True
                
        except Exception as e:
            logger.error(f"BRONZE: Erro fatal no ano {ano}: {e}")
        return False

if __name__ == "__main__":
    bronze = IngestaoBronze()
    bronze.executar_ingestao_continua(force=True)
