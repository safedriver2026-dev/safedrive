import os, boto3, requests, logging, io
import polars as pl
import h3
from botocore.config import Config
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
            resp = requests.head(url, headers=self.headers, timeout=15)
            return int(resp.headers.get('Content-Length', 0)) if resp.status_code == 200 else 0
        except:
            return 0

    def executar_ingestao_continua(self, force=False):
        logger.info(f"BRONZE: Iniciando purificação Trusted (Force={force}).")
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

        if (tamanho_remoto == tamanho_local) and not force:
            try:
                self.s3.head_object(Bucket=self.bucket, Key=path_trusted)
                logger.info(f"BRONZE: [{ano}] Cache validado. Pulando.")
                return False
            except: pass

        try:
            if tamanho_remoto != tamanho_local:
                logger.info(f"BRONZE: [{ano}] Baixando novo arquivo (Delta detected)...")
                resp = requests.get(url, headers=self.headers, timeout=300)
                excel_bytes = resp.content
                self.s3.put_object(Bucket=self.bucket, Key=path_raw, Body=excel_bytes)
            else:
                obj = self.s3.get_object(Bucket=self.bucket, Key=path_raw)
                excel_bytes = obj['Body'].read()

            xlsx_io = io.BytesIO(excel_bytes)
            dfs = []
            
            # --- COLUNAS QUE REALMENTE IMPORTAM PARA O SAFEDRIVER ---
            # Ao ler apenas estas, economizamos ~60% de RAM
            colunas_alvo = [
                "MUNICIPIO", "CIDADE", "NOME_MUNICIPIO", "BAIRRO", "LOGRADOURO",
                "DATA_OCORRENCIA_BO", "DATA_OCORRENCIA", "HORA_OCORRENCIA_BO", 
                "RUBRICA", "LATITUDE", "LONGITUDE", "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL"
            ]
            
            # Buscas estruturais (âncoras) para achar o cabeçalho real
            ancoras = {"LOGRADOURO", "MUNICIPIO", "RUBRICA", "LATITUDE"}
            
            for i in range(1, 4): # Otimização semestral (1º sem, 2º sem e reserva)
                try:
                    df_scan = pl.read_excel(xlsx_io, sheet_id=i, engine="calamine", has_header=False, read_options={"n_rows": 50})
                    
                    real_header_idx = None
                    for idx, row in enumerate(df_scan.iter_rows()):
                        row_values = [str(cell).upper().strip() for cell in row if cell is not None]
                        matches = [val for val in row_values if val in ancoras]
                        if len(matches) >= 3:
                            real_header_idx = idx
                            break
                    
                    if real_header_idx is not None:
                        # Lemos a aba partindo do cabeçalho real
                        df = pl.read_excel(xlsx_io, sheet_id=i, engine="calamine", read_options={"skip_rows": real_header_idx})
                        df.columns = [c.upper().strip() for c in df.columns]
                        
                        # --- FILTRO NUCLEAR DE COLUNAS ---
                        # Mantemos apenas o que a IA vai usar. O resto é descartado na hora.
                        cols_to_keep = [c for c in df.columns if c in colunas_alvo or c == "H3_INDEX"]
                        df = df.select(cols_to_keep)
                        
                        # Converte tudo para String (Regra da Bronze)
                        df = df.with_columns(pl.all().cast(pl.String))
                        
                        # Filtro Nuclear de Capa/Lixo
                        regex_capa = r"(?i)PRESENTE TABELA|FINALIDADE ESCLARECER|CAMPOS CONTIDOS|BASE DE DADOS"
                        df = df.filter(~pl.any_horizontal(pl.all().str.contains(regex_capa)))
                        df = df.filter(pl.any_horizontal(pl.all().is_not_null()))
                        
                        dfs.append(df)
                except: continue

            if dfs:
                # Concatenação Diagonal Segura (trata nomes de colunas variando entre anos)
                df_trusted = pl.concat(dfs, how="diagonal")

                logger.info(f"BRONZE: [{ano}] Calculando H9 para {df_trusted.height} registros...")
                
                # Geocodificação H3 interna
                df_trusted = df_trusted.with_columns([
                    pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0),
                    pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0)
                ]).with_columns(
                    pl.struct(["LATITUDE", "LONGITUDE"])
                    .map_elements(lambda x: self._motor_h3(x["LATITUDE"], x["LONGITUDE"]), return_dtype=pl.String)
                    .alias("H3_INDEX")
                )

                # Persistência final em Parquet (Compactado com LZ4 para velocidade)
                buffer = io.BytesIO()
                df_trusted.write_parquet(buffer, compression="lz4")
                self.s3.put_object(Bucket=self.bucket, Key=path_trusted, Body=buffer.getvalue())
                
                logger.info(f"BRONZE: [{ano}] Sincronizada com sucesso.")
                return True
                
        except Exception as e:
            logger.error(f"BRONZE: Erro no ano {ano}: {e}")
        return False
