import os
import boto3
import requests
import logging
import io
import polars as pl
import h3
import hashlib
from botocore.config import Config
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional

# Configuração de Log profissional
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfiguracaoIngestao:
    NOME_BUCKET = os.getenv("R2_BUCKET_NAME", "safedriver").strip()
    URL_BASE_SSP = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
    RESOLUCAO_H3 = 9
    
    # Chaves LGPD via GitHub Secrets (Cruciais para a segurança do Hash)
    PEPPER = os.getenv("LGPD_PEPPER", "default_pepper_key")
    SALT = os.getenv("LGPD_SALT", "default_salt_key")

    # Colunas necessárias para análise e anonimização
    COLUNAS_ALVO = [
        "NUM_BO", "MUNICIPIO", "BAIRRO", "LOGRADOURO",
        "DATA_OCORRENCIA_BO", "DATA_OCORRENCIA", "HORA_OCORRENCIA_BO", 
        "DESC_PERIODO", "RUBRICA", "LATITUDE", "LONGITUDE", 
        "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL"
    ]
    
    ANCORAS_CABECALHO = {"LOGRADOURO", "MUNICIPIO", "RUBRICA", "LATITUDE", "DESC_PERIODO"}
    PADRAO_LIMPEZA = r"(?i)PRESENTE TABELA|FINALIDADE ESCLARECER|CAMPOS CONTIDOS|BASE DE DADOS"

class IngestorSafeDriver:
    def __init__(self):
        self.config = ConfiguracaoIngestao()
        self.s3 = self._inicializar_s3()
        self.sessao = self._inicializar_http()

    def _inicializar_s3(self):
        return boto3.client(
            's3', 
            endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )

    def _inicializar_http(self):
        s = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=retries))
        return s

    def _gerar_hash_lgpd(self, logradouro: str, id_bo: str) -> str:
        """
        Gera um Hash Único por Registro: Pepper + Logradouro + ID_BO + Salt.
        Isso garante que dois crimes na mesma rua tenham hashes diferentes,
        protegendo contra ataques de correlação e cumprindo a LGPD.
        """
        if not logradouro or str(logradouro).upper() in ["NONE", "NULL", "NAN"]:
            return "LOCAL_ANONIMIZADO"
        
        # O 'tempero' (Salt) é o próprio ID do B.O. para garantir unicidade por linha
        payload = f"{self.config.PEPPER}{logradouro}{id_bo}{self.config.SALT}".encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def _get_tamanho_remoto(self, url):
        try:
            res = self.sessao.head(url, timeout=30)
            return int(res.headers.get('Content-Length', 0))
        except: return 0

    def _get_tamanho_local(self, key):
        try:
            res = self.s3.head_object(Bucket=self.config.NOME_BUCKET, Key=key)
            return int(res.get('ContentLength', 0))
        except: return 0

    def _calcular_h3(self, lat, lon):
        try:
            lat, lon = float(lat), float(lon)
            if lat == 0 or lon == 0 or abs(lat) > 90 or abs(lon) > 180: return None
            return h3.latlng_to_cell(lat, lon, self.config.RESOLUCAO_H3)
        except: return None

    def _processar_e_proteger(self, bytes_excel: bytes) -> Optional[pl.DataFrame]:
        """Processa o Excel e aplica a anonimização determinística por B.O."""
        fluxo = io.BytesIO(bytes_excel)
        dfs = []
        
        for aba in range(1, 4):
            try:
                preview = pl.read_excel(fluxo, sheet_id=aba, engine="calamine", has_header=False, read_options={"n_rows": 50})
                idx_header = None
                for i, linha in enumerate(preview.iter_rows()):
                    vals = [str(c).upper() for c in linha if c is not None]
                    if len([v for v in vals if v in self.config.ANCORAS_CABECALHO]) >= 3:
                        idx_header = i
                        break
                
                if idx_header is not None:
                    df = pl.read_excel(fluxo, sheet_id=aba, engine="calamine", read_options={"skip_rows": idx_header})
                    df.columns = [c.upper().strip() for c in df.columns]
                    cols = [c for c in df.columns if c in self.config.COLUNAS_ALVO]
                    df = df.select(cols).with_columns(pl.all().cast(pl.String))
                    df = df.filter(~pl.any_horizontal(pl.all().str.contains(self.config.PADRAO_LIMPEZA)))
                    dfs.append(df)
            except: continue

        if not dfs: return None
        
        df_final = pl.concat(dfs, how="diagonal")

        # 1. Tratamento de Coordenadas
        df_final = df_final.with_columns([
            pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0)
        ])

        # 2. Camada LGPD: Anonimização com Unicidade
        logger.info("🔒 Aplicando criptografia SHA-256 (Pepper + Logradouro + NUM_BO + Salt)")
        df_final = df_final.with_columns(
            pl.struct(["LOGRADOURO", "NUM_BO"]).map_elements(
                lambda x: self._gerar_hash_lgpd(x["LOGRADOURO"], x["NUM_BO"]),
                return_dtype=pl.String
            ).alias("HASH_LOCAL_UNICO")
        ).drop(["LOGRADOURO", "NUM_BO"]) # Remove os dados sensíveis originais imediatamente

        # 3. Enriquecimento Geoespacial (H3)
        # Calculamos H3 para as coordenadas únicas para otimizar performance
        geo_map = df_final.select(["LATITUDE", "LONGITUDE"]).unique().to_dicts()
        for d in geo_map:
            d["H3_INDEX"] = self._calcular_h3(d["LATITUDE"], d["LONGITUDE"])
        
        return df_final.join(pl.DataFrame(geo_map), on=["LATITUDE", "LONGITUDE"], how="left")

    def rodar_ano(self, ano: int):
        # O arquivo nasce na Bronze já protegido e em formato Parquet
        path_bronze_protegida = f"datalake/bronze/crimes_protected/ssp_protected_{ano}.parquet"
        url = self.config.URL_BASE_SSP.format(ano=ano)
        
        tamanho_remoto = self._get_tamanho_remoto(url)
        tamanho_local = self._get_tamanho_local(path_bronze_protegida)

        # Só baixa se o arquivo na SSP for maior que o que temos (novas ocorrências)
        if tamanho_remoto > 0 and tamanho_remoto <= tamanho_local:
            logger.info(f"[{ano}] Datalake atualizado ({tamanho_local/1024/1024:.2f} MB).")
            return

        logger.info(f"[{ano}] Novo conteúdo detectado. Iniciando processamento seguro...")
        try:
            res = self.sessao.get(url, timeout=400)
            if res.status_code == 200:
                # O processamento acontece em RAM. Nada sensível é escrito em disco antes do Hash.
                df_protegido = self._processar_e_proteger(res.content)
                
                if df_protegido is not None:
                    buffer = io.BytesIO()
                    df_protegido.write_parquet(buffer, compression="lz4")
                    
                    # Salva no R2 já anonimizado
                    self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_bronze_protegida, Body=buffer.getvalue())
                    logger.info(f"✅ [{ano}] Sucesso: Camada Bronze Protegida (Parquet) atualizada.")

        except Exception as e:
            logger.error(f"❌ Erro no ciclo do ano {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    ano_atual = datetime.now().year
    
    # Varredura completa de 2022 até o presente
    anos_ciclo = range(2022, ano_atual + 1)
    logger.info(f"🚀 SafeDriver Autobot Iniciado | Período: 2022 - {ano_atual}")
    
    for ano in anos_ciclo:
        ingestor.rodar_ano(ano)
