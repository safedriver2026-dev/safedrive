import os
import boto3
import requests
import logging
import time
import sys
import io
import hashlib
import re
import botocore.exceptions
import polars as pl
import h3
import fastexcel
from botocore.config import Config
from datetime import datetime

# Configuração de Log Profissional
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger(__name__)

class ConfiguracaoIngestao:
    NOME_BUCKET = os.getenv("R2_BUCKET_NAME")
    ENDPOINT_URL = os.getenv("R2_ENDPOINT_URL")
    ACCESS_KEY = os.getenv("R2_ACCESS_KEY_ID")
    SECRET_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
    PEPPER = os.getenv("LGPD_PEPPER", "safedriver_secret_2026")

    URL_BASE_SSP = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
    RESOLUCAO_H3 = 9
    
    # Mapeamento Ultra Flexível (RegEx) - Adicionado variações comuns de 2022/2023
    MAPA_COLUNAS = {
        "NUM_BO": [r"NUM.*BO", r"N.MERO.*BO", r"BO_NUMERO"],
        "MUNICIPIO": [r"MUNIC.PIO", r"CIDADE", r"NM_MUN", r"NOME_MUNICIPIO"],
        "BAIRRO": [r"BAIRRO", r"NM_BAIRRO"],
        "LOGRADOURO": [r"LOGRADOURO", r"RUA", r"DESCR_LOG", r"NM_LOG", r"ENDERECO"],
        "DATAOCORRENCIA": [r"DATA.*OCORR", r"DT_OCORR", r"DATA_OCORRENCIA"],
        "HORAOCORRENCIA": [r"HORA.*OCORR", r"HR_OCORR", r"HORA_OCORRENCIA"],
        "RUBRICA": [r"RUBRICA", r"NATUREZA", r"DESCR_RUBRICA"],
        "LATITUDE": [r"LATITUDE", r"LAT.*GEO", r"^LAT$", r"COORDENADA_X", r"LATITUD"],
        "LONGITUDE": [r"LONGITUDE", r"LON.*GEO", r"^LON$", r"COORDENADA_Y", r"LONGITUD"],
        "DESCR_TIPOLOCAL": [r"TIPOLOCAL", r"LOCAL_OCORR", r"DESCR_TIPOLOCAL", r"LOCAL"]
    }

class IngestorSafeDriver:
    def __init__(self):
        self.config = ConfiguracaoIngestao()
        self.s3 = self._inicializar_s3()
        self.ano_atual = datetime.now().year

    def _inicializar_s3(self):
        endpoint = self.config.ENDPOINT_URL.strip().rstrip('/')
        return boto3.client(
            's3',
            endpoint_url=endpoint,
            aws_access_key_id=self.config.ACCESS_KEY.strip(),
            aws_secret_access_key=self.config.SECRET_KEY.strip(),
            config=Config(signature_version='s3v4')
        )

    def _arquivo_existe(self, key: str) -> bool:
        try:
            self.s3.head_object(Bucket=self.config.NOME_BUCKET, Key=key)
            return True
        except: return False

    def _resolver_mapeamento(self, lista_colunas):
        """Mapeia os nomes originais encontrados para os nomes padrão do SafeDriver."""
        mapeamento = {}
        for alvo, padroes in self.config.MAPA_COLUNAS.items():
            for i, nome_col in enumerate(lista_colunas):
                col_limpa = str(nome_col).upper().strip()
                for p in padroes:
                    if re.search(p, col_limpa):
                        mapeamento[nome_col] = alvo
                        break
                if nome_col in mapeamento: break
        return mapeamento

    def _limpar_e_tipar(self, df: pl.DataFrame) -> pl.DataFrame:
        # Tratamento de coordenadas e filtro de SP (Coordenadas negativas típicas de SP)
        df = df.with_columns([
            pl.col("LATITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False),
            pl.col("LONGITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False)
        ])
        # Filtramos apenas o que tem GPS real e está na região de SP (Lat < -10)
        return df.filter((pl.col("LATITUDE").is_not_null()) & (pl.col("LATITUDE") < -10))

    def extrair_bronze(self, ano: int):
        path = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        if ano < self.ano_atual and self._arquivo_existe(path):
            logger.info(f"⏭️ Bronze {ano} já existe no R2.")
            return
        url = self.config.URL_BASE_SSP.format(ano=ano)
        try:
            res = requests.get(url, timeout=600)
            if res.status_code == 200:
                self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path, Body=res.content)
                logger.info(f"✅ Bronze {ano} salva com sucesso.")
            else: logger.error(f"❌ Erro SSP {ano}: Status {res.status_code}")
        except Exception as e: logger.error(f"❌ Falha no Download {ano}: {e}")

    def processar_prata(self, ano: int):
        path_b = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_p = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"

        if self._arquivo_existe(path_p):
            logger.info(f"⏭️ Prata {ano} já existe no R2. Pulando.")
            return

        try:
            logger.info(f"🥈 [Ano {ano}] Iniciando Garimpo de Abas...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=path_b)
            excel_bytes = obj['Body'].read()
            
            # FastExcel para listar abas
            excel_reader = fastexcel.read_excel(excel_bytes)
            abas = excel_reader.sheet_names
            logger.info(f"   📂 Abas detectadas: {abas}")
            
            list_dfs = []

            for nome_aba in abas:
                # Pula abas irrelevantes de metadados
                if any(x in nome_aba.upper() for x in ["CAPA", "DICIONARIO", "LEGENDA", "CAMPOS", "SPDADOS"]):
                    continue

                try:
                    # Scan profundo: Lemos as primeiras 50 linhas para achar o cabeçalho
                    df_scan = pl.read_excel(excel_bytes, sheet_name=nome_aba, engine="calamine", read_options={"has_header": False, "n_rows": 50})
                    
                    row_header_idx = None
                    mapping_found = {}

                    for i, row in enumerate(df_scan.iter_rows()):
                        row_values = [str(v).strip() for v in row if v is not None]
                        mapping_found = self._resolver_mapeamento(row_values)
                        
                        # Se acharmos Latitude e Longitude, Bingo!
                        if "LATITUDE" in mapping_found.values() and "LONGITUDE" in mapping_found.values():
                            row_header_idx = i
                            break
                    
                    if row_header_idx is not None:
                        logger.info(f"   🎯 Sucesso na Aba '{nome_aba}' (Linha {row_header_idx + 1})")
                        
                        # Lê a aba pulando o lixo do topo
                        df_mes = pl.read_excel(excel_bytes, sheet_name=nome_aba, engine="calamine", read_options={"skip_rows": row_header_idx})
                        
                        # Mapeamento final com colunas reais
                        map_final = self._resolver_mapeamento(df_mes.columns)
                        
                        if "LATITUDE" in map_final.values():
                            df_mes = df_mes.select(list(map_final.keys())).rename(map_final)
                            df_mes = df_mes.with_columns(pl.all().cast(pl.Utf8))
                            list_dfs.append(df_mes)
                    else:
                        logger.warning(f"   ❓ Aba '{nome_aba}' não parece conter dados de GPS nas primeiras 50 linhas.")

                except Exception as e:
                    logger.warning(f"   ⚠️ Erro ao processar aba '{nome_aba}': {e}")

            if not list_dfs:
                # Debug final: Se falhar em tudo, mostra o que tinha na primeira aba
                logger.error(f"💥 Falha total no ano {ano}. Verificando cabeçalhos da primeira aba para diagnóstico:")
                primeira_aba = pl.read_excel(excel_bytes, sheet_name=abas[0], engine="calamine", read_options={"n_rows": 5})
                logger.info(f"   Headers da primeira aba: {primeira_aba.columns}")
                raise ValueError(f"Dados de crimes não localizados no arquivo de {ano}.")

            # Concatenação e Limpeza
            df_final = pl.concat(list_dfs, how="diagonal")
            df_final = self._limpar_e_tipar(df_final)
            
            # LGPD e Geoprocessamento (H3)
            pepper = self.config.PEPPER
            res_h3 = self.config.RESOLUCAO_H3
            
            df_final = df_final.with_columns([
                pl.col("NUM_BO").map_elements(lambda v: hashlib.sha256(f"{v}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8),
                pl.col("LOGRADOURO").map_elements(lambda v: hashlib.sha256(f"{v}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8),
                pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                    lambda x: h3.latlng_to_cell(x["LATITUDE"], x["LONGITUDE"], res_h3),
                    return_dtype=pl.Utf8
                ).alias("H3_INDEX")
            ])

            # Upload final em Parquet
            buf = io.BytesIO()
            df_final.write_parquet(buf, compression="zstd")
            self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_p, Body=buf.getvalue())
            logger.info(f"✨ Prata {ano} finalizada com {df_final.height} registros.")

        except Exception as e:
            logger.error(f"💥 Erro fatal no processamento de {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    for ano in range(2022, ingestor.ano_atual + 1):
        if modo in ["bronze", "tudo"]: ingestor.extrair_bronze(ano)
        if modo in ["prata", "tudo"]: ingestor.processar_prata(ano)
