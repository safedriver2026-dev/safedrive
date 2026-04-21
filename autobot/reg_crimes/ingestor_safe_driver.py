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
from botocore.config import Config
from datetime import datetime

# Configuração de Log
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
    
    # Mapeamento Refinado com base nos arquivos de 2026
    MAPA_COLUNAS = {
        "NUM_BO": [r"NUM.*BO", r"N.MERO.*BO", r"BO_NUMERO"],
        "MUNICIPIO": [r"MUNIC.PIO", r"CIDADE", r"NM_MUN"],
        "BAIRRO": [r"BAIRRO", r"NM_BAIRRO"],
        "LOGRADOURO": [r"LOGRADOURO", r"RUA", r"DESCR_LOG", r"NM_LOG"],
        "DATAOCORRENCIA": [r"DATA.*OCORR", r"DT_OCORR", r"DATA_OCORRENCIA_BO"],
        "HORAOCORRENCIA": [r"HORA.*OCORR", r"HR_OCORR", r"HORA_OCORRENCIA_BO"],
        "RUBRICA": [r"RUBRICA", r"NATUREZA", r"DESCR_RUBRICA"],
        "LATITUDE": [r"LATITUDE", r"LAT.*GEO", r"^LAT$", r"COORDENADA_X"],
        "LONGITUDE": [r"LONGITUDE", r"LON.*GEO", r"^LON$", r"COORDENADA_Y"],
        "DESCR_TIPOLOCAL": [r"TIPOLOCAL", r"LOCAL_OCORR", r"DESCR_TIPOLOCAL"]
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

    def _resolver_mapeamento(self, colunas_da_linha):
        mapeamento = {}
        for alvo, padroes in self.config.MAPA_COLUNAS.items():
            for i, col_name in enumerate(colunas_da_linha):
                col_str = str(col_name).upper().strip()
                for p in padroes:
                    if re.search(p, col_str):
                        mapeamento[i] = alvo
                        break
                if i in mapeamento: break
        return mapeamento

    def _limpar_e_tipar(self, df: pl.DataFrame) -> pl.DataFrame:
        # 1. Tratamento de coordenadas (suporte a vírgula e strings sujas)
        df = df.with_columns([
            pl.col("LATITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False),
            pl.col("LONGITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False)
        ])
        # 2. Filtro de segurança (apenas pontos válidos em SP)
        return df.filter((pl.col("LATITUDE").is_not_null()) & (pl.col("LATITUDE") < -10))

    def extrair_bronze(self, ano: int):
        path = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        if ano < self.ano_atual and self._arquivo_existe(path):
            logger.info(f"⏭️ Bronze {ano} já existe.")
            return
        url = self.config.URL_BASE_SSP.format(ano=ano)
        try:
            res = requests.get(url, timeout=600)
            if res.status_code == 200:
                self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path, Body=res.content)
                logger.info(f"✅ Bronze {ano} salva.")
            else: logger.error(f"❌ Falha SSP {ano}: {res.status_code}")
        except Exception as e: logger.error(f"❌ Erro Download {ano}: {e}")

    def processar_prata(self, ano: int):
        path_b = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_p = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"

        if self._arquivo_existe(path_p):
            logger.info(f"⏭️ Prata {ano} já processada.")
            return

        try:
            logger.info(f"🥈 Iniciando Garimpo de Abas no arquivo de {ano}...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=path_b)
            excel_data = obj['Body'].read()
            
            # Usando fastexcel para listar abas (muito mais rápido para arquivos gigantes)
            import fastexcel
            excel_reader = fastexcel.read_xlsx(io.BytesIO(excel_data))
            abas_disponiveis = excel_reader.sheet_names
            
            list_dfs = []

            for nome_aba in abas_disponiveis:
                # Pula abas conhecidas de metadados para ganhar tempo
                if any(x in nome_aba.upper() for x in ["CAPA", "DICIONARIO", "LEGENDA", "CAMPOS"]):
                    continue

                try:
                    # Scan de cabeçalho na aba atual
                    df_scan = pl.read_excel(io.BytesIO(excel_data), sheet_name=nome_aba, engine="calamine", read_options={"has_header": False, "n_rows": 50})
                    
                    cabecalho_info = None
                    for i, row in enumerate(df_scan.iter_rows()):
                        map_idx = self._resolver_mapeamento(row)
                        if "LATITUDE" in map_idx.values() and "LONGITUDE" in map_idx.values():
                            cabecalho_info = (i, map_idx)
                            break
                    
                    if cabecalho_info:
                        skip, mapping = cabecalho_info
                        logger.info(f"   🎯 Aba '{nome_aba}': Dados encontrados na linha {skip+1}")
                        
                        df_mes = pl.read_excel(io.BytesIO(excel_data), sheet_name=nome_aba, engine="calamine", read_options={"skip_rows": skip})
                        
                        # Renomeia usando o mapeamento de índices para evitar erros de nomes duplicados
                        inverso = {df_mes.columns[idx]: alvo for idx, alvo in mapping.items()}
                        df_mes = df_mes.select(list(inverso.keys())).rename(inverso)
                        df_mes = df_mes.with_columns(pl.all().cast(pl.Utf8))
                        
                        list_dfs.append(df_mes)
                except Exception as e:
                    logger.warning(f"   ⚠️ Falha ao ler aba {nome_aba}: {e}")

            if not list_dfs:
                raise ValueError(f"Não foi possível extrair dados de nenhuma aba no arquivo de {ano}.")

            # Consolidação dos meses
            df_final = pl.concat(list_dfs, how="diagonal")
            df_final = self._limpar_e_tipar(df_final)
            
            # LGPD (Hash Seguro)
            pepper = self.config.PEPPER
            df_final = df_final.with_columns([
                pl.col("NUM_BO").map_elements(lambda v: hashlib.sha256(f"{v}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8),
                pl.col("LOGRADOURO").map_elements(lambda v: hashlib.sha256(f"{v}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8)
            ])

            # H3 Index
            res_h3 = self.config.RESOLUCAO_H3
            df_final = df_final.with_columns(
                pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                    lambda x: h3.latlng_to_cell(x["LATITUDE"], x["LONGITUDE"], res_h3),
                    return_dtype=pl.Utf8
                ).alias("H3_INDEX")
            )

            # Salvar como Parquet Único do Ano
            buf = io.BytesIO()
            df_final.write_parquet(buf, compression="zstd")
            self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_p, Body=buf.getvalue())
            logger.info(f"✨ Prata {ano} finalizada com SUCESSO: {df_final.height} crimes consolidados (todas as abas).")

        except Exception as e:
            logger.error(f"💥 Erro fatal no processamento de {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    for ano in range(2022, ingestor.ano_atual + 1):
        if modo in ["bronze", "tudo"]: ingestor.extrair_bronze(ano)
        if modo in ["prata", "tudo"]: ingestor.processar_prata(ano)
