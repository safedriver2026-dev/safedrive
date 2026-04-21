import os
import boto3
import requests
import logging
import time
import sys
import io
import hashlib
import re
import unicodedata
import botocore.exceptions
import polars as pl
import h3
import fastexcel
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
    
    MAPA_COLUNAS = {
        "NUM_BO": [r"NUM.*BO", r"N.MERO.*BO", r"BO_NUMERO"],
        "MUNICIPIO": [r"MUNIC.PIO", r"CIDADE", r"NM_MUN", r"NOME_MUN"],
        "BAIRRO": [r"BAIRRO", r"NM_BAIRRO"],
        "LOGRADOURO": [r"LOGRADOURO", r"RUA", r"DESCR_LOG", r"NM_LOG", r"ENDERECO"],
        "DATAOCORRENCIA": [r"DATA.*OCORR", r"DT_OCORR", r"DATA_OCORRENCIA"],
        "HORAOCORRENCIA": [r"HORA.*OCORR", r"HR_OCORR", r"HORA_OCORRENCIA"],
        "RUBRICA": [r"RUBRICA", r"NATUREZA", r"DESCR_RUBRICA"],
        "LATITUDE": [r"LATITUDE", r"LAT.*GEO", r"^LAT$", r"COORDENADA_X", r"LATITUD"],
        "LONGITUDE": [r"LONGITUDE", r"LON.*GEO", r"^LON$", r"COORDENADA_Y", r"LONGITUD"],
        "DESCR_TIPOLOCAL": [r"TIPOLOCAL", r"LOCAL_OCORR", r"DESCR_TIPOLOCAL", r"LOCAL"],
        # CALLBACK DE PERÍODO NATIVO (Resiliente a nomes de colunas diferentes)
        "PERIODO_NATIVO": [r"DESC.*PERIODO", r"PERIODO", r"DS_PERIODO", r"DESC_PERIODO_OCORRENCIA"]
    }

class IngestorSafeDriver:
    def __init__(self):
        self.config = ConfiguracaoIngestao()
        self.s3 = self._inicializar_s3()
        self.ano_atual = datetime.now().year

    def _inicializar_s3(self):
        endpoint = self.config.ENDPOINT_URL.strip().rstrip('/')
        if endpoint.endswith(f"/{self.config.NOME_BUCKET}"):
            endpoint = endpoint[: -len(f"/{self.config.NOME_BUCKET}")]
            
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

    def _normalizar_texto(self, valor):
        if valor is None or str(valor).upper() in ["NULL", "NAN", ".", "", "NONE"]: 
            return "NAO INFORMADO"
        texto = "".join(c for c in unicodedata.normalize('NFKD', str(valor)) if unicodedata.category(c) != 'Mn')
        return re.sub(r'[^a-zA-Z0-9\s]', '', texto).upper().strip()

    def _resolver_mapeamento(self, lista_colunas):
        mapeamento = {}
        for i, nome_col in enumerate(lista_colunas):
            if nome_col is None: continue
            col_limpa = "".join(c for c in str(nome_col).upper() if c.isalnum() or c == '_')
            for alvo, padroes in self.config.MAPA_COLUNAS.items():
                if alvo in mapeamento.values(): continue
                for p in padroes:
                    if re.search(p, col_limpa, re.IGNORECASE):
                        mapeamento[i] = alvo
                        break
        return mapeamento

    def _processar_periodo_mastigado(self, df: pl.DataFrame) -> pl.DataFrame:
        """Aplica a hierarquia de recuperação do período do dia."""
        
        # 1. Tentar extrair a hora e criar o período base
        df = df.with_columns(
            pl.col("HORAOCORRENCIA").str.to_time(format="%H:%M:%S", strict=False).dt.hour().alias("_hora_temp")
        )
        
        # 2. Lógica de Fallback
        df = df.with_columns(
            # Tenta pela Hora
            pl.when(pl.col("_hora_temp").is_between(0, 5)).then(pl.lit("MADRUGADA"))
            .when(pl.col("_hora_temp").is_between(6, 11)).then(pl.lit("MANHA"))
            .when(pl.col("_hora_temp").is_between(12, 17)).then(pl.lit("TARDE"))
            .when(pl.col("_hora_temp").is_between(18, 23)).then(pl.lit("NOITE"))
            # Se hora for nula, tenta pelo texto nativo da SSP
            .when(pl.col("PERIODO_NATIVO").str.to_uppercase().str.contains("MADRUGADA")).then(pl.lit("MADRUGADA"))
            .when(pl.col("PERIODO_NATIVO").str.to_uppercase().str.contains("MANHA|MANHÃ")).then(pl.lit("MANHA"))
            .when(pl.col("PERIODO_NATIVO").str.to_uppercase().str.contains("TARDE")).then(pl.lit("TARDE"))
            .when(pl.col("PERIODO_NATIVO").str.to_uppercase().str.contains("NOITE")).then(pl.lit("NOITE"))
            .otherwise(pl.lit(None))
            .alias("SAZON_PERIODO")
        ).drop(["_hora_temp", "PERIODO_NATIVO"]) # Limpa colunas temporárias
        
        # 3. EXCLUSÃO: Se não recuperou por nada, deleta a linha
        return df.filter(pl.col("SAZON_PERIODO").is_not_null())

    def _limpar_e_tipar(self, df: pl.DataFrame) -> pl.DataFrame:
        # 1. Normalização de Texto Universal
        colunas_texto = [c for c in df.columns if c not in ["LATITUDE", "LONGITUDE", "NUM_BO", "HORAOCORRENCIA", "PERIODO_NATIVO"]]
        for col in colunas_texto:
            df = df.with_columns(pl.col(col).map_elements(self._normalizar_texto, return_dtype=pl.Utf8).alias(col))

        # 2. Processar Período (Hierarquia de Recuperação)
        df = self._processar_periodo_mastigado(df)

        # 3. LAT/LON para Float e Filtro Geográfico
        df = df.with_columns([
            pl.col("LATITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False),
            pl.col("LONGITUDE").cast(pl.Utf8).str.replace(",", ".").str.extract(r"(-?\d+\.\d+)").cast(pl.Float64, strict=False)
        ])
        return df.filter((pl.col("LATITUDE").is_not_null()) & (pl.col("LATITUDE") < -10))

    def extrair_bronze(self, ano: int):
        path = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        if ano < self.ano_atual and self._arquivo_existe(path): return
        url = self.config.URL_BASE_SSP.format(ano=ano)
        try:
            res = requests.get(url, timeout=600)
            if res.status_code == 200:
                self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path, Body=res.content)
                logger.info(f"✅ Bronze {ano} salva.")
        except Exception as e: logger.error(f"❌ Erro Download {ano}: {e}")

    def processar_prata(self, ano: int):
        path_b = f"datalake/bronze/crimes_raw/ssp_raw_{ano}.xlsx"
        path_p = f"datalake/prata/crimes_trusted/ssp_trusted_{ano}.parquet"

        if self._arquivo_existe(path_p):
            logger.info(f"⏭️ Prata {ano} já pronta.")
            return

        try:
            logger.info(f"🥈 [Ano {ano}] Iniciando Normalização Universal...")
            obj = self.s3.get_object(Bucket=self.config.NOME_BUCKET, Key=path_b)
            excel_bytes = obj['Body'].read()
            
            excel_reader = fastexcel.read_excel(excel_bytes)
            abas = [n for n in excel_reader.sheet_names if not any(x in n.upper() for x in ["CAPA", "DICIONARIO", "LEGENDA", "CAMPOS", "SPDADOS"])]
            list_dfs = []

            for nome_aba in abas:
                try:
                    df_raw = pl.read_excel(excel_bytes, sheet_name=nome_aba, engine="calamine")
                    indices_map = self._resolver_mapeamento(df_raw.columns)
                    
                    # Lógica de salto de linha se cabeçalho não estiver na linha 0
                    if "LATITUDE" not in indices_map.values():
                        for row_idx in range(min(30, df_raw.height)):
                            indices_map = self._resolver_mapeamento(df_raw.row(row_idx))
                            if "LATITUDE" in indices_map.values():
                                df_raw = pl.read_excel(excel_bytes, sheet_name=nome_aba, engine="calamine", read_options={"skip_rows": row_idx + 1})
                                indices_map = self._resolver_mapeamento(df_raw.columns)
                                break
                    
                    df_mes = df_raw.select([df_raw.columns[i] for i in indices_map.keys()])
                    df_mes.columns = [indices_map[i] for i in indices_map.keys()]
                    df_mes = df_mes.with_columns(pl.all().cast(pl.Utf8))
                    list_dfs.append(df_mes)
                except: continue

            if not list_dfs: raise ValueError(f"Dados não localizados em {ano}.")
            df_final = pl.concat(list_dfs, how="diagonal")
            
            # Limpeza, Tipagem e Hierarquia Temporal
            df_final = self._limpar_e_tipar(df_final)
            
            # LGPD e H3
            pepper = self.config.PEPPER
            df_final = df_final.with_columns([
                pl.col("NUM_BO").map_elements(lambda v: hashlib.sha256(f"{str(v).upper()}{pepper}".encode()).hexdigest(), return_dtype=pl.Utf8),
                pl.struct(["LATITUDE", "LONGITUDE"]).map_elements(
                    lambda x: h3.latlng_to_cell(x["LATITUDE"], x["LONGITUDE"], self.config.RESOLUCAO_H3),
                    return_dtype=pl.Utf8
                ).alias("H3_INDEX")
            ])

            buf = io.BytesIO()
            df_final.write_parquet(buf, compression="zstd")
            self.s3.put_object(Bucket=self.config.NOME_BUCKET, Key=path_p, Body=buf.getvalue())
            logger.info(f"✨ Prata {ano} normalizada e 'mastigada' com SUCESSO.")

        except Exception as e:
            logger.error(f"💥 Erro fatal em {ano}: {e}")

if __name__ == "__main__":
    ingestor = IngestorSafeDriver()
    modo = sys.argv[1].lower() if len(sys.argv) > 1 else "tudo"
    for ano in range(2022, ingestor.ano_atual + 1):
        if modo in ["bronze", "tudo"]: ingestor.extrair_bronze(ano)
        if modo in ["prata", "tudo"]: ingestor.processar_prata(ano)
