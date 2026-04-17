import os
import boto3
import requests
import logging
import io
import polars as pl
import h3
from botocore.config import Config
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, List

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - [%(levelname)s] - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfiguracaoIngestao:
    NOME_BUCKET = os.getenv("R2_BUCKET_NAME", "").strip()
    URL_BASE_SSP = "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
    RESOLUCAO_H3 = 9
    
    COLUNAS_ALVO = [
        "MUNICIPIO", "CIDADE", "NOME_MUNICIPIO", "BAIRRO", "LOGRADOURO",
        "DATA_OCORRENCIA_BO", "DATA_OCORRENCIA", "HORA_OCORRENCIA_BO", 
        "DESC_PERIODO", "RUBRICA", "LATITUDE", "LONGITUDE", 
        "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL"
    ]
    ANCORAS_CABECALHO = {"LOGRADOURO", "MUNICIPIO", "RUBRICA", "LATITUDE", "DESC_PERIODO"}
    PADRAO_LIMPEZA = r"(?i)PRESENTE TABELA|FINALIDADE ESCLARECER|CAMPOS CONTIDOS|BASE DE DADOS"
    CABECALHOS_HTTP = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SafeDriverBot/1.0"}

class IngestaoBronze:
    def __init__(self):
        self.configuracao = ConfiguracaoIngestao()
        self.cliente_armazenamento = self._inicializar_cliente_armazenamento()
        self.sessao_http = self._inicializar_sessao_http()
        self.caminho_raiz = self._descobrir_raiz_datalake()

    def _inicializar_cliente_armazenamento(self):
        return boto3.client(
            's3', 
            endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/'),
            aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
            aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
            config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
        )

    def _inicializar_sessao_http(self):
        sessao = requests.Session()
        estrategia_tentativas = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adaptador = HTTPAdapter(max_retries=estrategia_tentativas)
        sessao.mount("http://", adaptador)
        sessao.mount("https://", adaptador)
        return sessao

    def _descobrir_raiz_datalake(self) -> str:
        try:
            resposta = self.cliente_armazenamento.list_objects_v2(Bucket=self.configuracao.NOME_BUCKET, MaxKeys=10)
            if 'Contents' in resposta:
                for objeto in resposta['Contents']:
                    if "datalake" in objeto['Key']: 
                        return objeto['Key'].split("datalake")[0] + "datalake"
            return "datalake"
        except Exception:
            return "datalake"

    def _obter_caminho(self, camada: str, subpasta: str, nome_arquivo: str) -> str:
        return f"{self.caminho_raiz}/{camada}/{subpasta}/{nome_arquivo}".replace("//", "/")

    def _calcular_indice_h3(self, latitude: float, longitude: float) -> Optional[str]:
        try:
            lat, lon = float(latitude), float(longitude)
            if lat == 0 or lon == 0 or abs(lat) > 90 or abs(lon) > 180:
                return None
            return h3.latlng_to_cell(lat, lon, self.configuracao.RESOLUCAO_H3)
        except Exception:
            return None

    def _verificar_tamanho_remoto(self, url: str) -> int:
        try:
            resposta = self.sessao_http.head(url, headers=self.configuracao.CABECALHOS_HTTP, timeout=20)
            return int(resposta.headers.get('Content-Length', 0)) if resposta.status_code == 200 else 0
        except Exception:
            return 0

    def _obter_tamanho_local(self, caminho_arquivo: str) -> int:
        try:
            metadados = self.cliente_armazenamento.head_object(Bucket=self.configuracao.NOME_BUCKET, Key=caminho_arquivo)
            return metadados['ContentLength']
        except Exception:
            return 0

    def _processar_planilha(self, bytes_excel: bytes, ano: int) -> Optional[pl.DataFrame]:
        fluxo_memoria = io.BytesIO(bytes_excel)
        lista_dataframes = []

        for aba in range(1, 4):
            try:
                fluxo_memoria.seek(0)
                dataframe_leitura = pl.read_excel(fluxo_memoria, sheet_id=aba, engine="calamine", has_header=False, read_options={"n_rows": 50})
                
                indice_cabecalho = None
                for indice, linha in enumerate(dataframe_leitura.iter_rows()):
                    valores_linha = [str(celula).upper().strip() for celula in linha if celula is not None]
                    correspondencias = [valor for valor in valores_linha if valor in self.configuracao.ANCORAS_CABECALHO]
                    if len(correspondencias) >= 3:
                        indice_cabecalho = indice
                        break
                
                if indice_cabecalho is not None:
                    fluxo_memoria.seek(0)
                    dataframe = pl.read_excel(fluxo_memoria, sheet_id=aba, engine="calamine", read_options={"skip_rows": indice_cabecalho})
                    dataframe.columns = [coluna.upper().strip() for coluna in dataframe.columns]
                    
                    colunas_presentes = [coluna for coluna in dataframe.columns if coluna in self.configuracao.COLUNAS_ALVO]
                    dataframe = dataframe.select(colunas_presentes).with_columns(pl.all().cast(pl.String))
                    
                    dataframe = dataframe.filter(~pl.any_horizontal(pl.all().str.contains(self.configuracao.PADRAO_LIMPEZA)))
                    dataframe = dataframe.filter(pl.any_horizontal(pl.all().is_not_null()))
                    
                    lista_dataframes.append(dataframe)
            except Exception:
                continue

        if not lista_dataframes:
            return None

        dataframe_consolidado = pl.concat(lista_dataframes, how="diagonal")
        
        dataframe_consolidado = dataframe_consolidado.with_columns([
            pl.col("LATITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0),
            pl.col("LONGITUDE").str.replace(",", ".").cast(pl.Float64, strict=False).fill_null(0.0)
        ])

        dados_coordenadas = dataframe_consolidado.select(["LATITUDE", "LONGITUDE"]).unique().to_dicts()
        for dicionario in dados_coordenadas:
            dicionario["H3_INDEX"] = self._calcular_indice_h3(dicionario["LATITUDE"], dicionario["LONGITUDE"])
        
        mapa_h3 = pl.DataFrame(dados_coordenadas, schema={"LATITUDE": pl.Float64, "LONGITUDE": pl.Float64, "H3_INDEX": pl.String})
        return dataframe_consolidado.join(mapa_h3, on=["LATITUDE", "LONGITUDE"], how="left")

    def _processar_ano(self, ano: int, forcar_execucao: bool = False) -> bool:
        caminho_bruto = self._obter_caminho("bronze", "raw", f"ssp_raw_{ano}.xlsx")
        caminho_confiavel = self._obter_caminho("bronze", "trusted", f"ssp_trusted_{ano}.parquet")
        url_origem = self.configuracao.URL_BASE_SSP.format(ano=ano)

        tamanho_remoto = self._verificar_tamanho_remoto(url_origem)
        tamanho_local = self._obter_tamanho_local(caminho_bruto)

        if (tamanho_remoto == tamanho_local) and not forcar_execucao and tamanho_remoto > 0:
            try:
                self.cliente_armazenamento.head_object(Bucket=self.configuracao.NOME_BUCKET, Key=caminho_confiavel)
                logger.info(f"[{ano}] Integridade confirmada. Processamento ignorado.")
                return False
            except Exception:
                pass

        try:
            if tamanho_remoto != tamanho_local or forcar_execucao or tamanho_local == 0:
                logger.info(f"[{ano}] Iniciando transferencia da fonte externa.")
                resposta_http = self.sessao_http.get(url_origem, headers=self.configuracao.CABECALHOS_HTTP, timeout=300)
                bytes_excel = resposta_http.content
                self.cliente_armazenamento.put_object(Bucket=self.configuracao.NOME_BUCKET, Key=caminho_bruto, Body=bytes_excel)
            else:
                objeto_armazenado = self.cliente_armazenamento.get_object(Bucket=self.configuracao.NOME_BUCKET, Key=caminho_bruto)
                bytes_excel = objeto_armazenado['Body'].read()

            dataframe_processado = self._processar_planilha(bytes_excel, ano)
            
            if dataframe_processado is not None:
                buffer = io.BytesIO()
                dataframe_processado.write_parquet(buffer, compression="lz4")
                self.cliente_armazenamento.put_object(Bucket=self.configuracao.NOME_BUCKET, Key=caminho_confiavel, Body=buffer.getvalue())
                logger.info(f"[{ano}] Dados refinados e armazenados com exito.")
                return True
                
        except Exception as erro:
            logger.error(f"Interrupcao no ciclo do ano {ano}: {erro}")
        return False

    def executar_ingestao_continua(self, forcar_execucao: bool = False) -> bool:
        logger.info(f"Iniciando varredura de dados.")
        houve_atualizacao = False
        ano_atual = datetime.now().year
        
        for ano in range(2022, ano_atual + 1):
            if self._processar_ano(ano, forcar_execucao):
                houve_atualizacao = True
                
        return houve_atualizacao

if __name__ == "__main__":
    ingestao = IngestaoBronze()
    ingestao.executar_ingestao_continua(forcar_execucao=True)
