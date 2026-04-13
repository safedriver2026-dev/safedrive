import os
import pandas as pd
import boto3
import traceback
import io
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from datetime import datetime
from .comunicador import RoboComunicador

class IngestaoBruta:
    def __init__(self, robo: RoboComunicador):
        self.robo = robo
        self.s3 = boto3.client('s3',
                                endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                                aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                                aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))
        self.bucket = os.environ.get("R2_BUCKET_NAME")

        self.sessao = requests.Session()
        tentativas = Retry(total=5, backoff_factor=3, status_forcelist=[500, 502, 503, 504])
        self.sessao.mount('https://', HTTPAdapter(max_retries=tentativas))

        self.ESQUEMA_ALVO = {
            'MUNICIPIO': ['NOME_MUNICIPIO', 'CIDADE', 'MUNICIPIO', 'NOME_MUNICIPIO_CIRCUNSCRICAO'],
            'BAIRRO': ['BAIRRO', 'BAIRRO_OCORRENCIA', 'NOME_BAIRRO'],
            'LOGRADOURO': ['LOGRADOURO', 'ENDERECO', 'DESC_LOGRADOURO'],
            'LATITUDE': ['LATITUDE', 'LAT'],
            'LONGITUDE': ['LONGITUDE', 'LONG'],
            'DATA_OCORRENCIA': ['DATA_OCORRENCIA_BO', 'DT_OCORRENCIA', 'DATA_FATO'],
            'HORA_OCORRENCIA': ['HORA_OCORRENCIA_BO', 'HORA_FATO'],
            'PERIODO': ['DESC_PERIODO', 'PERIODO_OCORRENCIA', 'PERIODO'],
            'NATUREZA': ['NATUREZA_APURADA', 'RUBRICA', 'DESCR_LEI']
        }

    def criar_mapeamento_sinonimos(self, df_metadados):
        mapeamento = {}
        if 'Campos' not in df_metadados.columns:
            return mapeamento
            
        colunas_origem = df_metadados['Campos'].dropna().unique()
        for coluna in colunas_origem:
            termo = str(coluna).strip().upper()
            for canonico, sinonimos in self.ESQUEMA_ALVO.items():
                if termo in [s.upper() for s in sinonimos]:
                    mapeamento[coluna] = canonico
                    break
        return mapeamento

    def baixar_arquivo(self, url):
        fluxo = io.BytesIO()
        with self.sessao.get(url, stream=True, timeout=900) as r:
            r.raise_for_status()
            for pedaco in r.iter_content(chunk_size=2097152):
                if pedaco:
                    fluxo.write(pedaco)
        fluxo.seek(0)
        return fluxo

    def processar_ano(self, ano):
        url = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano}.xlsx"
        caminho_destino = f"datalake/bruta/ssp_{ano}_bronze.parquet"
        
        try:
            self.robo.enviar_relatorio_operacional(f"Iniciando captura bruta {ano}")
            
            conteudo_binario = self.baixar_arquivo(url)
            leitor_excel = pd.ExcelFile(conteudo_binario)
            
            abas = leitor_excel.sheet_names
            aba_meta = [a for a in abas if 'CAMPOS' in a.upper()][0]
            aba_dados = [a for a in abas if any(mes in a.upper() for mes in ['JAN', 'FEV', 'MAR', 'DADOS']) or str(ano) in a][0]

            df_meta = pd.read_excel(leitor_excel, sheet_name=aba_meta, skiprows=3)
            df_origem = pd.read_excel(leitor_excel, sheet_name=aba_dados)

            dicionario_traducao = self.criar_mapeamento_sinonimos(df_meta)
            df_normalizado = df_origem.rename(columns=dicionario_traducao)

            for col in self.ESQUEMA_ALVO.keys():
                if col not in df_normalizado.columns:
                    df_normalizado[col] = None

            campos_obrigatorios = list(self.ESQUEMA_ALVO.keys()) + ['NUM_BO', 'ANO_BO']
            colunas_finais = [c for c in campos_obrigatorios if c in df_normalizado.columns]
            df_final = df_normalizado[colunas_finais].copy()

            if 'COD IBGE' in df_final.columns:
                df_final = df_final.drop(columns=['COD IBGE'])

            saida_parquet = io.BytesIO()
            df_final.to_parquet(saida_parquet, index=False)
            self.s3.put_object(Bucket=self.bucket, Key=caminho_destino, Body=saida_parquet.getvalue())

            self.robo.enviar_relatorio_operacional(f"Finalizado Bruta {ano}", {"Registros": len(df_final)})
            return True
            
        except Exception:
            self.robo.enviar_alerta_tecnico(f"Erro na Bruta {ano}", traceback.format_exc())
            return False

def iniciar_ingestao_total(robo):
    ingestor = IngestaoBruta(robo)
    ano_limite = datetime.now().year
    for ano in range(2022, ano_limite + 1):
        ingestor.processar_ano(ano)
