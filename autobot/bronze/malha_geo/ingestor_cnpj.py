import os
import requests
import polars as pl
import boto3
import time
import hashlib
from datetime import datetime

class IngestorCnpjBronze:
    def __init__(self):
        # 1. Configuracoes de Seguranca (LGPD)
        self.salt = os.getenv('LGPD_SALT', 'default_salt')
        self.pepper = os.getenv('LGPD_PEPPER', 'default_pepper')
        
        # 2. Configuracoes do R2 Cloudflare
        self.s3_client = boto3.client('s3',
            region_name='auto',
            endpoint_url=os.getenv('R2_ENDPOINT_URL', '').strip(),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID', '').strip(),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY', '').strip()
        )
        self.bucket_name = os.getenv('R2_BUCKET_NAME', '').strip()
        
        # 3. Caminhos de Arquivo
        self.r2_path = "datalake/bronze/malha_raw/CNPJ_SP_PROTEGIDO.parquet"
        self.local_temp_path = "temp_bronze.parquet"
        self.input_csv_path = "data_raw/cnpjs_input.csv"
        
        self.br_api_url = "https://brasilapi.com.br/api/cep/v2/"

    def classificar_impacto_urbano(self, cnae_principal):
        """Inteligencia de classificacao urbana baseada nos primeiros digitos do CNAE"""
        cnae_str = str(cnae_principal).strip()
        prefixo = cnae_str[:2]
        
        if prefixo in ["47", "56", "96"]: 
            return "VAREJO_COMERCIO"
        if prefixo in ["49", "52", "10", "11"]: 
            return "INDUSTRIA_GALPAO"
        if prefixo in ["84"]:
            return "SEGURANCA_DELEGACIA"
            
        return "OUTROS"

    def capturar_coordenadas(self, cep):
        """Busca Lat/Lon do CEP via BrasilAPI"""
        cep_limpo = "".join(filter(str.isdigit, str(cep)))
        if not cep_limpo:
            return None, None
            
        try:
            response = requests.get(f"{self.br_api_url}{cep_limpo}", timeout=10)
            if response.status_code == 200:
                coords = response.json().get('location', {}).get('coordinates', {})
                return coords.get('latitude'), coords.get('longitude')
        except Exception as e:
            print(f"   [Erro API] Falha ao buscar CEP {cep_limpo}: {e}")
        return None, None

    def gerar_hash_lgpd(self, cnpj):
        """Gera um Hash SHA-256 consistente e seguro para o CNPJ"""
        cnpj_limpo = "".join(filter(str.isdigit, str(cnpj)))
        base_string = f"{cnpj_limpo}{self.salt}{self.pepper}"
        return hashlib.sha256(base_string.encode('utf-8')).hexdigest()

    def carregar_dados_entrada(self):
        """Tenta ler um CSV real; se falhar, usa os dados de fallback/teste"""
        if os.path.exists(self.input_csv_path):
            print(f"Lendo base de entrada: {self.input_csv_path}")
            try:
                df = pl.read_csv(self.input_csv_path)
                return df.to_dicts()
            except Exception as e:
                print(f"Erro ao ler CSV: {e}. Abortando.")
                return []
        else:
            print(f"Arquivo {self.input_csv_path} nao encontrado. Rodando base de teste padrao.")
            return [
                {"cnpj": "12345678000199", "cep": "09725000", "cnae": "47113"},
                {"cnpj": "98765432000100", "cep": "09850550", "cnae": "52117"}
            ]

    def processar_lista(self, lista_estabelecimentos):
        """Processa a lista, aplica LGPD e gera o Parquet"""
        if not lista_estabelecimentos:
            print("Nenhum dado para processar.")
            return

        print(f"Iniciando processamento de {len(lista_estabelecimentos)} registros...")
        dados_processados = []

        for item in lista_estabelecimentos:
            cnpj = item.get('cnpj', '')
            cep = item.get('cep', '')
            cnae = item.get('cnae', '')

            lat, lon = self.capturar_coordenadas(cep)
            
            if lat and lon:
                hash_protegido = self.gerar_hash_lgpd(cnpj)
                categoria = self.classificar_impacto_urbano(cnae)

                dados_processados.append({
                    "HASH_IDENTIFICADOR": hash_protegido,
                    "CATEGORIA": categoria,
                    "LAT": lat,
                    "LON": lon
                })
            else:
                print(f"   Coordenada nao encontrada ou CEP invalido: {cep}")
            
            time.sleep(0.5)

        if not dados_processados:
            print("Nenhum dado valido gerado. Cancelando upload.")
            return

        df_bronze = pl.DataFrame(dados_processados)
        df_bronze.write_parquet(self.local_temp_path)
        print(f"Arquivo Parquet gerado localmente com {len(df_bronze)} linhas validas.")
        
        self.enviar_para_r2()

    def enviar_para_r2(self):
        """Envia o arquivo protegido para o datalake no Cloudflare"""
        try:
            print(f"Enviando para o R2 no caminho: {self.r2_path} ...")
            self.s3_client.upload_file(
                Filename=self.local_temp_path,
                Bucket=self.bucket_name,
                Key=self.r2_path
            )
            print("Upload concluido com sucesso!")
        except Exception as e:
            print(f"Erro ao enviar para o R2: {e}")
        finally:
            if os.path.exists(self.local_temp_path):
                os.remove(self.local_temp_path)

if __name__ == "__main__":
    ingestor = IngestorCnpjBronze()
    dados_entrada = ingestor.carregar_dados_entrada()
    ingestor.processar_lista(dados_entrada)
