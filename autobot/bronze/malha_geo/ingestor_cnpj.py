import os
import requests
import polars as pl
import boto3
import time
from datetime import datetime

class IngestorCnpjBronze:
    def __init__(self):
        # 1. Configurações de Segurança (LGPD)
        self.salt = os.getenv('LGPD_SALT', 'default_salt')
        self.pepper = os.getenv('LGPD_PEPPER', 'default_pepper')
        
        # 2. Configurações do R2 Cloudflare
        self.s3_client = boto3.client('s3',
            endpoint_url=os.getenv('R2_ENDPOINT_URL', '').strip(),
            aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID', '').strip(),
            aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY', '').strip()
        )
        self.bucket_name = os.getenv('R2_BUCKET_NAME', '').strip()
        
        # 3. Caminho Exato no R2 e Local Temporário
        self.r2_path = "datalake/bronze/malha_raw/CNPJ_SP_PROTEGIDO.parquet"
        self.local_temp_path = "temp_bronze.parquet"
        
        self.br_api_url = "https://brasilapi.com.br/api/cep/v2/"

    def classificar_impacto_urbano(self, cnae_principal):
        """Inteligência de classificação urbana baseada nos primeiros dígitos do CNAE"""
        cnae_str = str(cnae_principal)
        prefixo = cnae_str[:2]
        
        if prefixo in ["47", "56", "96"]: 
            return "VAREJO_COMERCIO" # Geradores de Movimento
        if prefixo in ["49", "52", "10", "11"]: 
            return "INDUSTRIA_GALPAO" # Desertos Urbanos
        if prefixo in ["84"]:
            return "SEGURANCA_DELEGACIA"
            
        return "OUTROS"

    def capturar_coordenadas(self, cep):
        """Busca Lat/Lon do CEP via BrasilAPI"""
        cep_limpo = "".join(filter(str.isdigit, str(cep)))
        try:
            response = requests.get(f"{self.br_api_url}{cep_limpo}", timeout=10)
            if response.status_code == 200:
                coords = response.json().get('location', {}).get('coordinates', {})
                return coords.get('latitude'), coords.get('longitude')
        except Exception as e:
            print(f"Erro na API para o CEP {cep}: {e}")
        return None, None

    def processar_lista(self, lista_estabelecimentos):
        """Processa a lista, aplica LGPD e gera o Parquet"""
        print(f"Iniciando processamento de {len(lista_estabelecimentos)} registros...")
        dados_processados = []

        for item in lista_estabelecimentos:
            cnpj = str(item.get('cnpj', ''))
            cep = str(item.get('cep', ''))
            cnae = item.get('cnae', '')

            # 1. Captura a Coordenada
            lat, lon = self.capturar_coordenadas(cep)
            
            if lat and lon:
                # 2. Proteção LGPD (Hash)
                base_hash = f"{cnpj}{self.salt}{self.pepper}"
                hash_protegido = pl.Series([base_hash]).hash().cast(pl.Utf8)[0]

                # 3. Classificação de Impacto
                categoria = self.classificar_impacto_urbano(cnae)

                dados_processados.append({
                    "HASH_IDENTIFICADOR": hash_protegido,
                    "CATEGORIA": categoria,
                    "LAT": lat,
                    "LON": lon
                })
            else:
                print(f"⚠️ Coordenada não encontrada para o CEP: {cep}")
            
            # Rate limit gentil
            time.sleep(0.5)

        # 4. Criar DataFrame Polars e Salvar Localmente
        df_bronze = pl.DataFrame(dados_processados)
        df_bronze.write_parquet(self.local_temp_path)
        print(f"✅ Arquivo Parquet gerado localmente com {len(df_bronze)} linhas válidas.")
        
        # 5. Enviar para o R2 no caminho exato
        self.enviar_para_r2()

    def enviar_para_r2(self):
        """Envia o arquivo protegido para o datalake no Cloudflare"""
        try:
            print(f"🚀 Enviando para o R2 no caminho: {self.r2_path} ...")
            self.s3_client.upload_file(
                Filename=self.local_temp_path,
                Bucket=self.bucket_name,
                Key=self.r2_path
            )
            print("✅ Upload concluído com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao enviar para o R2: {e}")
        finally:
            # Limpa o ficheiro temporário local para segurança
            if os.path.exists(self.local_temp_path):
                os.remove(self.local_temp_path)

if __name__ == "__main__":
    # Exemplo de uso - Substitua esta lista pelos seus dados reais
    lista_teste = [
        {"cnpj": "12345678000199", "cep": "09725000", "cnae": "47113"}, # Varejo (SBC)
        {"cnpj": "98765432000100", "cep": "09850550", "cnae": "52117"}  # Galpão (SBC)
    ]
    
    ingestor = IngestorCnpjBronze()
    ingestor.processar_lista(lista_teste)
