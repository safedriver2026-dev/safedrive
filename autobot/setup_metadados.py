import os
import requests
import json
import boto3
from botocore.config import Config

def descobrir_metadados():
    R2_URL = os.getenv("R2_ENDPOINT_URL")
    R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
    BUCKET = os.getenv("R2_BUCKET_NAME")

    try:
        print("\n" + "="*50)
        print("🔍 EXPLORANDO API DE METADADOS DO IBGE")
        print("="*50)
        
        # Endpoint raiz para listar TODAS as pesquisas disponíveis
        url_lista_pesquisas = "https://apimetadados.ibge.gov.br/api/v1/pesquisas"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        print(f"📡 Acessando: {url_lista_pesquisas}")
        response = requests.get(url_lista_pesquisas, headers=headers)
        response.raise_for_status()
        todas_pesquisas = response.json()

        # --- EXIBIÇÃO NA TELA ---
        print(f"\n✅ {len(todas_pesquisas)} Pesquisas Encontradas!")
        print("-" * 50)
        # Exibe as pesquisas para você identificar a sigla correta no log
        for p in todas_pesquisas:
            print(f"Sigla: {p.get('sigla')} | Nome: {p.get('nome')}")
        print("-" * 50)
        # ------------------------

        print(f"\n☁️ Salvando lista completa no R2: config/lista_pesquisas_ibge.json")
        r2 = boto3.client("s3", endpoint_url=R2_URL, aws_access_key_id=R2_KEY, 
                          aws_secret_access_key=R2_SECRET, config=Config(region_name="auto"))
        
        r2.put_object(
            Bucket=BUCKET, 
            Key="config/lista_pesquisas_ibge.json",
            Body=json.dumps(todas_pesquisas, indent=4, ensure_ascii=False)
        )
        print("✅ Backup salvo no R2.")

    except Exception as e:
        print(f"\n❌ ERRO NA EXPLORAÇÃO: {e}")
        raise e

if __name__ == "__main__":
    descobrir_metadados()
