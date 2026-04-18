import os
import requests
import json
import boto3
from botocore.config import Config

def descobrir_metadados_sidra():
    R2_URL = os.getenv("R2_ENDPOINT_URL")
    R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
    BUCKET = os.getenv("R2_BUCKET_NAME")

    try:
        print("\n" + "="*50)
        print("🔍 EXPLORANDO METADADOS VIA API SIDRA (IBGE)")
        print("="*50)
        
        # Endpoint de tabelas (Metadados de pesquisas)
        # Vamos listar as tabelas do Censo Demográfico (Pesquisa 1) que é a mais rica
        url_sidra = "https://servicodados.ibge.gov.br/api/v3/agregados?pesquisa=1"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        print(f"📡 Acessando: {url_sidra}")
        response = requests.get(url_sidra, headers=headers)
        response.raise_for_status()
        tabelas = response.json()

        print(f"\n✅ {len(tabelas)} Grupos de Metadados Encontrados!")
        print("-" * 50)
        
        # Exibindo os primeiros grupos para você escolher a "Feature"
        for grupo in tabelas[:15]:
            print(f"ID: {grupo.get('id')} | Nome: {grupo.get('nome')}")
            # Mostra as tabelas dentro de cada grupo
            for agregado in grupo.get('agregados', [])[:2]:
                print(f"   -> Tabela {agregado.get('id')}: {agregado.get('nome')}")
        
        print("-" * 50)

        print(f"\n☁️ Salvando Catálogo no R2...")
        r2 = boto3.client("s3", endpoint_url=R2_URL, aws_access_key_id=R2_KEY, 
                          aws_secret_access_key=R2_SECRET, config=Config(region_name="auto"))
        
        r2.put_object(
            Bucket=BUCKET, 
            Key="config/catalogo_sidra_ibge.json",
            Body=json.dumps(tabelas, indent=4, ensure_ascii=False)
        )
        print("✅ Concluído.")

    except Exception as e:
        print(f"\n❌ ERRO NA EXPLORAÇÃO SIDRA: {e}")
        raise e

if __name__ == "__main__":
    descobrir_metadados_sidra()
