import os
import requests
import json
import boto3
from botocore.config import Config
import io

def descobrir_metadados():
    R2_URL = os.getenv("R2_ENDPOINT_URL")
    R2_KEY = os.getenv("R2_ACCESS_KEY_ID")
    R2_SECRET = os.getenv("R2_SECRET_ACCESS_KEY")
    BUCKET = os.getenv("R2_BUCKET_NAME")

    try:
        print("\n" + "="*50)
        print("🔍 ESCANEANDO API DE METADADOS DO IBGE")
        print("="*50)
        
        # Consultando a pesquisa de Malhas (CL)
        url_meta = "https://apimetadados.ibge.gov.br/api/v1/pesquisas/CL"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(url_meta, headers=headers)
        response.raise_for_status()
        dados_brutos = response.json()

        # --- EXIBIÇÃO NA TELA ---
        print("\n📄 CONTEÚDO DO CATÁLOGO ENCONTRADO:")
        print(json.dumps(dados_brutos, indent=4, ensure_ascii=False))
        print("\n" + "="*50)
        # ------------------------

        print(f"\n☁️ Salvando cópia de segurança no R2: config/catalogo_ibge_bruto.json")
        r2 = boto3.client("s3", endpoint_url=R2_URL, aws_access_key_id=R2_KEY, 
                          aws_secret_access_key=R2_SECRET, config=Config(region_name="auto"))
        
        r2.put_object(
            Bucket=BUCKET, 
            Key="config/catalogo_ibge_bruto.json",
            Body=json.dumps(dados_brutos, indent=4, ensure_ascii=False)
        )
        print("✅ Processo concluído com sucesso!")

    except Exception as e:
        print(f"\n❌ ERRO AO EXTRAIR METADADOS: {e}")
        raise e

if __name__ == "__main__":
    descobrir_metadados()
