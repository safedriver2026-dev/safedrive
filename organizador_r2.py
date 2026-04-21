import os
import boto3
from botocore.config import Config

# --- CONFIGURAÇÃO VIA SECRETS ---
BUCKET = os.getenv("R2_BUCKET_NAME")
# Usamos o BQ_PROJECT_ID para identificar o nome da pasta do projeto no R2
PROJETO = os.getenv("BQ_PROJECT_ID") 

# Define a origem (onde caíram por erro) e o destino (dentro do projeto)
CAMINHO_ORIGEM = "datalake/bronze/malha_raw/"
CAMINHO_DESTINO = f"{PROJETO}/datalake/bronze/malha_raw/"

def mover_blocos_para_projeto():
    raw_endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
    
    # Blindagem de endpoint para o Boto3
    if raw_endpoint.endswith(f"/{BUCKET}"):
        raw_endpoint = raw_endpoint[: -len(f"/{BUCKET}")]

    s3 = boto3.client(
        's3',
        endpoint_url=raw_endpoint,
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID").strip(),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
    )

    print(f"🔍 À procura de blocões órfãos em: {CAMINHO_ORIGEM}")
    print(f"🎯 Destino correto: {CAMINHO_DESTINO}")
    
    # Lista o conteúdo da pasta errada
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=CAMINHO_ORIGEM)
    
    if 'Contents' not in response:
        print("ℹ️ Nenhum ficheiro encontrado para mover. A pasta já deve estar organizada!")
        return

    count = 0
    for obj in response['Contents']:
        old_key = obj['Key']
        
        # Filtra apenas os blocões agregados que criamos
        if "CNPJ_SP_HISTORICO_LOTE_" in old_key:
            nome_arquivo = old_key.split('/')[-1]
            new_key = f"{CAMINHO_DESTINO}{nome_arquivo}"
            
            print(f"🚚 A mover {nome_arquivo}...")
            
            # 1. Copia para a pasta do projeto (operação interna no R2)
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={'Bucket': BUCKET, 'Key': old_key},
                Key=new_key
            )
            
            # 2. Elimina o ficheiro da localização errada
            s3.delete_object(Bucket=BUCKET, Key=old_key)
            count += 1
            
    print(f"✅ Operação concluída! {count} ficheiros movidos para a pasta do projeto '{PROJETO}'.")

if __name__ == "__main__":
    mover_blocos_para_projeto()
