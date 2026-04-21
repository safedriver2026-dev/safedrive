import os
import boto3
from botocore.config import Config

# --- CONFIGURAÇÃO ---
BUCKET = os.getenv("R2_BUCKET_NAME")
# Forçamos o destino para a pasta do projeto no R2
CAMINHO_DESTINO = "safedriver/datalake/bronze/malha_raw/"
CAMINHO_ORIGEM = "datalake/bronze/malha_raw/"

def organizar_r2():
    raw_endpoint = os.getenv("R2_ENDPOINT_URL", "").strip().rstrip('/')
    if raw_endpoint.endswith(f"/{BUCKET}"):
        raw_endpoint = raw_endpoint[: -len(f"/{BUCKET}")]

    s3 = boto3.client(
        's3',
        endpoint_url=raw_endpoint,
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID").strip(),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        config=Config(signature_version='s3v4', s3={'addressing_style': 'path'})
    )

    print(f"🔍 Verificando ficheiros órfãos em: {CAMINHO_ORIGEM}")
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=CAMINHO_ORIGEM)
    
    if 'Contents' not in response:
        print("✅ Nada para mover! A raiz já está limpa.")
        return

    for obj in response['Contents']:
        old_key = obj['Key']
        if "CNPJ_SP_HISTORICO_LOTE_" in old_key:
            nome_arquivo = old_key.split('/')[-1]
            new_key = f"{CAMINHO_DESTINO}{nome_arquivo}"
            
            print(f"🚚 Movendo para projeto: {nome_arquivo}")
            
            # Copia para o destino correto
            s3.copy_object(
                Bucket=BUCKET,
                CopySource={'Bucket': BUCKET, 'Key': old_key},
                Key=new_key
            )
            # Apaga o rastro na pasta errada
            s3.delete_object(Bucket=BUCKET, Key=old_key)
            
    print("✨ Organização concluída com sucesso no R2!")

if __name__ == "__main__":
    organizar_r2()
