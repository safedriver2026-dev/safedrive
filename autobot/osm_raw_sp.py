import boto3
import os
import subprocess
import sys

def filtrar_osm_sp():
    print("✂️ INICIANDO RECORTE: Sudeste.pbf -> SP.pbf")
    
    # Configuração R2 (Usando o padrão que você validou como funcional)
    s3 = boto3.client('s3', 
                      endpoint_url=os.getenv('R2_ENDPOINT_URL').strip(), 
                      aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID').strip(), 
                      aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY').strip())
    
    bucket = os.getenv('R2_BUCKET_NAME').strip()
    input_file = "sudeste-latest.osm.pbf"
    output_file = "sp-latest.osm.pbf"
    
    # 1. Download do Sudeste Bruto da Bronze
    print(f"📥 Baixando {input_file} do R2...")
    s3.download_file(bucket, f'datalake/bronze/malha_raw/{input_file}', input_file)

    # 2. Executar Osmium Extract
    # Limites Geográficos (BBOX) de São Paulo: oeste, sul, leste, norte
    bbox = "-53.15,-25.35,-44.10,-19.70"
    
    print(f"🚜 Recortando para limites de SP (BBOX: {bbox})...")
    try:
        # O osmium é muito mais eficiente que qualquer biblioteca Python pura
        subprocess.run([
            "osmium", "extract", 
            "--bbox", bbox, 
            input_file, 
            "-o", output_file, 
            "--overwrite"
        ], check=True)
        print("✅ Recorte concluído com sucesso!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao executar Osmium: {e}")
        sys.exit(1)

    # 3. Upload do arquivo SP para a Camada Raw (Substituindo o uso do Sudeste nas próximas etapas)
    print(f"🚀 Subindo {output_file} para malha_raw...")
    s3.upload_file(output_file, bucket, f'datalake/bronze/malha_raw/{output_file}')
    
    # Limpeza local para não estourar o disco do runner
    if os.path.exists(input_file): os.remove(input_file)
    if os.path.exists(output_file): os.remove(output_file)
    print("🧹 Workspace limpo.")

if __name__ == "__main__":
    filtrar_osm_sp()
