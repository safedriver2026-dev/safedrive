import polars as pl
import os, boto3, io, sys
from botocore.config import Config

def auditar_malha_bronze():
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL").strip(),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME").strip()
    CAMINHO = "datalake/bronze/malha_geografica/malha_mestra_bronze.parquet"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        obj = r2.get_object(Bucket=BUCKET, Key=CAMINHO)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        print("\n" + "═"*70)
        print("🔍 RELATÓRIO DE AUDITORIA TÉCNICA - CAMADA BRONZE (FIXED)")
        print("═"*70)

        # 1. Unicidade
        total_rows = len(df)
        unique_h3 = df["id_h3_int"].n_unique()
        duplicatas = total_rows - unique_h3
        
        print(f"📊 VOLUMETRIA TOTAL: {total_rows:,} hexágonos")
        print(f"🆔 UNICIDADE H3: {'✅ OK' if duplicatas == 0 else '❌ FALHA'} ({duplicatas:,} duplicatas)")

        # 2. Check de Qualidade (Corrigido to_uppercase)
        print("\n🛠️  CHECK DE QUALIDADE POR COLUNA")
        print(f"{'COLUNA':<25} | {'NULOS':<8} | {'NAO MAPEADO':<12}")
        print("-" * 60)
        
        for col in ["logradouro", "bairro", "cidade_nome", "setor_id"]:
            if col in df.columns:
                nulos = df[col].null_count()
                # AQUI ESTAVA O ERRO: Corrigido para to_uppercase()
                mascarados = df.filter(pl.col(col).str.to_uppercase() == "NAO MAPEADO").height if df[col].dtype == pl.String else 0
                print(f"{col:<25} | {nulos:<8} | {mascarados:<12}")

        if duplicatas > 0:
            print(f"\n⚠️  ALERTA: Você tem {duplicatas:,} registros repetidos. Isso vai corromper o modelo de IA!")

        print("═"*70)

    except Exception as e:
        print(f"❌ Erro crítico na auditoria: {e}")
        sys.exit(1)

if __name__ == "__main__":
    auditar_malha_bronze()
