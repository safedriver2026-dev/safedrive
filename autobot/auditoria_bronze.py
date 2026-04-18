import polars as pl
import os, boto3, io, sys
from botocore.config import Config

def auditar_malha_bronze():
    # Configuração de Acesso (R2)
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL").strip(),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME").strip()
    # Atualizado para o novo caminho da Camada Bronze
    CAMINHO = "datalake/bronze/malha_geografica/malha_mestra_bronze.parquet"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        print(f"📡 Conectando ao Data Lake R2...")
        
        obj = r2.get_object(Bucket=BUCKET, Key=CAMINHO)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        print("\n" + "═"*70)
        print("🔍 RELATÓRIO DE AUDITORIA TÉCNICA - CAMADA BRONZE")
        print("═"*70)

        # 1. Validação de Unicidade (H3 é a PK)
        total_rows = len(df)
        unique_h3 = df["id_h3_int"].n_unique()
        duplicatas = total_rows - unique_h3
        
        # 2. Check de Tipagem
        h3_dtype = df.schema["id_h3_int"]
        
        print(f"📊 VOLUMETRIA TOTAL: {total_rows:,} hexágonos")
        print(f"🆔 UNICIDADE H3: {'✅ OK' if duplicatas == 0 else '❌ FALHA'} ({duplicatas} duplicatas)")
        print(f"🧬 TIPO ID_H3: {h3_dtype} {'(UInt64 - Excelente)' if h3_dtype == pl.UInt64 else '(Atenção: não é UInt64)'}")

        # 3. Auditoria de Preenchimento (Nulos Reais e Mascarados)
        print("\n" + "🛠️  CHECK DE QUALIDADE POR COLUNA")
        print(f"{'COLUNA':<25} | {'NULOS':<8} | {'NAO MAPEADO':<12}")
        print("-" * 60)
        
        cols_audit = ["logradouro", "bairro", "cidade_nome", "setor_id"]
        for col in cols_audit:
            if col in df.columns:
                nulos = df[col].null_count()
                mascarados = df.filter(pl.col(col).str.to_upper() == "NAO MAPEADO").height if df[col].dtype == pl.String else 0
                print(f"{col:<25} | {nulos:<8} | {mascarados:<12}")

        # 4. Distribuição Regional (Top 5 Cidades)
        print("\n" + "📍 COBERTURA GEOGRÁFICA (TOP 5 CIDADES)")
        dist_cidades = df.group_by("cidade_nome").agg(pl.len().alias("contagem")).sort("contagem", descending=True).head(5)
        print(dist_cidades)

        # 5. Amostra de Validação Visual
        print("\n" + "📋 AMOSTRA DE REGISTROS (HEAD 5)")
        with pl.Config(tbl_rows=5, tbl_width_chars=120, fmt_str_lengths=30):
            print(df.select(["id_h3_int", "logradouro", "bairro", "cidade_nome"]).head(5))

        # 6. Memória
        memoria_mb = df.estimated_size() / (1024 * 1024)
        print(f"\n📦 PEGADA DE MEMÓRIA: {memoria_mb:.2f} MB")
        print("═"*70)

    except Exception as e:
        print(f"❌ Erro crítico na auditoria: {e}")
        sys.exit(1)

if __name__ == "__main__":
    auditar_malha_bronze()
