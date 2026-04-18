import os, boto3, io, polars as pl, sys, traceback
from botocore.config import Config

def auditoria_qualidade():
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL").strip(),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME").strip()
    
    # Arquivos que vamos auditar
    ARQUIVOS = [
        "malha geografica/malha_mestra_consolidada_2025.parquet",
        "datalake/bronze/malha_geografica_infraestrutura.parquet",
        "malha geografica social/malha_social_bruta.parquet"
    ]

    try:
        r2 = boto3.client("s3", **R2_CONF)
        print("🔍 --- INICIANDO AUDITORIA DE QUALIDADE (DATA QUALITY) ---")

        for caminho in ARQUIVOS:
            print(f"\n📄 Analisando: {caminho}")
            
            # Download para a memória
            obj = r2.get_object(Bucket=BUCKET, Key=caminho)
            df = pl.read_parquet(io.BytesIO(obj['Body'].read()))

            total_linhas = len(df)
            print(f"📊 Total de registros: {total_linhas}")

            # 1. TESTE DE NULOS
            null_report = df.null_count()
            tem_nulo = False
            for col in df.columns:
                nulos = null_report[col][0]
                if nulos > 0:
                    perc = (nulos / total_linhas) * 100
                    print(f"   ⚠️ COLUNA [{col}]: {nulos} nulos ({perc:.2f}%)")
                    tem_nulo = True
            
            if not tem_nulo:
                print("   ✅ Nenhum campo nulo encontrado!")

            # 2. TESTE DE UNICIDADE (PK)
            if "id_h3_int" in df.columns:
                duplicados = total_linhas - df.select("id_h3_int").n_unique()
                if duplicados > 0:
                    print(f"   🚨 ERRO: {duplicados} hexágonos duplicados encontrados!")
                else:
                    print("   ✅ Integridade H3: Todos os IDs são únicos.")

            # 3. TESTE DE COBERTURA (Zonas Vazias)
            # Verifica se colunas de infra (ex: bar, police) estão todas zeradas
            if "police" in df.columns or "bus_stop" in df.columns:
                cols_infra = [c for c in df.columns if c not in ["id_h3_int", "bairro", "distrito", "cidade"]]
                # Soma total de equipamentos encontrados
                soma_total = df.select(pl.sum(pl.col(cols_infra))).sum_horizontal()[0]
                if soma_total == 0:
                    print("   ⚠️ AVISO: Todas as colunas de infraestrutura estão zeradas.")
                else:
                    print(f"   ✅ Cobertura: {soma_total} pontos de interesse mapeados.")

        print("\n--- AUDITORIA CONCLUÍDA ---")

    except Exception:
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    auditoria_qualidade()
