import polars as pl
import os, boto3, io, sys
from botocore.config import Config

def amostra_tabela_final():
    # Configuração de Acesso (R2)
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL").strip(),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME").strip()
    CAMINHO = "malha geografica/malha_mestra_consolidada_2025.parquet"

    try:
        r2 = boto3.client("s3", **R2_CONF)
        print(f"📥 Baixando amostra de: {CAMINHO}...")
        
        # Lendo o arquivo
        obj = r2.get_object(Bucket=BUCKET, Key=CAMINHO)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        print("\n--- 🧩 ESTRUTURA DA TABELA FINAL (ESQUEMA) ---")
        for col, dtype in df.schema.items():
            print(f"🔹 {col:25} | Tipo: {dtype}")

        print("\n--- 📊 AMOSTRA DOS DADOS (TOP 10) ---")
        # Selecionamos as colunas principais para a visualização não quebrar na tela
        cols_viz = ["id_h3_int", "setor_id", "bairro", "cidade_nome", "regiao_imediata"]
        
        # Filtramos as colunas que realmente existem para evitar erro
        cols_existentes = [c for c in cols_viz if c in df.columns]
        
        # Mostra a amostra
        with pl.Config(tbl_rows=10, tbl_width_chars=120):
            print(df.select(cols_existentes).head(10))

        print("\n--- 📈 RESUMO DE PREENCHIMENTO ---")
        total = len(df)
        print(f"Total de Hexágonos H3: {total}")
        print(f"Setores Vinculados:   {df.select(pl.col('setor_id').is_not_null().sum())[0,0]} (Erro: {df.select(pl.col('setor_id').is_null().sum())[0,0]} nulos)")

    except Exception as e:
        print(f"❌ Erro ao gerar amostra: {e}")
        sys.exit(1)

if __name__ == "__main__":
    amostra_tabela_final()
