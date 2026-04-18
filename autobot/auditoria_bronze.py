import polars as pl
import os, boto3, io, sys
from botocore.config import Config

def auditar_engenharia_bronze():
    R2_CONF = {
        "endpoint_url": os.getenv("R2_ENDPOINT_URL").strip(),
        "aws_access_key_id": os.getenv("R2_ACCESS_KEY_ID").strip(),
        "aws_secret_access_key": os.getenv("R2_SECRET_ACCESS_KEY").strip(),
        "config": Config(region_name="auto")
    }
    BUCKET = os.getenv("R2_BUCKET_NAME").strip()
    CAMINHO = "datalake/bronze/malha_geografica/malha_mestra_bronze.parquet"

    # Configuração de Sucesso (O Contrato)
    SCHEMA_ESPERADO = {
        'id_h3_int': pl.UInt64,
        'latitude': pl.Float32,
        'longitude': pl.Float32,
        'cidade_nome': pl.String,
        'logradouro': pl.String,
        'bairro': pl.String,
        'setor_id': pl.String
    }

    try:
        r2 = boto3.client("s3", **R2_CONF)
        obj = r2.get_object(Bucket=BUCKET, Key=CAMINHO)
        df = pl.read_parquet(io.BytesIO(obj['Body'].read()))

        print("\n" + "═"*70)
        print("🏗️  AUDITORIA DE ENGENHARIA: ESTRUTURA E INTEGRIDADE")
        print("═"*70)

        # 1. VERIFICAÇÃO DE CONTRATO (SCHEMA)
        erros_schema = []
        for col, dtype in SCHEMA_ESPERADO.items():
            if col not in df.columns:
                erros_schema.append(f"Faltando coluna: {col}")
            elif df.schema[col] != dtype:
                erros_schema.append(f"Tipo errado em {col}: Esperado {dtype}, Recebido {df.schema[col]}")
        
        print(f"🧬 SCHEMA: {'✅ VALIDADO' if not erros_schema else '❌ FALHA'}")
        for erro in erros_schema: print(f"   ↳ {erro}")

        # 2. VERIFICAÇÃO DE UNICIDADE (PK)
        total = len(df)
        duplicatas = total - df["id_h3_int"].n_unique()
        print(f"🆔 UNICIDADE (PK): {'✅ OK' if duplicatas == 0 else '❌ FALHA'} ({duplicatas:,} duplicatas)")

        # 3. VERIFICAÇÃO GEOMÉTRICA (BOUNDING BOX SP)
        # Limites aproximados do estado de São Paulo
        fora_limites = df.filter(
            (pl.col("latitude") < -25.5) | (pl.col("latitude") > -19.7) |
            (pl.col("longitude") < -53.2) | (pl.col("longitude") > -44.1)
        ).height
        print(f"🗺️  GEOMETRIA: {'✅ DENTRO DE SP' if fora_limites == 0 else '❌ FORA DE SP'} ({fora_limites} pontos fora)")

        # 4. VOLUMETRIA DE SEGURANÇA
        # Se o arquivo for muito menor que o esperado (ex: < 2M linhas), o pipeline falhou silenciosamente
        print(f"📊 VOLUMETRIA: {total:,} registros")
        if total < 2700000:
            print("   ⚠️ AVISO: Volume abaixo do esperado para o Estado de SP!")

        print("═"*70)
        
        # Se houver erro crítico de unicidade ou schema, encerra com erro para travar o CI/CD
        if duplicatas > 0 or erros_schema:
            sys.exit(1)

    except Exception as e:
        print(f"❌ Erro na conexão/leitura: {e}")
        sys.exit(1)

if __name__ == "__main__":
    auditar_engenharia_bronze()
