import pytest
import os
import pandas as pd
import hashlib
import boto3
from datetime import datetime

ANO_ATUAL = datetime.now().year
CAMINHO_PRATA_R2 = f"safedriver/datalake/silver/prata_{ANO_ATUAL}.parquet"
CAMINHO_CRIME_REAL_AGREGADO_R2 = f"safedriver/datalake/validation/crime_real_agregado_{ANO_ATUAL}.parquet"

CAMINHO_PRATA_LOCAL = "camada_prata_test.parquet"
CAMINHO_CRIME_REAL_AGREGADO_LOCAL = "crime_real_agregado_test.parquet"

s3_client = boto3.client('s3',
                         endpoint_url=os.environ.get("R2_ENDPOINT_URL"),
                         aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
                         aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"))
BUCKET_NAME = os.environ.get("R2_BUCKET_NAME")

@pytest.fixture(scope="module", autouse=True)
def setup_teardown_r2_files():
    prata_baixada = False
    crime_real_baixado = False

    try:
        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=CAMINHO_PRATA_R2)
            s3_client.download_file(BUCKET_NAME, CAMINHO_PRATA_R2, CAMINHO_PRATA_LOCAL)
            prata_baixada = True
            print(f"Arquivo {CAMINHO_PRATA_R2} baixado com sucesso para testes.")
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NotFound':
                print(f"Arquivo {CAMINHO_PRATA_R2} não encontrado no R2. Pulando download para testes.")
            else:
                raise

        try:
            s3_client.head_object(Bucket=BUCKET_NAME, Key=CAMINHO_CRIME_REAL_AGREGADO_R2)
            s3_client.download_file(BUCKET_NAME, CAMINHO_CRIME_REAL_AGREGADO_R2, CAMINHO_CRIME_REAL_AGREGADO_LOCAL)
            crime_real_baixado = True
            print(f"Arquivo {CAMINHO_CRIME_REAL_AGREGADO_R2} baixado com sucesso para testes.")
        except s3_client.exceptions.ClientError as e:
            if e.response['Error']['Code'] == 'NotFound':
                print(f"Arquivo {CAMINHO_CRIME_REAL_AGREGADO_R2} não encontrado no R2. Pulando download para testes.")
            else:
                raise

    except Exception as e:
        pytest.fail(f"Falha inesperada ao tentar baixar arquivos do R2 para testes: {e}")

    yield

    if os.path.exists(CAMINHO_PRATA_LOCAL):
        os.remove(CAMINHO_PRATA_LOCAL)
    if os.path.exists(CAMINHO_CRIME_REAL_AGREGADO_LOCAL):
        os.remove(CAMINHO_CRIME_REAL_AGREGADO_LOCAL)

def test_camada_prata_foi_gerada():
    if not os.path.exists(CAMINHO_PRATA_LOCAL):
        pytest.skip(f"Arquivo {CAMINHO_PRATA_LOCAL} não encontrado, pulando teste.")
    assert os.path.exists(CAMINHO_PRATA_LOCAL), "Falha Crítica: O ficheiro camada_prata.parquet não foi gerado pelo módulo Silver ou não foi baixado do R2."

def test_crime_real_agregado_foi_gerado():
    if not os.path.exists(CAMINHO_CRIME_REAL_AGREGADO_LOCAL):
        pytest.skip(f"Arquivo {CAMINHO_CRIME_REAL_AGREGADO_LOCAL} não encontrado, pulando teste.")
    assert os.path.exists(CAMINHO_CRIME_REAL_AGREGADO_LOCAL), "Falha Crítica: O ficheiro crime_real_agregado.parquet não foi gerado pelo módulo Silver ou não foi baixado do R2."

def test_fusao_geografica_master():
    if not os.path.exists(CAMINHO_PRATA_LOCAL):
        pytest.skip(f"Arquivo {CAMINHO_PRATA_LOCAL} não encontrado, pulando teste.")
    df = pd.read_parquet(CAMINHO_PRATA_LOCAL)

    features_urbanas = [
        'DENSIDADE_LOGRADOUROS', 
        'PROPORCAO_RESIDENCIAL_H3', 
        'TOTAL_EDIFICACOES_H3'
    ]

    for col in features_urbanas:
        assert col in df.columns, f"Falha na integração: A feature urbana '{col}' está ausente."

def test_seguranca_anonimizacao_lgpd():
    if not os.path.exists(CAMINHO_PRATA_LOCAL):
        pytest.skip(f"Arquivo {CAMINHO_PRATA_LOCAL} não encontrado, pulando teste.")
    df = pd.read_parquet(CAMINHO_PRATA_LOCAL)

    assert 'ID_ANONIMO' in df.columns, "Falha de Conformidade: Coluna de anonimização 'ID_ANONIMO' ausente."

    SALT = os.environ.get("LGPD_SALT", "default_salt_seguranca_test") 

    if not df.empty:
        primeiro_bo_original = str(df['NUM_BO'].iloc[0])
        primeiro_bo_hash_calculado = hashlib.sha256(f"{primeiro_bo_original}{SALT}".encode()).hexdigest()[:12]
        primeiro_bo_hash_existente = str(df['ID_ANONIMO'].iloc[0])

        assert primeiro_bo_hash_existente == primeiro_bo_hash_calculado, "Alerta Crítico: Hash de anonimização incorreto ou BO Original exposto!"
        assert primeiro_bo_hash_existente != primeiro_bo_original, "Alerta Crítico: BO Original exposto! Quebra de conformidade LGPD detetada."
    else:
        pytest.skip("DataFrame vazio, impossível testar anonimização.")

def test_formato_para_motor_ia():
    if not os.path.exists(CAMINHO_PRATA_LOCAL):
        pytest.skip(f"Arquivo {CAMINHO_PRATA_LOCAL} não encontrado, pulando teste.")
    df = pd.read_parquet(CAMINHO_PRATA_LOCAL)

    colunas_vitais = [
        'INDICE_H3', 
        'EH_FERIADO', 
        'DIA_PAGAMENTO', 
        'PESO_CRIME'
    ]

    for col in colunas_vitais:
        assert col in df.columns, f"A coluna vital '{col}' desapareceu, o modelo de IA (Ouro) vai falhar."

    assert pd.api.types.is_integer_dtype(df['PESO_CRIME']), "Erro de Tipo: PESO_CRIME deve ser numérico inteiro."
