import pytest
import pandas as pd
import os
import hashlib

CAMINHO_PRATA = "camada_prata.parquet"
CAMINHO_CRIME_REAL_AGREGADO = "crime_real_agregado.parquet" # Novo caminho para teste

def test_camada_prata_foi_gerada():
    assert os.path.exists(CAMINHO_PRATA), "Falha Crítica: O ficheiro camada_prata.parquet não foi gerado pelo módulo Silver."

def test_crime_real_agregado_foi_gerado():
    assert os.path.exists(CAMINHO_CRIME_REAL_AGREGADO), "Falha Crítica: O ficheiro crime_real_agregado.parquet não foi gerado pelo módulo Silver."

def test_fusao_geografica_master():
    if os.path.exists(CAMINHO_PRATA):
        df = pd.read_parquet(CAMINHO_PRATA)

        features_urbanas = [
            'DENSIDADE_LOGRADOUROS', 
            'PROPORCAO_RESIDENCIAL_H3', 
            'TOTAL_EDIFICACOES_H3'
        ]

        for col in features_urbanas:
            assert col in df.columns, f"Falha na integração: A feature urbana '{col}' está ausente."

def test_seguranca_anonimizacao_lgpd():
    if os.path.exists(CAMINHO_PRATA):
        df = pd.read_parquet(CAMINHO_PRATA)

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
    if os.path.exists(CAMINHO_PRATA):
        df = pd.read_parquet(CAMINHO_PRATA)

        colunas_vitais = [
            'INDICE_H3', 
            'EH_FERIADO', 
            'DIA_PAGAMENTO', 
            'PESO_CRIME'
        ]

        for col in colunas_vitais:
            assert col in df.columns, f"A coluna vital '{col}' desapareceu, o modelo de IA (Ouro) vai falhar."

        assert pd.api.types.is_integer_dtype(df['PESO_CRIME']), "Erro de Tipo: PESO_CRIME deve ser numérico inteiro."
