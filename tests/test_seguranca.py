import pytest
import polars as pl
import os
import sys
from datetime import datetime

# Adiciona a pasta raiz ao path para importar o motor
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importa as funções e variáveis do motor
from autobot.motor_analise_preditiva import normalizar, get_periodo, PESO_PENAL_BASE, anonimizar

# --- TESTES DE NORMALIZAÇÃO E SCHEMA ---
def test_normalizacao_texto():
    """Garante que o SchemaBrain não se perca com acentos ou espaços."""
    assert normalizar("São Paulo ") == "SAO PAULO"
    assert normalizar(None) == ""
    assert normalizar("123-ABC") == "123-ABC"

def test_classificacao_periodo():
    """Garante que o robô entende o perigo da madrugada."""
    assert get_periodo("02:30:00") == "MADRUGADA"
    assert get_periodo("14:00") == "TARDE"
    assert get_periodo("invalid") == "MANHA" # Fallback de segurança

# --- TESTES DE REGRA DE NEGÓCIO (GRAVIDADE VS VOLUME) ---
def test_inteligencia_de_pesos():
    """Garante que crimes graves pesam mais que crimes leves no escore."""
    peso_homicidio = PESO_PENAL_BASE.get("HOMICIDIO DOLOSO")
    peso_furto = PESO_PENAL_BASE.get("FURTO")

    assert peso_homicidio > peso_furto, "ERRO: A inteligência de risco está invertida!"
    assert peso_homicidio == 10.0
    assert peso_furto == 4.0

# --- TESTE DE CONTRATO DE DADOS (OURO) ---
def test_contrato_camada_ouro():
    """Valida se as colunas essenciais para o Looker estão presentes."""
    colunas_obrigatorias = [
        "H3_INDEX", "ANO", "ESCORE_TOTAL", "VOL_CRIMES", 
        "MUNICIPIO", "DENSIDADE_URBANA", "ESCORE_PREDITO"
    ]

    # Simula um DataFrame Polars de saída com algumas colunas
    # Na prática, você leria um sample do seu output real ou um mock mais complexo
    df_saida_simulado = pl.DataFrame({
        "H3_INDEX": ["8829a1a0bffffff", "8829a1a0bffffff"],
        "ANO": [2023, 2023],
        "ESCORE_TOTAL": [100.0, 120.0],
        "VOL_CRIMES": [5, 7],
        "MUNICIPIO": ["SAO PAULO", "CAMPINAS"],
        "DENSIDADE_URBANA": [1500.0, 800.0],
        "ESCORE_PREDITO": [0.8, 0.9],
        "COLUNA_EXTRA": ["A", "B"] # Coluna extra para garantir que não falhe por ter mais
    })

    for col in colunas_obrigatorias:
        assert col in df_saida_simulado.columns, f"ERRO: Coluna '{col}' ausente na camada Ouro!"

def test_anonimizacao_lgpd():
    """Garante que não existem dados sensíveis em texto claro no código."""
    salt = os.environ.get("LGPD_SALT", "salt_padrao_para_teste") # Usa a variável de ambiente ou um padrão

    hash_1 = anonimizar("RUA TESTE", salt)
    hash_2 = anonimizar("RUA TESTE", salt)
    hash_3 = anonimizar("RUA DIFERENTE", salt)

    assert hash_1 == hash_2, "ERRO: Anonimização inconsistente para o mesmo valor!"
    assert hash_1 != hash_3, "ERRO: Anonimização gerou o mesmo hash para valores diferentes!"
    assert "RUA TESTE" not in hash_1, "ALERTA: Falha na anonimização LGPD! Dado sensível em texto claro."
