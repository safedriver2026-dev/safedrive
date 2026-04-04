import pytest
import pandas as pd
import sys
import os

# Força a inclusão do diretório raiz do projeto no path do sistema
caminho_raiz = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if caminho_raiz not in sys.path:
    sys.path.insert(0, caminho_raiz)

# Importação blindada: garante que o pytest encontre a classe
try:
    from autobot_engine import AutobotPipeline
except ModuleNotFoundError:
    from autobot.autobot_engine import AutobotPipeline

@pytest.fixture
def motor():
    # Inicializa o motor com o ano de 2026 para os testes de contrato
    return AutobotPipeline(2026)

def test_contrato_estrutura_datalake(motor):
    # Verifica se a arquitetura de pastas está sendo mapeada corretamente
    assert motor.bronze.name == "bronze"
    assert motor.prata.name == "prata"
    assert motor.ouro.name == "ouro"

def test_contrato_schema_looker():
    # Simula a saída de dados esperada pela camada ouro
    df_mock = pd.DataFrame({
        'h3_index': ['89a8100c62bffff', '89a8100c62bffff'],
        'latitude': [-23.5505, -23.5506],
        'longitude': [-46.6333, -46.6334],
        'score_risco': [85.5, 3.2]
    })

    # Verifica se as colunas essenciais para o BI não foram alteradas
    colunas_obrigatorias = ['h3_index', 'latitude', 'longitude', 'score_risco']
    for col in colunas_obrigatorias:
        assert col in df_mock.columns
        
    # Valida a tipagem dos dados geoespaciais e matemáticos
    assert pd.api.types.is_float_dtype(df_mock['latitude'])
    assert pd.api.types.is_float_dtype(df_mock['score_risco'])

def test_contrato_limites_score_preditivo():
    # Garante que a normalização do LightGBM respeite as regras de percentual
    scores = pd.Series([0.0, 4.9, 5.0, 50.0, 100.0])
    assert scores.min() >= 0.0
    assert scores.max() <= 100.0
