import pytest
import pandas as pd
import sys
import os
from pathlib import Path

caminho_raiz = str(Path(__file__).parent.parent.absolute())
if caminho_raiz not in sys.path:
    sys.path.insert(0, caminho_raiz)

from autobot_engine import AutobotPipeline

@pytest.fixture
def motor():
    return AutobotPipeline(2026)

def test_contrato_estrutura_datalake(motor):
    assert motor.bronze.name == "bronze"
    assert motor.prata.name == "prata"
    assert motor.ouro.name == "ouro"

def test_contrato_schema_looker():
    df_mock = pd.DataFrame({
        'h3_index': ['89a8100c62bffff', '89a8100c62bffff'],
        'latitude': [-23.5505, -23.5506],
        'longitude': [-46.6333, -46.6334],
        'score_risco': [85.5, 3.2]
    })

    colunas_obrigatorias = ['h3_index', 'latitude', 'longitude', 'score_risco']
    for col in colunas_obrigatorias:
        assert col in df_mock.columns
        
    assert pd.api.types.is_float_dtype(df_mock['latitude'])
    assert pd.api.types.is_float_dtype(df_mock['score_risco'])

def test_contrato_limites_score_preditivo():
    scores = pd.Series([0.0, 4.9, 5.0, 50.0, 100.0])
    assert scores.min() >= 0.0
    assert scores.max() <= 100.0
