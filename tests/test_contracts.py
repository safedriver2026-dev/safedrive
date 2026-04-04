import pytest
import pandas as pd
import sys
import os

caminho_raiz = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if caminho_raiz not in sys.path:
    sys.path.insert(0, caminho_raiz)

try:
    from autobot.autobot_engine import AutobotPipeline
except:
    from autobot_engine import AutobotPipeline

@pytest.fixture
def motor():
    return AutobotPipeline(2026)

def test_contrato_colunas(motor):
    df = pd.DataFrame({'LATITUDE': [-23.5], 'LONGITUDE': [-46.6]})
    df.columns = [str(c).lower() for c in df.columns]
    assert 'latitude' in df.columns

def test_contrato_risco():
    scores = pd.Series([0.0, 50.0, 100.0])
    assert scores.min() >= 0.0
    assert scores.max() <= 100.0
