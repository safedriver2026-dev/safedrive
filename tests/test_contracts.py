import pytest
import pandas as pd
import os
from autobot.autobot_engine import SafeDriverEngine

def test_full_pipeline():
    bot = SafeDriverEngine(persistence=False)
    df = pd.DataFrame({
        'LATITUDE': ['-23.5'], 'LONGITUDE': ['-46.6'],
        'HORA_OCORRENCIA_BO': ['18:00'], 'NATUREZA_APURADA': ['ROUBO'],
        'DATA_OCORRENCIA_BO': ['2026-03-10']
    })
    res = bot._generate_gold(df)
    assert not res.empty
    assert os.path.exists('datalake/gold_refined/fato_risco.csv')

def test_weight_accuracy():
    bot = SafeDriverEngine(persistence=False)
    assert bot._get_weight(pd.Series({'N': 'LATROCINIO'})) == 10.0
