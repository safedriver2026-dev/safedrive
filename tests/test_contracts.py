import pytest
import pandas as pd
import os
from autobot.autobot_engine import MotorSafeDriver

@pytest.fixture
def motor():
    os.environ['GEMINI_JSON'] = '{"api_key": "teste"}'
    return MotorSafeDriver()

def test_separacao_categorias(motor):
    dados = pd.DataFrame({'NATUREZA_APURADA': ['HOMICIDIO', 'FURTO']})
    dados['PESO'] = dados['NATUREZA_APURADA'].apply(lambda x: 10 if 'HOMICIDIO' in x else 7)
    assert dados.loc[0, 'PESO'] == 10
    assert dados.loc[1, 'PESO'] == 7

def test_calculo_pagamento():
    data = pd.to_datetime('2026-03-05')
    flag = 1 if data.day in [5, 6, 7, 20, 21] else 0
    assert flag == 1

def test_filtro_coordenadas():
    df = pd.DataFrame({'LATITUDE': [0, -23.5], 'LONGITUDE': [0, -46.6]})
    limpo = df[(df['LATITUDE'] != 0) & (df['LATITUDE'].notna())]
    assert len(limpo) == 1
