import pytest
import pandas as pd
import os
from autobot.autobot_engine import MotorSeguranca

def test_processamento_ouro():
    bot = MotorSeguranca(persistencia=False)
    amostra = pd.DataFrame({
        'LATITUDE': ['-23.5'], 'LONGITUDE': ['-46.6'],
        'HORA_OCORRENCIA_BO': ['15:00'], 'NATUREZA_APURADA': ['ROUBO'],
        'DATA_OCORRENCIA_BO': ['2026-03-01']
    })
    resultado = bot._gerar_camada_ouro(amostra)
    assert not resultado.empty
    assert os.path.exists('datalake/gold_refined/fato_risco.csv')

def test_calculo_peso():
    bot = MotorSeguranca(persistencia=False)
    assert bot._definir_peso(pd.Series({'X': 'LATROCINIO'})) == 10.0
