import pytest
import pandas as pd
import os
from autobot.autobot_engine import MotorSeguranca

def test_geracao_ouro():
    motor = MotorSeguranca(persistencia=False)
    dados = pd.DataFrame({
        'LATITUDE': ['-23.5'], 'LONGITUDE': ['-46.6'],
        'HORA_OCORRENCIA_BO': ['14:00'], 'NATUREZA_APURADA': ['ROUBO'],
        'DATA_OCORRENCIA_BO': ['2026-03-19']
    })
    res = motor._gerar_camada_ouro(dados)
    assert not res.empty
    assert os.path.exists('datalake/camada_ouro_refinada/esquema_estrela/fato_risco.csv')

def test_pesos_ia():
    motor = MotorSeguranca(persistencia=False)
    assert motor._definir_peso(pd.Series({'X': 'LATROCINIO'})) == 10.0
