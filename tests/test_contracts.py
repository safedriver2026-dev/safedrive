import pytest
import pandas as pd
import os
from autobot.autobot_engine import MotorSeguranca

def test_processamento_camada_ouro():
    motor = MotorSeguranca(persistencia=False)
    dados = pd.DataFrame({
        'LATITUDE': ['-23.55'], 'LONGITUDE': ['-46.63'],
        'HORA_OCORRENCIA_BO': ['14:00'], 'NATUREZA_APURADA': ['ROUBO'],
        'DATA_OCORRENCIA_BO': ['2026-03-18']
    })
    resultado = motor._gerar_camada_ouro(dados)
    assert not resultado.empty
    assert os.path.exists('datalake/camada_ouro_refinada/esquema_estrela/fato_risco.csv')

def test_calculo_peso_risco():
    motor = MotorSeguranca(persistencia=False)
    assert motor._definir_peso(pd.Series({'X': 'LATROCINIO'})) == 10.0
