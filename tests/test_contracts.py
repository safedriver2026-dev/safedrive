import pytest
import pandas as pd
import numpy as np
from autobot.autobot_engine import MotorSeguranca

def test_volumetria_multimodal():
    motor = MotorSeguranca(persistencia=False)
    dados = pd.DataFrame({
        'LATITUDE': [-23.5, -23.51, -23.52],
        'LONGITUDE': [-46.6, -46.61, -46.62],
        'RUBRICA': ['ROUBO DE VEICULO', 'FURTO DE BICICLETA', 'ROUBO CELULAR PEDESTRE']
    })
    motor._gerar_camada_ouro(dados)
    assert motor.telemetria['perfis']['Motorista'] == 1
    assert motor.telemetria['perfis']['Ciclista'] == 1
    assert motor.telemetria['perfis']['Pedestre'] == 1

def test_precisao_estatistica():
    motor = MotorSeguranca(persistencia=False)
    dados = pd.DataFrame({
        'LATITUDE': np.random.uniform(-24, -23, 100),
        'LONGITUDE': np.random.uniform(-47, -46, 100),
        'RUBRICA': ['ROUBO'] * 100
    })
    motor._gerar_camada_ouro(dados)
    assert motor.telemetria['ia']['mae'] >= 0
    assert -1 <= motor.telemetria['ia']['r2'] <= 1

def test_sanitizacao_silver():
    motor = MotorSeguranca(persistencia=False)
    dados = pd.DataFrame({'LATITUDE': ['-23,5'], 'LONGITUDE': ['-46,6']})
    res = motor._gerar_camada_silver(dados)
    assert res.iloc[0]['LATITUDE'] == -23.5
