import pytest
import pandas as pd
import os
import numpy as np
from autobot.autobot_engine import MotorSeguranca

def test_integridade_silver_trusted():
    motor = MotorSeguranca(persistencia=False)
    dados_entrada = pd.DataFrame({
        'LATITUDE': ['-23,5432', '0.0', 'invalido', '-23.5'],
        'LONGITUDE': ['-46,6321', '0.0', '-46.6', '-46.6'],
        'RUBRICA': ['ROUBO', 'FURTO', 'ROUBO', 'ROUBO']
    })
    resultado = motor._gerar_camada_silver(dados_entrada)
    assert len(resultado) == 2
    assert (resultado['LATITUDE'] < -19).all()

def test_metricas_preditivas_ia():
    motor = MotorSeguranca(persistencia=False)
    dados_treino = pd.DataFrame({
        'LATITUDE': np.random.uniform(-24, -23, 50),
        'LONGITUDE': np.random.uniform(-47, -46, 50),
        'RUBRICA': ['ROUBO'] * 25 + ['LATROCINIO'] * 25
    })
    motor._gerar_camada_ouro(dados_treino)
    ia = motor.auditoria['ia']
    assert 'mae' in ia
    assert 'rmse' in ia
    assert 'r2' in ia
    assert ia['mae'] >= 0

def test_pesos_severidade_estatistica():
    motor = MotorSeguranca(persistencia=False)
    assert motor._definir_peso(pd.Series({'V': 'LATROCINIO'})) == 10.0
    assert motor._definir_peso(pd.Series({'V': 'FURTO'})) == 4.0
    assert motor._definir_peso(pd.Series({'V': 'OUTROS'})) == 1.0

def test_persistência_esquema_estrela():
    motor = MotorSeguranca(persistencia=False)
    dados = pd.DataFrame({
        'LATITUDE': [-23.5], 'LONGITUDE': [-46.6],
        'RUBRICA': ['ROUBO']
    })
    motor._gerar_camada_ouro(dados)
    assert os.path.exists('datalake/camada_ouro_refinada/esquema_estrela/fato_risco.csv')
