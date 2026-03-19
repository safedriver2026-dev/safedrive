import pytest
import pandas as pd
import os
from autobot.motor_seguranca import MotorSegurancaPublica

def test_geracao_camada_ouro_refinada():
    motor = MotorSegurancaPublica(persistencia=False)
    dados_teste = pd.DataFrame({
        'LATITUDE': ['-23.5505'], 'LONGITUDE': ['-46.6333'],
        'HORA_OCORRENCIA_BO': ['19:00'], 'NATUREZA_APURADA': ['ROUBO DE CARGA'],
        'DATA_OCORRENCIA_BO': ['2026-03-18']
    })
    resultado = motor._gerar_camada_ouro(dados_teste)
    assert not resultado.empty
    assert os.path.exists('datalake/camada_ouro_refinada/esquema_estrela/fato_risco.csv')

def test_validacao_pesos_incidente():
    motor = MotorSegurancaPublica(persistencia=False)
    assert motor._atribuir_peso_crime(pd.Series({'X': 'LATROCINIO'})) == 10.0
