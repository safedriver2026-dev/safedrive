import pytest
import pandas as pd
import os
from autobot.autobot_engine import MotorSeguranca

def test_processamento_camada_ouro_total():
    motor = MotorSeguranca(persistencia=False)
    dados = pd.DataFrame({
        'LATITUDE': ['-23.5505'], 'LONGITUDE': ['-46.6333'],
        'HORA_OCORRENCIA_BO': ['18:00'], 'NATUREZA_APURADA': ['ROUBO DE VEICULO'],
        'DATA_OCORRENCIA_BO': ['2026-03-18']
    })
    resultado = motor._gerar_camada_ouro(dados)
    assert not resultado.empty
    assert os.path.exists('datalake/camada_ouro_refinada/esquema_estrela/fato_risco.csv')
    assert os.path.exists('datalake/camada_ouro_refinada/malha_mapa_auditavel.parquet')

def test_logica_peso_incidente():
    motor = MotorSeguranca(persistencia=False)
    assert motor._definir_peso(pd.Series({'X': 'SEQUESTRO'})) == 10.0
