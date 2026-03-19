import pytest
import pandas as pd
from autobot.autobot_engine import MotorSeguranca

def test_reconstrucao_bronze():
    motor = MotorSeguranca(persistencia=False)
    motor._atualizar_bronze()
    # Verifica se ao menos um arquivo parquet foi criado na ausência de dados
    arquivos = os.listdir('datalake/camada_bronze_bruta')
    assert any(".parquet" in f for f in arquivos)

def test_integridade_multimodal():
    motor = MotorSeguranca(persistencia=False)
    dados = pd.DataFrame({
        'LATITUDE': [-23.5], 'LONGITUDE': [-46.6],
        'RUBRICA': ['ROUBO DE VEICULO']
    })
    res = motor._gerar_camada_ouro(dados)
    assert motor.telemetria['perfis']['Motorista'] == 1
    assert not res.empty
