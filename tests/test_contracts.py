import pytest
import pandas as pd
from autobot.autobot_engine import MotorSafeDriver

def test_estrutura_diretorios():
    motor = MotorSafeDriver()
    motor.gerenciar_execucao()
    assert motor.bronze.exists()
    assert motor.prata.exists()
    assert motor.ouro.exists()

def test_processamento_ia():
    motor = MotorSafeDriver()
    df_fake = pd.DataFrame({
        'latitude': [-23.5, -23.6, -23.7],
        'longitude': [-46.6, -46.7, -46.8]
    })
    resultado = motor.aplicar_modelos(df_fake)
    assert 'score_risco' in resultado.columns
    assert len(resultado) == 3
