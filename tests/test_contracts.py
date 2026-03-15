import os
import sys
import pandas as pd
import numpy as np
from autobot.autobot_engine import MotorSafeDriver

def test_engine_init():
    e = MotorSafeDriver(habilitar_firestore=False)
    assert hasattr(e, "executar_pipeline_completo")

def test_coordinate_cleaning():
    e = MotorSafeDriver(habilitar_firestore=False)
    df = pd.DataFrame({'LATITUDE': ['-23.5', '0', '-'], 'LONGITUDE': ['-46.6', '0', '-'], 'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-01-01')]*3})
    df['DATA_OCORRENCIA_BO'] = pd.to_datetime(df['DATA_OCORRENCIA_BO'])
    dt, dr = e._etrar_trusted(df, 2026)
    assert len(dt) == 1
    assert dt['LATITUDE'].iloc[0] == -23.5

def test_janela():
    e = MotorSafeDriver(habilitar_firestore=False)
    assert (e.data_execucao - e.janela_inicio).days == 730
