import os, sys, pandas as pd, numpy as np
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autobot.autobot_engine import MotorSafeDriver

def test_hard_reset_state():
    if os.path.exists('datalake/metadata/baseline.lock'):
        os.remove('datalake/metadata/baseline.lock')
    m = MotorSafeDriver(persistencia=False)
    assert m.auditoria["modo"] == "HARD_RESET"

def test_fuzzy_logic():
    m = MotorSafeDriver(persistencia=False)
    c = ['RUBRICA', 'LATITUDE', 'LONGITUDE']
    n = [m._normalizar(x) for x in c]
    assert 'NATUREZA_APURADA' in n

def test_data_integrity():
    m = MotorSafeDriver(persistencia=False)
    d = {'LATITUDE': ['-23.5', '0'], 'LONGITUDE': ['-46.6', '0'], 'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-01-01')]*2, 'NATUREZA_APURADA': ['ROUBO DE VEICULO']*2}
    t, r = m._qualificar(pd.DataFrame(d), 2026)
    assert len(t) == 1
