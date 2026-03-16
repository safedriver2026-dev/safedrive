import os, sys, pandas as pd, numpy as np
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autobot.autobot_engine import MotorSafeDriver

def test_hard_reset_operational():
    if os.path.exists('datalake/metadata/baseline.lock'):
        os.remove('datalake/metadata/baseline.lock')
    m = MotorSafeDriver(persistencia=False)
    assert m.auditoria["modo"] == "HARD_RESET"

def test_bi_export_persistence():
    m = MotorSafeDriver(persistencia=False)
    df = pd.DataFrame({
        'NATUREZA_APURADA': ['ROUBO'], 'LATITUDE': [-23.5], 'LONGITUDE': [-46.6], 
        'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-01-01')], 'NUM_BO': ['1'], 'HORA_OCORRENCIA_BO': ['12:00']
    })
    t, r = m._qualificar(df, 2026)
    p, b = m._modelar(r)
    m._finalizar(p, b)
    assert os.path.exists('datalake/refined/power_bi_visualizacao.csv')

def test_fuzzy_match_resilience():
    m = MotorSafeDriver(persistencia=False)
    df = pd.DataFrame({'NATUREZA_APURADA': ['FURTO DE CELULAR'], 'LATITUDE': [-23.5], 'LONGITUDE': [-46.6], 'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-01-01')], 'NUM_BO': ['1']})
    t, r = m._qualificar(df, 2026)
    assert len(r) == 1
