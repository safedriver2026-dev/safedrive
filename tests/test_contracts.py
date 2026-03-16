import os, sys, pandas as pd, numpy as np
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autobot.autobot_engine import MotorSafeDriver

def test_engine_instancia():
    engine = MotorSafeDriver(persistencia=False)
    assert hasattr(engine, "rodar")

def test_correcao_coordenada_ssp():
    engine = MotorSafeDriver(persistencia=False)
    valor_errado = -46779670.0
    resultado = engine._corrigir_ponto_decimal(valor_errado, is_lat=False)
    assert resultado == -46.779670

def test_normalizacao_semantica():
    engine = MotorSafeDriver(persistencia=False)
    assert engine._normalizar("RUBRICA") == "NATUREZA_APURADA"

def test_qualificacao_fluxo():
    engine = MotorSafeDriver(persistencia=False)
    df = pd.DataFrame({
        'LATITUDE': ['-23.5'], 'LONGITUDE': ['-46.6'],
        'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-01-01')],
        'NATUREZA_APURADA': ['ROUBO DE VEICULO'],
        'NUM_BO': ['1']
    })
    t, r = engine._qualificar(df, 2026)
    assert len(t) == 1
    assert len(r) == 1

def test_modo_reset():
    if os.path.exists('datalake/metadata/baseline.lock'):
        os.remove('datalake/metadata/baseline.lock')
    engine = MotorSafeDriver(persistencia=False)
    assert engine.auditoria["modo"] == "HARD_RESET"
