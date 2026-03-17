import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autobot.autobot_engine import MotorSafeDriver

def test_engine_instancia():
    engine = MotorSafeDriver(persistencia=False)
    assert hasattr(engine, "rodar")

def test_correcao_coordenada_ssp():
    engine = MotorSafeDriver(persistencia=False)
    assert -47.0 < engine._corrigir_ponto_decimal(-46779670.0, is_lat=False) < -46.0

def test_normalizacao_semantica():
    engine = MotorSafeDriver(persistencia=False)
    # Agora o método existe com o nome exato esperado pelo teste
    assert engine._normalizar("RUBRICA") == "NATUREZA_APURADA"

def test_classificacao_crime():
    engine = MotorSafeDriver(persistencia=False)
    # Agora o método existe com o nome exato esperado pelo teste
    assert engine._classificar_crime("ROUBO DE CARGA") == "ROUBO DE CARGA"
    assert engine._classificar_crime("?") == "OUTROS"

def test_qualificacao_fluxo():
    engine = MotorSafeDriver(persistencia=False)
    df = pd.DataFrame({
        'LATITUDE':['-23.5'],
        'LONGITUDE':['-46.6'],
        'DATA_OCORRENCIA_BO':[pd.Timestamp(datetime.now().date())],
        'NATUREZA_APURADA':['ROUBO DE VEICULO'],
        'NUM_BO':['1'],
        'DESCR_TIPOLOCAL':['VIA PUBLICA']
    })
    t, r = engine._qualificar(df)
    assert len(t) == 1
    assert len(r) == 1

def test_modo_reset():
    if os.path.exists('datalake/metadata/baseline.lock'):
        os.remove('datalake/metadata/baseline.lock')
    engine = MotorSafeDriver(persistencia=False)
    assert engine.auditoria["modo"] == "HARD_RESET"
