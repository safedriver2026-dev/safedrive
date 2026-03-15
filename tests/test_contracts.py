import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path: sys.path.insert(0, BASE_DIR)

from autobot.autobot_engine import MotorSafeDriver

def test_engine_instancia():
    engine = MotorSafeDriver(habilitar_firestore=False)
    assert hasattr(engine, "executar_pipeline_completo")

def test_engine_metodos():
    engine = MotorSafeDriver(habilitar_firestore=False)
    for m in ["_verificar_atualizacao", "_construir_raw_operacional", "_features", "_treinar", "_runbook", "_sync"]:
        assert hasattr(engine, m)

def test_turno():
    engine = MotorSafeDriver(habilitar_firestore=False)
    assert engine._turno("02:30") == "Madrugada"
    assert engine._turno("08:15") == "Manha"
    assert engine._turno("Invalido") == "Noite"

def test_janela():
    engine = MotorSafeDriver(habilitar_firestore=False)
    assert (engine.data_execucao - engine.janela_inicio).days == 730

def test_pastas():
    MotorSafeDriver(habilitar_firestore=False)
    for p in ["reports", "metadata", "raw", "trusted", "refined"]:
        assert os.path.exists(f"datalake/{p}")

def test_schema_resilience():
    engine = MotorSafeDriver(habilitar_firestore=False)
    df = pd.DataFrame({"LATITUDE": ["-23.5"], "LONGITUDE": ["-46.6"], "BATATA": ["1"]})
    df_c = engine._construir_raw_operacional(df, 2026)
    assert "NATUREZA_APURADA" in df_c.columns
    assert "BATATA" not in df_c.columns
    assert any(t in df_c["NATUREZA_APURADA"].dtype.name for t in ['object', 'string', 'str'])
    assert any(t in df_c["DESCR_TIPOLOCAL"].dtype.name for t in ['object', 'string', 'str'])

def test_higienizacao():
    engine = MotorSafeDriver(habilitar_firestore=False)
    assert engine._higienizar_texto("São Paulo") == "SAO PAULO"
    assert engine._higienizar_texto(np.nan) == ""
