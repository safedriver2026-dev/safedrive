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

def test_engine_tem_metodos_essenciais():
    engine = MotorSafeDriver(habilitar_firestore=False)
    metodos = ["_verificar_atualizacao", "_construir_raw_operacional", "_construir_features_preditivas", "_treinar_modelo", "_gerar_documentacao_runbook"]
    for nome in metodos: assert hasattr(engine, nome)

def test_normalizacao_turno():
    engine = MotorSafeDriver(habilitar_firestore=False)
    assert engine._classificar_turno("02:30") == "Madrugada"
    assert engine._classificar_turno("22:00") == "Noite"

def test_janela_historica_tem_730_dias():
    engine = MotorSafeDriver(habilitar_firestore=False)
    assert (engine.data_execucao - engine.janela_inicio).days == 730

def test_criacao_diretorios():
    MotorSafeDriver(habilitar_firestore=False)
    assert os.path.exists("datalake/reports")
    assert os.path.exists("datalake/metadata")

def test_fallback_schema():
    engine = MotorSafeDriver(habilitar_firestore=False)
    df_falso = pd.DataFrame({"LATITUDE": ["-23.5"], "LONGITUDE": ["-46.6"], "BATATA": ["1"]})
    df_corrigido = engine._construir_raw_operacional(df_falso, 2026)
    assert "NATUREZA_APURADA" in df_corrigido.columns
    assert "BATATA" not in df_corrigido.columns
    assert df_corrigido["NATUREZA_APURADA"].dtype == 'object'
