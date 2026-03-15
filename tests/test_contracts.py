import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from autobot.autobot_engine import MotorSafeDriver
from autobot.config import COLUNAS_REFINED_EVENTOS

def test_engine_instancia():
    engine = MotorSafeDriver(habilitar_firestore=False)
    assert hasattr(engine, "executar_pipeline_completo")

def test_metodos_obrigatorios():
    engine = MotorSafeDriver(habilitar_firestore=False)
    metodos = ["_ingerir_fonte", "_qualificar_dados", "_ml_features", "_treinar_disseminar", "_notificar_discord", "_gerar_runbook"]
    for m in metodos:
        assert hasattr(engine, m)

def test_resiliencia_tipagem_e_limpeza():
    engine = MotorSafeDriver(habilitar_firestore=False)
    data = {
        'NUM_BO': ['1', '2', '3'],
        'HORA_OCORRENCIA_BO': ['10:00', '11:00', '12:00'],
        'DESCR_TIPOLOCAL': ['VIA PUBLICA', 'VIA PUBLICA', 'VIA PUBLICA'],
        'DESCR_SUBTIPOLOCAL': ['RUA', 'RUA', 'RUA'],
        'LATITUDE': ['-23.5', '0', '-'], 
        'LONGITUDE': ['-46.6', '0', '-'], 
        'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-01-01')]*3,
        'NATUREZA_APURADA': ['ROUBO DE VEICULO', 'ROUBO DE VEICULO', 'ROUBO DE VEICULO']
    }
    df = pd.DataFrame(data)
    dt, dr = engine._qualificar_dados(df, 2026)
    
    assert len(dt) == 1
    assert dt['LATITUDE'].iloc[0] == -23.5
    for col in COLUNAS_REFINED_EVENTOS:
        assert col in dr.columns

def test_normalizacao_turnos():
    engine = MotorSafeDriver(habilitar_firestore=False)
    df_teste = pd.DataFrame({
        'LATITUDE': [-23.5], 'LONGITUDE': [-46.6], 
        'HORA_OCORRENCIA_BO': ['08:15'], 'NATUREZA_APURADA': ['ROUBO DE VEICULO'],
        'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-01-01')]
    })
    pnl, prj = engine._gerar_inteligencia(df_teste)
    assert not pnl.empty
    assert pnl['turno'].iloc[0] == 'Manha'

def test_modo_checkpointing():
    # Testa se o lock_file determina o modo de execução
    if os.path.exists('datalake/metadata/baseline.lock'):
        os.remove('datalake/metadata/baseline.lock')
    e_full = MotorSafeDriver(habilitar_firestore=False)
    assert e_full.auditoria["modo"] == "FULL_RELOAD"
    
    with open('datalake/metadata/baseline.lock', 'w') as f: f.write("lock")
    e_inc = MotorSafeDriver(habilitar_firestore=False)
    assert e_inc.auditoria["modo"] == "INCREMENTAL"
