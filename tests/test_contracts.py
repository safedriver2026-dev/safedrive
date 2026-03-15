import os, sys, pandas as pd, numpy as np
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autobot.autobot_engine import MotorSafeDriver

def test_engine_instancia():
    engine = MotorSafeDriver(habilitar_firestore=False)
    assert hasattr(engine, "executar_pipeline_completo")

def test_normalizacao_semantica_2026():
    engine = MotorSafeDriver(habilitar_firestore=False)
    cols = ['RUBRICA', 'NATUREZA_APURADA', 'LATITUDE', 'LONGITUDE']
    norm_cols = [engine._normalizar_coluna(c) for c in cols]
    df = pd.DataFrame([['A', 'B', '0', '0']], columns=norm_cols)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    assert 'NATUREZA_APURADA' in df.columns
    assert len(df.columns) == 3

def test_qualificacao_coordenadas_sujas():
    engine = MotorSafeDriver(habilitar_firestore=False)
    data = {
        'LATITUDE': ['-23.5', '0', '-', 'NULL'],
        'LONGITUDE': ['-46.6', '0', '-', 'NULL'],
        'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-01-01')]*4,
        'NATUREZA_APURADA': ['ROUBO DE VEICULO']*4,
        'HORA_OCORRENCIA_BO': ['12:00']*4,
        'NUM_BO': ['1','2','3','4'],
        'DESCR_TIPOLOCAL': ['VIA PUBLICA']*4 # ADICIONADO PARA CUMPRIR O CONTRATO
    }
    df = pd.DataFrame(data)
    dt, dr = engine._qualificar_dados(df, 2026)
    assert len(dt) == 1
    assert dt['LATITUDE'].iloc[0] == -23.5
    assert 'DESCR_TIPOLOCAL' in dr.columns

def test_formatador_data_portugues():
    engine = MotorSafeDriver(habilitar_firestore=False)
    dt = datetime(2026, 3, 15, 20, 0)
    res = engine._formatar_data_pt(dt)
    assert "15 de março de 2026 às 20:00" in res

def test_checkpoint_cold_start():
    if os.path.exists('datalake/metadata/baseline.lock'):
        os.remove('datalake/metadata/baseline.lock')
    engine = MotorSafeDriver(habilitar_firestore=False)
    assert engine.auditoria["modo"] == "FULL_RELOAD"
