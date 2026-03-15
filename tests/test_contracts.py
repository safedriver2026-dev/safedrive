import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

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
    metodos = ["_ingerir_fonte", "_qualificar_dados", "_gerar_inteligencia", "_treinar_disseminar", "_notificar_discord", "_gerar_runbook", "_formatar_data_pt"]
    for m in metodos:
        assert hasattr(engine, m)

def test_formatador_data_pt():
    engine = MotorSafeDriver(habilitar_firestore=False)
    dt_teste = datetime(2026, 3, 15, 19, 15)
    resultado = engine._formatar_data_pt(dt_teste)
    assert "15 de março de 2026 às 19:15" in resultado

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

def test_normalizacao_turnos_e_prophet_safe():
    engine = MotorSafeDriver(habilitar_firestore=False)
    df_teste = pd.DataFrame({
        'LATITUDE': [-23.5, -23.6], 'LONGITUDE': [-46.6, -46.7], 
        'HORA_OCORRENCIA_BO': ['08:15', '22:30'], 'NATUREZA_APURADA': ['ROUBO DE VEICULO', 'FURTO DE VEICULO'],
        'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-01-01'), pd.Timestamp('2026-01-02')],
        'NUM_BO': ['1', '2']
    })
    pnl, prj = engine._gerar_inteligencia(df_teste)
    assert not pnl.empty
    assert 'Manha' in pnl['periodo_dia'].values
    assert 'Noite' in pnl['periodo_dia'].values

def test_modo_checkpointing():
    if os.path.exists('datalake/metadata/baseline.lock'):
        os.remove('datalake/metadata/baseline.lock')
    e_full = MotorSafeDriver(habilitar_firestore=False)
    assert e_full.auditoria["modo"] == "FULL_RELOAD"
    
    with open('datalake/metadata/baseline.lock', 'w') as f: f.write("lock")
    e_inc = MotorSafeDriver(habilitar_firestore=False)
    assert e_inc.auditoria["modo"] == "INCREMENTAL"
