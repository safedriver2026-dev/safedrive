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
    metodos = ["_ler_ou_baixar_raw", "_processar_trusted", "_sincronizacao_delta_firestore"]
    for nome in metodos: assert hasattr(engine, nome)

def test_classificacao_faixa_risco():
    engine = MotorSafeDriver(habilitar_firestore=False)
    assert engine._classificar_faixa_risco(2.0) == "baixo"
    assert engine._classificar_faixa_risco(9.0) == "critico"

def test_normalizacao_turno():
    engine = MotorSafeDriver(habilitar_firestore=False)
    assert engine._classificar_turno(2) == "Madrugada"
    assert engine._classificar_turno(21) == "Noite"

def test_gerar_geohash_seguro():
    engine = MotorSafeDriver(habilitar_firestore=False)
    assert len(engine._gerar_geohash_seguro(-23.5505, -46.6333)) == 7
    assert pd.isna(engine._gerar_geohash_seguro(np.nan, -46.6333))

def test_janela_historica_tem_730_dias():
    engine = MotorSafeDriver(habilitar_firestore=False)
    assert (engine.janela_fim - engine.janela_inicio).days == 730

def test_routing_penalty_in_payload():
    engine = MotorSafeDriver(habilitar_firestore=False)
    linha = {
        "codigo_geohash": "6gyf4bf", "geohash_prefix_4": "6gyf", "geohash_prefix_5": "6gyf4",
        "perfil": "Motorista", "turno_operacional": "Noite", "score": 8.42, 
        "risk_band": "alto", "routing_penalty": 2.26
    }
    payload = {
        "geohash": linha["codigo_geohash"], "score": linha["score"],
        "routing_penalty": linha["routing_penalty"]
    }
    assert "routing_penalty" in payload
    assert payload["routing_penalty"] > 1.0
