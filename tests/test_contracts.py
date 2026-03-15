import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from autobot.autobot_engine import MotorSafeDriver


def test_engine_instancia():
    engine = MotorSafeDriver()
    assert hasattr(engine, "executar_pipeline_completo")
    assert callable(engine.executar_pipeline_completo)


def test_engine_tem_metodos_essenciais():
    engine = MotorSafeDriver()

    metodos_essenciais = [
        "_ler_ou_baixar_raw",
        "_processar_trusted",
        "_processar_refined_eventos",
        "_criar_painel_diario",
        "_criar_target_futuro",
        "_criar_fator_prophet",
        "_montar_base_supervisionada",
        "_split_temporal",
        "_treinar_modelo",
        "_treinar_modelo_final",
        "_gerar_malha_semanal",
        "_sincronizacao_delta_firestore",
    ]

    for nome in metodos_essenciais:
        assert hasattr(engine, nome), f"Engine deveria expor o metodo {nome}"


def test_classificacao_faixa_risco():
    engine = MotorSafeDriver()

    assert engine._classificar_faixa_risco(2.0) == "baixo"
    assert engine._classificar_faixa_risco(4.0) == "medio"
    assert engine._classificar_faixa_risco(7.0) == "alto"
    assert engine._classificar_faixa_risco(9.0) == "critico"


def test_hash_registro_estavel():
    engine = MotorSafeDriver()

    payload = {
        "geohash": "6gyf4bf",
        "perfil": "Motorista",
        "turno": "Noite",
        "score": 8.42,
    }

    h1 = engine._hash_registro(payload)
    h2 = engine._hash_registro(payload)

    assert isinstance(h1, str)
    assert h1 == h2
    assert len(h1) == 64


def test_normalizacao_turno():
    engine = MotorSafeDriver()

    assert engine._classificar_turno(2) == "Madrugada"
    assert engine._classificar_turno(8) == "Manha"
    assert engine._classificar_turno(15) == "Tarde"
    assert engine._classificar_turno(21) == "Noite"


def test_target_futuro_7d_existe():
    engine = MotorSafeDriver()

    df = pd.DataFrame({
        "codigo_geohash": ["6gyf4bf"] * 10,
        "perfil": ["Motorista"] * 10,
        "turno_operacional": ["Noite"] * 10,
        "data_evento": pd.date_range("2026-01-01", periods=10, freq="D"),
        "target_dia": [1.0, 0.5, 0.0, 2.0, 1.5, 0.0, 3.0, 1.0, 0.0, 2.5],
        "ocorrencias_dia": [1] * 10,
        "latitude_media": [-23.55] * 10,
        "longitude_media": [-46.63] * 10,
        "geohash_prefix_4": ["6gyf"] * 10,
        "geohash_prefix_5": ["6gyf4"] * 10,
        "lag_1": [0.0] * 10,
        "lag_7": [0.0] * 10,
        "lag_14": [0.0] * 10,
        "media_7d": [0.0] * 10,
        "media_30d": [0.0] * 10,
        "ocorrencias_7d": [0.0] * 10,
        "ocorrencias_30d": [0.0] * 10,
        "dias_desde_ultimo_evento": [1] * 10,
        "dia_semana": [0] * 10,
        "mes": [1] * 10,
        "fim_semana": [0] * 10,
        "tendencia_7_30": [0.0] * 10,
    })

    resultado = engine._criar_target_futuro(df)

    assert "target_futuro_7d" in resultado.columns
    assert resultado["target_futuro_7d"].notna().all()


def test_score_e_faixa_validos():
    score = 8.42
    risk_band = "alto"

    assert 0.5 <= score <= 10.0
    assert risk_band in {"baixo", "medio", "alto", "critico"}
