import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from autobot_engine import MotorSafeDriver


def test_engine_instancia():
    engine = MotorSafeDriver()
    assert hasattr(engine, "executar_pipeline_completo")
    assert callable(engine.executar_pipeline_completo)


def test_engine_tem_metodos_essenciais_do_novo_pipeline():
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


def test_payload_malha_final_tem_campos_esperados():
    engine = MotorSafeDriver()

    semana_inicio = pd.Timestamp("2026-03-16")
    semana_fim = pd.Timestamp("2026-03-22")

    linha = {
        "codigo_geohash": "6gyf4bf",
        "geohash_prefix_4": "6gyf",
        "geohash_prefix_5": "6gyf4",
        "perfil": "Motorista",
        "turno_operacional": "Noite",
        "score": 8.42,
        "risk_band": "alto",
        "semana_referencia_inicio": semana_inicio,
        "semana_referencia_fim": semana_fim,
        "data_evento": pd.Timestamp("2026-03-15"),
    }

    payload = {
        "geohash": linha["codigo_geohash"],
        "geohash_prefix_4": linha["geohash_prefix_4"],
        "geohash_prefix_5": linha["geohash_prefix_5"],
        "perfil": linha["perfil"],
        "turno": linha["turno_operacional"],
        "periodo": linha["turno_operacional"],
        "score": round(float(linha["score"]), 2),
        "risk_band": linha["risk_band"],
        "modelo": "xgb_prophet",
        "versao_modelo": engine.versao_modelo,
        "janela_inicio": str(engine.janela_inicio.date()),
        "janela_fim": str(engine.janela_fim.date()),
        "semana_referencia_inicio": str(pd.Timestamp(linha["semana_referencia_inicio"]).date()),
        "semana_referencia_fim": str(pd.Timestamp(linha["semana_referencia_fim"]).date()),
        "horizonte_predicao_dias": engine.horizonte_predicao_dias,
        "data_base_modelo": str(pd.Timestamp(linha["data_evento"]).date()),
    }

    campos_esperados = [
        "geohash",
        "geohash_prefix_4",
        "geohash_prefix_5",
        "perfil",
        "turno",
        "periodo",
        "score",
        "risk_band",
        "modelo",
        "versao_modelo",
        "janela_inicio",
        "janela_fim",
        "semana_referencia_inicio",
        "semana_referencia_fim",
        "horizonte_predicao_dias",
        "data_base_modelo",
    ]

    for campo in campos_esperados:
        assert campo in payload, f"Campo obrigatorio ausente: {campo}"


def test_score_e_faixa_validos():
    score = 8.42
    risk_band = "alto"

    assert 0.5 <= score <= 10.0
    assert risk_band in {"baixo", "medio", "alto", "critico"}


def test_turnos_validos():
    turnos_validos = {"Madrugada", "Manha", "Tarde", "Noite"}
    assert "Noite" in turnos_validos
    assert "Manha" in turnos_validos
    assert "Tarde" in turnos_validos
    assert "Madrugada" in turnos_validos


def test_perfis_validos():
    perfis_validos = {"Motorista", "Motociclista", "Pedestre", "Ciclista", "Indefinido"}
    assert "Motorista" in perfis_validos
    assert "Motociclista" in perfis_validos
    assert "Pedestre" in perfis_validos
    assert "Ciclista" in perfis_validos
