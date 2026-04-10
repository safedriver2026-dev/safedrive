import pytest, polars as pl
from pathlib import Path


def test_check_ouro():
    p = Path("datalake/ouro/dashboard_final.parquet")
    if not p.exists():
        pytest.skip("Arquivo ausente")
    df = pl.read_parquet(p)
    assert all(
        c in df.columns
        for c in ["ESCORE_RISCO", "GEOMETRIA_WKT", "CODIGO_H3", "PERFIL_VITIMA"]
    )


def test_filtro_sp():
    p = Path("datalake/ouro/dashboard_final.parquet")
    if not p.exists():
        pytest.skip("Arquivo ausente")
    df = pl.read_parquet(p)
    f = df.filter(
        (pl.col("LATITUDE_MEDIA") < -25.5) | (pl.col("LATITUDE_MEDIA") > -19.5)
    )
    assert f.height == 0


def test_lgpd():
    p = Path("datalake/prata/camada_prata.parquet")
    if not p.exists():
        pytest.skip("Arquivo ausente")
    df = pl.read_parquet(p)
    assert not any("NUM" in c.upper() for c in df.columns)
    assert "ID_ANONIMO" in df.columns


def test_auditoria_shap():
    p = Path("datalake/ouro/shap_audit.parquet")
    if not p.exists():
        pytest.skip("Arquivo ausente")
    df = pl.read_parquet(p)
    assert "VARIAVEL" in df.columns and "GRAU_IMPORTANCIA" in df.columns


def test_validacao_real_previsto():
    p = Path("datalake/ouro/validacao_modelo.parquet")
    if not p.exists():
        pytest.skip("Arquivo de validação ausente")
    df = pl.read_parquet(p)
    assert "CRIMES_REAIS" in df.columns
    assert "ESCORE_RISCO" in df.columns
    assert "CODIGO_H3" in df.columns
