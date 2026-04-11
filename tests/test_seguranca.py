import pytest
import polars as pl
from pathlib import Path


def test_check_ouro_schema():
    """
    Verifica se a camada Ouro está no novo formato H3+:
    - chave H3
    - métricas de crimes reais e escore de risco
    - localização média
    - features agregadas da Prata (proporções)
    """
    p = Path("datalake/ouro/dashboard_final.parquet")
    if not p.exists():
        pytest.skip("Arquivo da camada Ouro ausente")

    df = pl.read_parquet(p)

    colunas_obrigatorias = [
        "CODIGO_H3",
        "CRIMES_REAIS",
        "ESCORE_RISCO",
        "LATITUDE_MEDIA",
        "LONGITUDE_MEDIA",
        "PROP_NOITE",
        "PROP_PATRIMONIO",
        "PROP_MOTORISTA",
        "PROP_MOTO",
        "PROP_FERIADO",
        "PROP_PAGAMENTO",
    ]

    assert all(c in df.columns for c in colunas_obrigatorias)


def test_geometria_wkt_se_existir():
    """
    GEOMETRIA_WKT é opcional. Se existir na camada Ouro,
    pelo menos um valor deve ser não-nulo (caso contrário está "toda vazia").
    """
    p = Path("datalake/ouro/dashboard_final.parquet")
    if not p.exists():
        pytest.skip("Arquivo da camada Ouro ausente")

    df = pl.read_parquet(p)

    if "GEOMETRIA_WKT" not in df.columns:
        pytest.skip("GEOMETRIA_WKT ausente (uso opcional).")

    non_null = df.select(pl.col("GEOMETRIA_WKT").is_not_null().sum())[0, 0]
    assert non_null > 0, "Coluna GEOMETRIA_WKT presente, mas totalmente nula."


def test_filtro_sp():
    """
    Garante que a camada Ouro só contém LATITUDE_MEDIA dentro da faixa esperada para SP.
    """
    p = Path("datalake/ouro/dashboard_final.parquet")
    if not p.exists():
        pytest.skip("Arquivo da camada Ouro ausente")

    df = pl.read_parquet(p)
    f = df.filter(
        (pl.col("LATITUDE_MEDIA") < -25.5) | (pl.col("LATITUDE_MEDIA") > -19.5)
    )
    assert f.height == 0


def test_lgpd():
    """
    Garante que a camada Prata não carrega colunas com 'NUM' (potencial dado sensível)
    e que usamos apenas ID_ANONIMO como identificador.
    """
    p = Path("datalake/prata/camada_prata.parquet")
    if not p.exists():
        pytest.skip("Arquivo da camada Prata ausente")

    df = pl.read_parquet(p)
    assert not any("NUM" in c.upper() for c in df.columns)
    assert "ID_ANONIMO" in df.columns


def test_auditoria_shap():
    """
    Verifica se a auditoria SHAP está presente e com as colunas básicas.
    """
    p = Path("datalake/ouro/shap_audit.parquet")
    if not p.exists():
        pytest.skip("Arquivo shap_audit ausente")

    df = pl.read_parquet(p)
    assert "VARIAVEL" in df.columns and "GRAU_IMPORTANCIA" in df.columns


def test_validacao_real_previsto():
    """
    Verifica se a tabela de validação está coerente com o novo modelo H3+:
    - CODIGO_H3
    - CRIMES_REAIS
    - ESCORE_RISCO
    - ERRO_ABS (erro absoluto)
    """
    p = Path("datalake/ouro/validacao_modelo.parquet")
    if not p.exists():
        pytest.skip("Arquivo de validação ausente")

    df = pl.read_parquet(p)
    colunas_obrigatorias = ["CRIMES_REAIS", "ESCORE_RISCO", "CODIGO_H3", "ERRO_ABS"]
    assert all(c in df.columns for c in colunas_obrigatorias)
