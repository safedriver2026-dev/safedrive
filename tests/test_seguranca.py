import pytest
import polars as pl
from pathlib import Path


def test_check_ouro_schema():
    """
    Verifica se a camada Ouro está no novo formato H3+:
    - chave H3
    - métricas de crimes reais e escore de risco
    - localização média
    - features agregadas da Prata (proporções, incluindo períodos detalhados e perfis de vítima)
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
        # Proporções de Período Detalhado
        "PROP_MANHA",
        "PROP_TARDE",
        "PROP_NOITE",
        "PROP_MADRUGADA",
        # Proporções de Perfil de Vítima
        "PROP_PATRIMONIO", # Já existia, mas é um tipo de crime, não perfil de vítima. Mantido para compatibilidade.
        "PROP_MOTORISTA",
        "PROP_MOTO", # Corresponde a MOTOCICLISTA
        "PROP_CICLISTA", # Nova proporção
        "PROP_PEDESTRE", # Nova proporção
        "PROP_GERAL_VITIMA", # Nova proporção para 'GERAL'
        # Outras proporções
        "PROP_FERIADO",
        "PROP_PAGAMENTO",
        # Novas features de risco combinadas
        "RISCO_NOITE_PATRIMONIO",
        "RISCO_MOTO_NOITE",
        "RISCO_MOTORISTA_PAGTO",
        # Outras features do H3
        "LAT_STD",
        "LON_STD",
        "CRIMES_POR_ANO",
        "CRIMES_POND_POR_ANO",
    ]

    # Verificando se todas as colunas obrigatórias estão presentes
    assert all(c in df.columns for c in colunas_obrigatorias), \
        f"Colunas ausentes na camada Ouro: {set(colunas_obrigatorias) - set(df.columns)}"


def test_geometria_wkt_se_existir():
    """
    GEOMETRIA_WKT é opcional. Se existir na camada Ouro,
    pelo menos um valor deve ser não-nulo (caso contrário está "toda vazia").
    """
    p = Path("datalake/ouro/dashboard_final.parquet")
    if not p.exists():
        pytest.skip("Arquivo da camada Ouro ausente")

    df = pl.read_parquet(p)
    if "GEOMETRIA_WKT" in df.columns:
        assert df["GEOMETRIA_WKT"].is_not_null().any(), \
            "Coluna GEOMETRIA_WKT existe, mas está totalmente nula."


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
    assert f.height == 0, \
        "Registros fora da faixa de latitude esperada para SP encontrados na camada Ouro."


def test_lgpd():
    """
    Garante que a camada Prata não carrega colunas com 'NUM' (potencial dado sensível)
    e que usamos apenas ID_ANONIMO como identificador.
    """
    p = Path("datalake/prata/camada_prata.parquet")
    if not p.exists():
        pytest.skip("Arquivo da camada Prata ausente")

    df = pl.read_parquet(p)
    assert not any("NUM" in c.upper() for c in df.columns), \
        "Coluna com 'NUM' (potencial dado sensível) encontrada na camada Prata."
    assert "ID_ANONIMO" in df.columns, \
        "Coluna ID_ANONIMO ausente na camada Prata."


def test_auditoria_shap():
    """
    Verifica se a auditoria SHAP está presente e com as colunas básicas.
    """
    p = Path("datalake/ouro/shap_audit.parquet")
    if not p.exists():
        pytest.skip("Arquivo shap_audit ausente")

    df = pl.read_parquet(p)
    assert "VARIAVEL" in df.columns and "GRAU_IMPORTANCIA" in df.columns, \
        "Colunas VARIAVEL ou GRAU_IMPORTANCIA ausentes no shap_audit.parquet."


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
    assert all(c in df.columns for c in colunas_obrigatorias), \
        f"Colunas ausentes na validação do modelo: {set(colunas_obrigatorias) - set(df.columns)}"
