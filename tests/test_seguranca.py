import pytest
import polars as pl
from pathlib import Path

def test_esquema_dados_ouro():
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    assert caminho.exists()
    df = pl.read_parquet(caminho)
    # Garante que os novos filtros existem para o Power BI
    cols = ["PERFIL", "TURNO", "NATUREZA_CRIME", "IS_PAGAMENTO", "IS_FERIADO", "RISCO_SCORE", "H3"]
    for col in cols:
        assert col in df.columns

def test_validacao_espacial():
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    df = pl.read_parquet(caminho)
    # Garante que nenhum ponto está fora de SP (Bounding Box)
    assert df.filter((pl.col("LAT_M") < -25.5) | (pl.col("LAT_M") > -19.5)).height == 0
    assert df.filter((pl.col("LON_M") < -53.5) | (pl.col("LON_M") > -44.0)).height == 0

def test_anonimizacao():
    caminho = Path("datalake/prata/camada_prata.parquet")
    if caminho.exists():
        df = pl.read_parquet(caminho)
        # Nomes proibidos de colunas originais
        assert "NUM_BO" not in df.columns
        assert "NOME_VITIMA" not in df.columns
