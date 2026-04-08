import pytest
import polars as pl
from pathlib import Path

def test_check_ouro():
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    assert caminho.exists()
    df = pl.read_parquet(caminho)
    # Verifica colunas essenciais
    cols = ["PERFIL", "TURNO", "NATUREZA_CRIME", "IS_PAGAMENTO", "RISCO_SCORE", "H3"]
    for c in cols:
        assert c in df.columns

def test_filtro_sp():
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    df = pl.read_parquet(caminho)
    # Nada pode estar fora da caixa de SP
    fora = df.filter((pl.col("LAT_M") < -25.5) | (pl.col("LAT_M") > -19.5))
    assert fora.height == 0

def test_lgpd():
    caminho = Path("datalake/prata/camada_prata.parquet")
    if caminho.exists():
        df = pl.read_parquet(caminho)
        # Verifica se o NUM_BO foi removido
        for c in df.columns:
            assert "NUM" not in c.upper() or "ANON" in c.upper()
