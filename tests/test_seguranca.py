import pytest
import polars as pl
from pathlib import Path

def test_check_ouro():
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    if not caminho.exists():
        pytest.skip("Arquivo Ouro não gerado - Pule este teste.")
    
    df = pl.read_parquet(caminho)
    cols = ["PERFIL", "TURNO", "NATUREZA_CRIME", "IS_PAGAMENTO", "IS_FERIADO", "RISCO_SCORE", "H3", "GEOMETRIA_WKT"]
    for c in cols:
        assert c in df.columns

def test_filtro_sp():
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    if not caminho.exists():
        pytest.skip("Arquivo Ouro não gerado.")
    
    df = pl.read_parquet(caminho)
    fora_sp = df.filter((pl.col("LAT_M") < -25.5) | (pl.col("LAT_M") > -19.5))
    assert fora_sp.height == 0

def test_lgpd():
    caminho = Path("datalake/prata/camada_prata.parquet")
    if caminho.exists():
        df = pl.read_parquet(caminho)
        for c in df.columns:
            c_upper = c.upper()
            if "NUM" in c_upper:
                assert "ANON" in c_upper
