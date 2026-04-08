import pytest
import polars as pl
from pathlib import Path

def test_validacao_esquema_ouro():
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    assert caminho.exists()
    df = pl.read_parquet(caminho)
    colunas = ["PERFIL", "TURNO", "NATUREZA_CRIME", "IS_PAGAMENTO", "RISCO_SCORE", "H3"]
    for col in colunas:
        assert col in df.columns

def test_privacidade_dados():
    caminho = Path("datalake/prata/camada_prata.parquet")
    if caminho.exists():
        df = pl.read_parquet(caminho)
        assert "NUM_BO" not in df.columns
        assert "ID_ANONIMO" in df.columns

def test_volume_historico():
    arquivos = list(Path("datalake/raw").glob("*.parquet"))
    assert len(arquivos) >= 3
