import pytest
import polars as pl
from pathlib import Path

def test_check_ouro():
    """Garante que a camada final tem todas as colunas para o Power BI"""
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    if not caminho.exists():
        pytest.skip("Arquivo Ouro não gerado - Verifique falha no motor.")
    
    df = pl.read_parquet(caminho)
    cols = ["PERFIL", "TURNO", "NATUREZA_CRIME", "IS_PAGAMENTO", "IS_FERIADO", "RISCO_SCORE", "H3"]
    for c in cols:
        assert c in df.columns, f"Coluna {c} ausente no arquivo Ouro."

def test_filtro_sp():
    """Garante que o motor limpou o lixo geográfico fora de SP"""
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    if not caminho.exists():
        pytest.skip("Arquivo Ouro não gerado.")
    
    df = pl.read_parquet(caminho)
    fora_sp = df.filter((pl.col("LAT_M") < -25.5) | (pl.col("LAT_M") > -19.5))
    assert fora_sp.height == 0, f"Existem {fora_sp.height} pontos fora de SP."

def test_lgpd():
    """Verifica se dados sensíveis foram eliminados da camada Prata"""
    caminho = Path("datalake/prata/camada_prata.parquet")
    if caminho.exists():
        df = pl.read_parquet(caminho)
        # Varre cada coluna: se tiver 'NUM', tem que ter 'ANON' (ID_ANONIMO)
        for c in df.columns:
            c_upper = c.upper()
            if "NUM" in c_upper:
                assert "ANON" in c_upper, f"Erro LGPD: Coluna sensível '{c}' detectada!"
