import pytest
import polars as pl
from pathlib import Path

def test_check_ouro():
    """Garante que a camada final tem todas as colunas para o Power BI"""
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    assert caminho.exists(), "Erro: Arquivo Ouro não foi gerado."
    
    df = pl.read_parquet(caminho)
    # Lista de colunas que o seu Dashboard vai usar
    cols = ["PERFIL", "TURNO", "NATUREZA_CRIME", "IS_PAGAMENTO", "IS_FERIADO", "RISCO_SCORE", "H3"]
    for c in cols:
        assert c in df.columns, f"Erro: Coluna {c} ausente no arquivo Ouro."

def test_filtro_sp():
    """Garante que o motor limpou o lixo geográfico fora de SP"""
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    df = pl.read_parquet(caminho)
    # Verifica se há pontos fora da caixa delimitadora de São Paulo
    fora_sp = df.filter((pl.col("LAT_M") < -25.5) | (pl.col("LAT_M") > -19.5))
    assert fora_sp.height == 0, f"Erro: Detectados {fora_sp.height} pontos fora de SP."

def test_lgpd():
    """Verifica se dados sensíveis (PII) foram eliminados da camada Prata"""
    caminho = Path("datalake/prata/camada_prata.parquet")
    if caminho.exists():
        df = pl.read_parquet(caminho)
        # O teste varre cada coluna. Se achar 'NUM' (de NUM_BO), 
        # ela precisa ter também 'ANON' (do ID_ANONIMO), caso contrário é vazamento.
        for c in df.columns:
            c_upper = c.upper()
            if "NUM" in c_upper:
                assert "ANON" in c_upper, f"Erro LGPD: Coluna identificável '{c}' encontrada!"
