import pytest
import polars as pl
from pathlib import Path

def test_check_arquivos_ouro():
    # Garante que o arquivo final para o Power BI foi gerado
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    assert caminho.exists(), "Erro: Arquivo Ouro nao encontrado."
    
    df = pl.read_parquet(caminho)
    # Verifica se as colunas essenciais de inteligencia estao la
    cols_obrigatorias = ["PERFIL", "TURNO", "NATUREZA_CRIME", "IS_PAGAMENTO", "RISCO_SCORE", "H3"]
    for col in cols_obrigatorias:
        assert col in df.columns, f"Erro: Coluna {col} ausente no arquivo final."

def test_verificar_geografia_sp():
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    df = pl.read_parquet(caminho)
    # Garante que o filtro de perimetro funcionou (nada fora de SP)
    fora_sp = df.filter((pl.col("LAT_M") < -25.5) | (pl.col("LAT_M") > -19.5))
    assert fora_sp.height == 0, "Erro: Existem coordenadas fora do estado de SP."

def test_privacidade_lgpd():
    caminho_prata = Path("datalake/prata/camada_prata.parquet")
    if caminho_prata.exists():
        df = pl.read_parquet(caminho_prata)
        # O sistema nunca deve subir o numero do BO ou dados identificaveis
        for col in df.columns:
            assert "NUM" not in col.upper() or "ANON" in col.upper(), f"Erro: Dado sensivel exposto na coluna {col}"
