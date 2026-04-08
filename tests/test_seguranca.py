import pytest
import polars as pl
from pathlib import Path

def test_integridade_total_vision():
    caminho_ouro = Path("datalake/ouro/dashboard_final.parquet")
    assert caminho_ouro.exists(), "[FALHA] Arquivo Ouro não gerado."
    
    df = pl.read_parquet(caminho_ouro)
    
    # 1. Validação de Colunas de Inteligência
    colunas_novas = ["PERFIL", "TURNO", "NATUREZA_CRIME", "IS_PAGAMENTO", "RISCO_SCORE"]
    for col in colunas_novas:
        assert col in df.columns, f"[FALHA] Coluna crítica ausente: {col}"

    # 2. Validação de Diversidade de Dados (Garante que o robô não pegou só um tipo de crime)
    assert df.select("PERFIL").n_unique() >= 2, "[FALHA] Baixa diversidade de perfis detectada."
    assert df.select("NATUREZA_CRIME").n_unique() >= 2, "[FALHA] Divisão Patrimônio/Pessoa falhou."

def test_blindagem_lgpd():
    caminho_prata = Path("datalake/prata/camada_prata.parquet")
    if caminho_prata.exists():
        df = pl.read_parquet(caminho_prata)
        # O sistema deve ter destruído o NUM_BO original
        assert "NUM_BO" not in df.columns, "[FALHA CRÍTICA] NUM_BO exposto na camada Prata!"
        assert "ID_ANONIMO" in df.columns, "[FALHA] Chave de anonimização não gerada."

def test_cobertura_historica():
    caminho_raw = list(Path("datalake/raw").glob("*.parquet"))
    # Verificamos se temos arquivos de múltiplos anos (mínimo 3 anos para ser confiável)
    assert len(caminho_raw) >= 3, f"[FALHA] Base histórica insuficiente. Encontrados apenas {len(caminho_raw)} anos."

def test_consistencia_geografica():
    caminho_ouro = Path("datalake/ouro/dashboard_final.parquet")
    df = pl.read_parquet(caminho_ouro)
    # Garante que todos os pontos possuem um índice H3 (vizinhança)
    assert df.filter(pl.col("H3").is_null()).height == 0, "[FALHA] Existem pontos sem indexação H3."
