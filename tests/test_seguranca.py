import pytest
import polars as pl
from pathlib import Path

def test_check_ouro():
    """Garante que a camada Ouro tem todas as colunas necessárias para o Looker Studio."""
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    if not caminho.exists():
        pytest.skip("Arquivo Ouro não encontrado. Ignorando teste (possível execução sem novos dados CDC).")
    
    df = pl.read_parquet(caminho)
    cols_esperadas = ["PERFIL", "TURNO", "NATUREZA_CRIME", "IS_PAGAMENTO", "IS_FERIADO", "RISCO_SCORE", "H3", "GEOMETRIA_WKT"]
    
    for c in cols_esperadas:
        assert c in df.columns, f"Erro Crítico de Esquema: Coluna '{c}' está ausente."

def test_filtro_sp():
    """Garante que o motor eliminou dados geográficos inválidos (fora da Box de São Paulo)."""
    caminho = Path("datalake/ouro/dashboard_final.parquet")
    if not caminho.exists():
        pytest.skip("Arquivo Ouro não encontrado.")
    
    df = pl.read_parquet(caminho)
    fora_sp = df.filter((pl.col("LAT_M") < -25.5) | (pl.col("LAT_M") > -19.5))
    
    assert fora_sp.height == 0, f"Falha de Qualidade: Detectados {fora_sp.height} registos fora das coordenadas alvo."

def test_lgpd():
    """Verifica a conformidade com a proteção de dados (Privacy by Design) na camada Prata."""
    caminho = Path("datalake/prata/camada_prata.parquet")
    if not caminho.exists():
        pytest.skip("Camada Prata não encontrada.")
    
    df = pl.read_parquet(caminho)
    for c in df.columns:
        c_upper = c.upper()
        # Se a coluna contiver 'NUM' (ex: NUM_BO), deve obrigatoriamente conter 'ANON' (ex: ID_ANONIMO)
        if "NUM" in c_upper:
            assert "ANON" in c_upper, f"Risco de Fuga de Dados (PII): Coluna sensível '{c}' detetada!"

def test_auditoria_shap():
    """Valida se o motor gerou o log de explicabilidade do modelo Ensemble."""
    caminho = Path("datalake/ouro/shap_audit.parquet")
    if not caminho.exists():
        pytest.skip("Log de Auditoria SHAP não encontrado.")
    
    df = pl.read_parquet(caminho)
    
    assert "FEATURE" in df.columns and "IMPORTANCIA" in df.columns, "Esquema do ficheiro SHAP está incorreto."
    assert df.height > 0, "Falha na Auditoria: O log SHAP está vazio."
