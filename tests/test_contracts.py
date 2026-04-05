import pytest
import pandas as pd
import json
from pathlib import Path

def test_infraestrutura_camadas():
    # Verifica se o motor criou as pastas necessárias
    assert Path("datalake/bronze").exists()
    assert Path("datalake/ouro").exists()

def test_contrato_metricas():
    caminho = Path("datalake/ouro/metricas.json")
    if not caminho.exists():
        pytest.skip("Ambiente inicial: arquivos ouro ainda não gerados.")
    with open(caminho, 'r') as f:
        data = json.load(f)
        assert "MAE" in data

def test_contrato_looker():
    caminho = Path("datalake/ouro/base_looker.csv")
    if not caminho.exists():
        pytest.skip("Ambiente inicial: base do Looker ainda não gerada.")
    df = pd.read_csv(caminho)
    assert isinstance(df, pd.DataFrame)
