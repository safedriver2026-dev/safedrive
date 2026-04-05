import pytest
import pandas as pd
import json
from pathlib import Path

def test_metricas_existentes():
    caminho = Path("datalake/ouro/metricas.json")
    assert caminho.exists()
    with open(caminho, 'r') as f:
        data = json.load(f)
        assert "R2" in data
        assert data["R2"] <= 1.0

def test_base_looker_formatacao():
    caminho = Path("datalake/ouro/base_looker.csv")
    assert caminho.exists()
    df = pd.read_csv(caminho)
    assert "score_risco" in df.columns
    assert df["score_risco"].max() <= 100.0
    assert df["score_risco"].min() >= 0.0
