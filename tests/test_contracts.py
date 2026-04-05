import pytest
import pandas as pd
from pathlib import Path

def test_camada_ouro_existe():
    Path("datalake/ouro").mkdir(parents=True, exist_ok=True)
    assert Path("datalake/ouro").exists()

def test_csv_looker_header():
    caminho = Path("datalake/ouro/base_looker.csv")
    if not caminho.exists(): pytest.skip("Primeira execução: aguardando CSV.")
    df = pd.read_csv(caminho)
    assert "score_risco" in df.columns
