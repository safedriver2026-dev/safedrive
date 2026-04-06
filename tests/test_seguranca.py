import pytest
import pandas as pd
import json
from pathlib import Path

def test_integridade_e_performance():
    manifesto = Path("datalake/auditoria/controle_integridade.json")
    base_dash = Path("datalake/ouro/dashboard_risco_consolidado.csv")
    assert manifesto.exists()
    assert base_dash.exists()
    with open(manifesto, "r") as f:
        log = json.load(f)
        assert log["r2"] > 0.60
        assert len(log["sha256"]) == 64

def test_schema_dashboard():
    df = pd.read_csv("datalake/ouro/dashboard_risco_consolidado.csv")
    for col in ['H3_INDEX', 'PREDICAO_RISCO', 'TURNO']:
        assert col in df.columns
