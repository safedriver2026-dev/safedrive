import pytest
import json
from pathlib import Path

def test_metrics_and_files():
    manifesto = Path("datalake/auditoria/controle_integridade.json")
    base_dash = Path("datalake/ouro/dashboard_risco_real.csv")
    
    assert manifesto.exists(), "Manifesto de auditoria não encontrado!"
    assert base_dash.exists(), "Arquivo do Dashboard não foi gerado!"
    
    with open(manifesto, "r") as f:
        log = json.load(f)
        # O R2 deve estar entre 0.60 (mínimo real) e 0.98 (evita overfitting)
        assert 0.60 <= log["r2"] <= 0.98, f"R2 fora da margem de segurança: {log['r2']}"
        assert len(log["sha256"]) == 64, "Assinatura digital inválida!"
