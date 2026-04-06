import pytest
import json
from pathlib import Path

def test_cloud_pipeline_metrics():
    manifesto = Path("datalake/auditoria/auditoria_pipeline.json")
    
    assert manifesto.exists(), "Pipeline falhou: Manifesto de auditoria não foi gerado!"
    
    with open(manifesto, "r") as f:
        log = json.load(f)
        stats = log.get("auditoria_estatistica", {})
        
        assert stats.get("r2_teste", 0) > 0.40, f"R² de teste insuficiente: {stats.get('r2_teste')}"
        assert stats.get("degradacao_overfitting", 1) < 0.20, f"Overfitting detectado: {stats.get('degradacao_overfitting'):.1%}"
        assert log.get("linhas_processadas", 0) > 0, "Datalake vazio!"
