import pytest
import json
from pathlib import Path

def test_cloud_pipeline_metrics():
    manifesto = Path("datalake/auditoria/auditoria_pipeline.json")
    
 
    assert manifesto.exists(), "Falha Crítica: O Manifesto de auditoria não foi gerado pelo pipeline!"
    
    with open(manifesto, "r") as f:
        log = json.load(f)
        stats = log.get("auditoria_estatistica", {})
        
    
        assert stats.get("r2_teste", 0) > 0.40, f"Reprovado: O modelo perdeu a capacidade de previsão (R² de teste baixo: {stats.get('r2_teste')})"
        
     
        assert stats.get("degradacao_overfitting", 1) < 0.20, f"Reprovado: Overfitting detetado! Degradação de {stats.get('degradacao_overfitting'):.1%}"
        
       
        assert log.get("linhas_processadas", 0) > 0, "Reprovado: O Datalake processou um volume vazio de dados."
        
     
        hashes = log.get("seguranca_antifraude", {})
        assert len(hashes) > 0, "Reprovado Falha de Segurança: As impressões digitais (SHA-256) dos ficheiros brutos não foram geradas!"
