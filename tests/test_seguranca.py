import pytest
import json
from pathlib import Path

def test_cloud_pipeline_metrics():
    manifesto = Path("datalake/auditoria/auditoria_pipeline.json")
    
    # Valida se o pipeline chegou até ao fim e gerou o manifesto
    assert manifesto.exists(), "Falha Crítica: O Manifesto de auditoria não foi gerado pelo pipeline!"
    
    with open(manifesto, "r") as f:
        log = json.load(f)
        stats = log.get("auditoria_estatistica", {})
        
        # 1. Valida capacidade de generalização (Previsão do futuro)
        assert stats.get("r2_teste", 0) > 0.40, f"Reprovado: O modelo perdeu a capacidade de previsão (R² de teste baixo: {stats.get('r2_teste')})"
        
        # 2. Valida a trava de Overfitting (Diferença entre o R² de Treino e o R² de Teste)
        assert stats.get("degradacao_overfitting", 1) < 0.20, f"Reprovado: Overfitting Crítico detetado! Degradação de {stats.get('degradacao_overfitting'):.1%}"
        
        # 3. Valida a saúde da ingestão de dados na camada Ouro
        assert log.get("linhas_processadas", 0) > 0, "Reprovado: O Datalake processou um volume vazio de dados."
