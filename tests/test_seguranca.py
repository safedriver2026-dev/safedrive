import pytest
import json
from pathlib import Path

def test_cloud_pipeline_metrics():
    manifesto = Path("datalake/auditoria/auditoria_pipeline.json")
    assert manifesto.exists(), "Pipeline falhou: Manifesto de auditoria não foi gerado!"
    
    with open(manifesto, "r") as f:
        log = json.load(f)
        
        # Acessa o bloco de estatísticas do manifesto JSON
        stats = log.get("auditoria_estatistica", {})
        
        # 1. Valida capacidade de generalização (Previsão do futuro)
        assert stats.get("r2_teste", 0) > 0.40, f"Reprovado: O modelo perdeu a capacidade de previsão em dados invisíveis (R² baixo: {stats.get('r2_teste')})"
        
        # 2. Valida a trava de Overfitting (Diferença entre o R² de Treino e o R² de Teste)
        assert stats.get("degradacao_overfitting", 1) < 0.20, f"Reprovado: Overfitting Crítico detectado! Degradação de {stats.get('degradacao_overfitting'):.1%}"
        
        # 3. Valida a saúde da ingestão de dados
        assert log.get("linhas_processadas", 0) > 0, "Reprovado: O Datalake processou um arquivo vazio."
