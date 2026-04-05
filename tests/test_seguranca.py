import pytest
import hashlib
from pathlib import Path

def test_integridade_ouro():
    csv = Path("datalake/ouro/base_looker.csv")
    sha = Path("datalake/ouro/base_looker.sha256")
    
    assert csv.exists()
    assert sha.exists()
    
    with open(csv, "rb") as f:
        check = hashlib.sha256(f.read()).hexdigest()
    with open(sha, "r") as f:
        original = f.read().strip()
        
    assert check == original

def test_modelo_shap():
    assert Path("documentacao/explicabilidade_ia.png").exists()
