import hashlib
import json
from pathlib import Path

def test_verificar_selos_digitais():
    controle_path = Path("datalake/auditoria/controle_integridade.json")
    ia_path = Path("datalake/ouro/predicao_risco_mapa.csv")
    detalhes_path = Path("datalake/ouro/crimes_detalhados.csv")
    
    assert controle_path.exists()
    with open(controle_path, "r") as f: controle = json.load(f)
    
    def calcular(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(4096), b""): h.update(b)
        return h.hexdigest()
    
    assert calcular(ia_path) == controle.get("selo_ia")
    assert calcular(detalhes_path) == controle.get("selo_detalhes")

def test_validar_chaves_unicas():
    import pandas as pd
    df = pd.read_csv("datalake/ouro/crimes_detalhados.csv")
    assert df['num_bo'].is_unique
