import hashlib
import json
import pandas as pd
from pathlib import Path

def test_validar_assinaturas_auditoria():
    manifesto_path = Path("datalake/auditoria/manifesto.json")
    assert manifesto_path.exists()
    
    with open(manifesto_path, "r") as f:
        manifesto = json.load(f)
    
    ouro_path = Path("datalake/ouro/base_final_looker.csv")
    assert ouro_path.exists()
    
    sha256 = hashlib.sha256()
    with open(ouro_path, "rb") as f:
        for bloco in iter(lambda: f.read(4096), b""): sha256.update(bloco)
    
    assert sha256.hexdigest() == manifesto.get("camada_ouro")

def test_verificar_dados_ia():
    df = pd.read_csv("datalake/ouro/base_final_looker.csv")
    colunas = ['h3_index', 'score_risco', 'influencia_perfil_idx', 'influencia_periodo_idx']
    for col in colunas:
        assert col in df.columns
