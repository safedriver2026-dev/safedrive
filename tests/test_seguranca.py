import hashlib
import json
import pandas as pd
from pathlib import Path

def test_auditoria_criptografica():
    manifesto_path = Path("datalake/auditoria/manifesto.json")
    ouro_path = Path("datalake/ouro/base_final_looker.csv")
    
    assert manifesto_path.exists()
    assert ouro_path.exists()
    
    with open(manifesto_path, "r") as f: manifesto = json.load(f)
    
    sha256 = hashlib.sha256()
    with open(ouro_path, "rb") as f:
        for bloco in iter(lambda: f.read(4096), b""): sha256.update(bloco)
    
    assert sha256.hexdigest() == manifesto.get("hash_ouro")

def test_kpis_analiticos_looker():
    df = pd.read_csv("datalake/ouro/base_final_looker.csv")
    
    assert 'score_risco' in df.columns
    assert 'influencia_is_pagamento' in df.columns
    assert 'influencia_is_feriado' in df.columns
    
    assert 'h3_index' in df.columns
    assert 'perfil' in df.columns
