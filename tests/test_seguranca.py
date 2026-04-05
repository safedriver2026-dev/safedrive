import hashlib
import json
import pandas as pd
from pathlib import Path

def test_auditoria_criptografica():
    manifesto_path = Path("datalake/auditoria/manifesto.json")
    ouro_ia_path = Path("datalake/ouro/base_final_looker.csv")
    ouro_detalhes_path = Path("datalake/ouro/base_crimes_detalhados.csv")
    
    assert manifesto_path.exists()
    assert ouro_ia_path.exists()
    assert ouro_detalhes_path.exists()
    
    with open(manifesto_path, "r") as f: manifesto = json.load(f)
    
    sha256_ia = hashlib.sha256()
    with open(ouro_ia_path, "rb") as f:
        for bloco in iter(lambda: f.read(4096), b""): sha256_ia.update(bloco)
    assert sha256_ia.hexdigest() == manifesto.get("hash_ouro_ia")

    sha256_detalhes = hashlib.sha256()
    with open(ouro_detalhes_path, "rb") as f:
        for bloco in iter(lambda: f.read(4096), b""): sha256_detalhes.update(bloco)
    assert sha256_detalhes.hexdigest() == manifesto.get("hash_ouro_detalhes")

def test_kpis_analiticos_looker():
    df_ia = pd.read_csv("datalake/ouro/base_final_looker.csv")
    df_detalhes = pd.read_csv("datalake/ouro/base_crimes_detalhados.csv")
    
    assert 'score_risco' in df_ia.columns
    assert 'influencia_is_pagamento' in df_ia.columns
    assert 'h3_index' in df_ia.columns
    
    assert 'num_bo' in df_detalhes.columns
    assert 'h3_index' in df_detalhes.columns
