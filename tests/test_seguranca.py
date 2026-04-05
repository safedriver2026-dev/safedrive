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
    
    def calcular_hash(p):
        sha = hashlib.sha256()
        with open(p, "rb") as f:
            for bloco in iter(lambda: f.read(4096), b""): sha.update(bloco)
        return sha.hexdigest()
    
    assert calcular_hash(ouro_ia_path) == manifesto.get("hash_ouro_ia")
    assert calcular_hash(ouro_detalhes_path) == manifesto.get("hash_ouro_detalhes")

def test_kpis_analiticos_looker():
    df_ia = pd.read_csv("datalake/ouro/base_final_looker.csv")
    assert 'score_risco' in df_ia.columns
    assert 'h3_index' in df_ia.columns
