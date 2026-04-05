import hashlib
import json
import pandas as pd
from pathlib import Path

def test_auditoria_incremental():
    man = Path("datalake/auditoria/manifesto.json")
    out = Path("datalake/ouro/base_final_bi.csv")
    assert man.exists() and out.exists()
    
    with open(man, "r") as f: dados = json.load(f)
    hash_bi = hashlib.sha256(open(out, "rb").read()).hexdigest()
    assert hash_bi == dados.get("hash_ouro")

def test_ausencia_de_duplicatas():
    df = pd.read_csv("datalake/ouro/base_final_bi.csv")
    # No Looker, o h3+periodo+perfil deve ser unico
    assert df.duplicated(subset=['h3_index', 'desc_periodo', 'perfil']).sum() == 0
