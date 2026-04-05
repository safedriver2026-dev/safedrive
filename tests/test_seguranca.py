import hashlib
from pathlib import Path
import pandas as pd

def test_assinatura_digital():
    csv = Path("datalake/ouro/base_looker.csv")
    sha = Path("datalake/ouro/base_looker.sha256")
    assert csv.exists() and sha.exists()
    check = hashlib.sha256(open(csv, "rb").read()).hexdigest()
    with open(sha, "r") as f:
        assert check == f.read().strip()

def test_perfis_gerados():
    df = pd.read_csv("datalake/ouro/base_looker.csv")
    assert 'perfil' in df.columns
    assert len(df['perfil'].unique()) >= 1
