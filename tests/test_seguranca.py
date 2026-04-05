import hashlib
from pathlib import Path
import pandas as pd

def test_validar_assinatura_ouro():
    csv = Path("datalake/ouro/base_looker.csv")
    sha = Path("datalake/ouro/base_looker.sha256")
    assert csv.exists() and sha.exists()
    check = hashlib.sha256(open(csv, "rb").read()).hexdigest()
    with open(sha, "r") as f:
        assert check == f.read().strip()

def test_verificar_perfis():
    df = pd.read_csv("datalake/ouro/base_looker.csv")
    assert 'perfil' in df.columns
    assert set(['Motorista', 'Pedestre', 'Ciclista']).intersection(set(df['perfil'].unique()))

def test_diagramas_existem():
    for d in ["automacao", "dados", "api"]:
        assert Path(f"documentacao/arquitetura_{d}.png").exists()
