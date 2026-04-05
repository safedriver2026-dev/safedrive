import hashlib
from pathlib import Path

def test_integridade_assinatura_digital():
    csv = Path("datalake/ouro/base_looker.csv")
    sha = Path("datalake/ouro/base_looker.sha256")
    assert csv.exists()
    assert sha.exists()
    calculado = hashlib.sha256(open(csv, "rb").read()).hexdigest()
    original = open(sha, "r").read().strip()
    assert calculado == original

def test_evidencia_grafica_ia():
    assert Path("documentacao/explicabilidade_ia.png").exists()

def test_diagramas_gerados():
    for d in ["automacao", "dados", "api"]:
        assert Path(f"documentacao/arquitetura_{d}.png").exists()
