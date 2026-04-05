import hashlib
import json
from pathlib import Path

def test_integridade_dos_dados():
    pasta_seguranca = Path("datalake/auditoria/controle_integridade.json")
    base_ia = Path("datalake/ouro/predicao_risco_mapa.csv")
    base_detalhes = Path("datalake/ouro/crimes_detalhados.csv")
    
    assert pasta_seguranca.exists(), "Arquivo de controle sumiu!"
    
    with open(pasta_seguranca, "r") as f: controle = json.load(f)
    
    def conferir_selo(caminho):
        h = hashlib.sha256()
        with open(caminho, "rb") as f:
            for bloco in iter(lambda: f.read(4096), b""): h.update(bloco)
        return h.hexdigest()
    
    # O teste falha se alguém mudar um único B.O. no arquivo final
    assert conferir_selo(base_ia) == controle.get("selo_ia"), "Base de IA foi violada!"
    assert conferir_selo(base_detalhes) == controle.get("selo_detalhes"), "Base de Detalhes foi violada!"

def test_unicidade_bo():
    import pandas as pd
    df = pd.read_csv("datalake/ouro/crimes_detalhados.csv")
    # Garante que o motor não deixou passar B.O.s repetidos
    assert df['num_bo'].is_unique, "Existem B.O.s duplicados na base final!"
