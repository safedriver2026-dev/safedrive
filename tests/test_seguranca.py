import hashlib
import json
import pandas as pd
from pathlib import Path

def test_auditoria_criptografica():
    manifesto = Path("datalake/auditoria/controle_integridade.json")
    base_ia = Path("datalake/ouro/predicao_risco_mapa.csv")
    base_detalhes = Path("datalake/ouro/crimes_detalhados.parquet")
    
    assert manifesto.exists(), "ERRO: Manifesto de integridade nao encontrado"
    with open(manifesto, "r") as f: controle = json.load(f)
    
    def gerar_hash(caminho):
        sha = hashlib.sha256()
        with open(caminho, "rb") as f:
            for bloco in iter(lambda: f.read(4096), b""): sha.update(bloco)
        return sha.hexdigest()
    
    assert gerar_hash(base_ia) == controle.get("selo_ia"), "ERRO: Base de IA violada"
    assert gerar_hash(base_detalhes) == controle.get("selo_detalhes"), "ERRO: Base de Detalhes violada"

def test_qualidade_dados_ouro():
    df_ia = pd.read_csv("datalake/ouro/predicao_risco_mapa.csv")
    df_detalhes = pd.read_parquet("datalake/ouro/crimes_detalhados.parquet")
    
    assert 'score_risco' in df_ia.columns
    assert 'inf_hora' in df_ia.columns
    assert df_detalhes['num_bo'].is_unique, "ERRO: Existem duplicatas na camada ouro"
