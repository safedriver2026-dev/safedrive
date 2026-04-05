import hashlib
import json
import pandas as pd
from pathlib import Path

def test_auditoria_integridade_total():
    manifesto_path = Path("datalake/auditoria/manifesto.json")
    ouro_path = Path("datalake/ouro/base_final_looker.csv")
    
    assert manifesto_path.exists(), "FALHA: Manifesto de auditoria não encontrado"
    assert ouro_path.exists(), "FALHA: Base Ouro não encontrada"
    
    with open(manifesto_path, "r") as f:
        manifesto = json.load(f)
    
    sha256 = hashlib.sha256()
    with open(ouro_path, "rb") as f:
        for bloco in iter(lambda: f.read(4096), b""): sha256.update(bloco)
    
    assert sha256.hexdigest() == manifesto.get("hash_ouro"), "CRÍTICO: A base Ouro foi adulterada"

def test_conferência_kpis_ia_explicavel():
    df = pd.read_csv("datalake/ouro/base_final_looker.csv")
    
    # KPIs obrigatórios para o Looker Studio
    assert 'score_risco' in df.columns, "FALHA: Score de risco ausente"
    assert 'influencia_perfil_idx' in df.columns, "FALHA: SHAP perfil ausente"
    assert 'influencia_periodo_idx' in df.columns, "FALHA: SHAP horário ausente"
    
    # Garante o modelo Estrela (Dimensões presentes)
    assert 'perfil' in df.columns
    assert 'desc_periodo' in df.columns
    assert 'h3_index' in df.columns
