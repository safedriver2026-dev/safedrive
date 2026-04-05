import hashlib
import json
import pandas as pd
from pathlib import Path

def test_validar_infraestrutura_pastas():
    pastas_obrigatorias = ["bronze", "ouro", "auditoria", "documentacao"]
    for p in pastas_obrigatorias:
        assert Path(f"datalake/{p}" if p != "documentacao" else p).exists()

def test_auditoria_assinaturas_ouro():
    caminho_manifesto = Path("datalake/auditoria/manifesto.json")
    assert caminho_manifesto.exists()
    
    with open(caminho_manifesto, "r") as f_audit:
        dados_manifesto = json.load(f_audit)
    
    caminho_csv_ouro = Path("datalake/ouro/base_inteligencia.csv")
    assert caminho_csv_ouro.exists()
    
    sha256_verificador = hashlib.sha256()
    with open(caminho_csv_ouro, "rb") as f_bin:
        for bloco in iter(lambda: f_bin.read(4096), b""): sha256_verificador.update(bloco)
    
    assert sha256_verificador.hexdigest() == dados_manifesto.get("camada_ouro")

def test_validar_kpis_explicabilidade():
    df_ouro = pd.read_csv("datalake/ouro/base_inteligencia.csv")
    colunas_ia = ['score_final', 'influencia_perfil_cod', 'influencia_periodo_cod']
    for col_ia in colunas_ia:
        assert col_ia in df_ouro.columns
