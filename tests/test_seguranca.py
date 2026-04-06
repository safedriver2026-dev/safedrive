import pytest
import pandas as pd
import json
from pathlib import Path

def test_auditoria_criptografica():
    manifesto_path = Path("datalake/auditoria/controle_integridade.json")
    base_dash = Path("datalake/ouro/dashboard_risco_consolidado.csv")
    
    assert manifesto_path.exists(), "ERRO: Manifesto de integridade não encontrado"
    assert base_dash.exists(), "ERRO: Base de saída para Dashboard não encontrada"
    
    with open(manifesto_path, "r") as f:
        dados = json.load(f)
        assert "sha256" in dados, "ERRO: Selo SHA-256 ausente no manifesto"
        assert dados["r2"] > 0.55, f"ERRO: Confiabilidade abaixo da meta (Atual: {dados['r2']:.2%})"

def test_qualidade_dados_ouro():
    df = pd.read_csv("datalake/ouro/dashboard_risco_consolidado.csv")
    colunas_obrigatorias = ['H3_INDEX', 'PREDICAO_RISCO', 'LAT', 'LON']
    
    for col in colunas_obrigatorias:
        assert col in df.columns, f"ERRO: Coluna vital {col} ausente na camada Ouro"
    
    assert df['PREDICAO_RISCO'].min() >= 0, "ERRO: Predição de risco negativa detectada"
    assert not df['H3_INDEX'].isnull().any(), "ERRO: Existem hexágonos nulos na base final"
