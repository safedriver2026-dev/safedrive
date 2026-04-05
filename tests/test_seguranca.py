import hashlib
import pandas as pd
from pathlib import Path

def test_integridade_assinatura_digital():
    """Valida se a base Ouro não foi adulterada manualmente"""
    caminho_csv = Path("datalake/ouro/base_final_looker.csv")
    caminho_sha = Path("datalake/ouro/assinatura.sha256")
    
    assert caminho_csv.exists(), "ERRO: Base Ouro (CSV) não encontrada."
    assert caminho_sha.exists(), "ERRO: Assinatura Digital (SHA256) não encontrada."
    
    hash_calculado = hashlib.sha256(open(caminho_csv, "rb").read()).hexdigest()
    with open(caminho_sha, "r") as f:
        hash_original = f.read().strip()
        
    assert hash_calculado == hash_original, "CRÍTICO: A assinatura digital não coincide com os dados!"

def test_qualidade_dados_ia_explicavel():
    """Valida se as colunas de inteligência (SHAP) estão presentes para o Looker"""
    df = pd.read_csv("datalake/ouro/base_final_looker.csv")
    
    # Colunas obrigatórias do Modelo Estrela + XAI
    colunas_esperadas = [
        'h3_index', 'score_predito', 'perfil', 
        'influencia_perfil_idx', 'influencia_periodo_idx'
    ]
    
    for col in colunas_esperadas:
        assert col in df.columns, f"ERRO: A coluna estratégica '{col}' está em falta no CSV final."

def test_presenca_diagramas_arquitetura():
    """Valida se a documentação técnica foi gerada corretamente"""
    diagramas = ["automacao", "modelo_estrela", "integracao"]
    for diag in diagramas:
        path_diag = Path(f"documentacao/arquitetura_{diag}.png")
        assert path_diag.exists(), f"ERRO: O diagrama '{diag}' não foi gerado."
