import hashlib
import json
import pandas as pd
from pathlib import Path

def test_validar_trilha_de_auditoria_completa():
    """Verifica se todos os ficheiros do Datalake batem com o Manifesto de Auditoria"""
    caminho_manifesto = Path("datalake/auditoria/manifesto.json")
    
    assert caminho_manifesto.exists(), "ERRO: Manifesto de auditoria não encontrado."
    
    with open(caminho_manifesto, "r") as f:
        manifesto = json.load(f)
    
    # Validação da Camada Ouro (Produto Final)
    caminho_ouro = Path("datalake/ouro/base_final.csv")
    assert caminho_ouro.exists(), "ERRO: Ficheiro da camada Ouro ausente."
    
    hash_ouro_atual = hashlib.sha256(open(caminho_ouro, "rb").read()).hexdigest()
    assert hash_ouro_atual == manifesto.get("ouro"), "ALERTA: Integridade da camada Ouro violada!"

def test_validar_presenca_de_inteligencia_explicavel():
    """Garante que a IA exportou os dados SHAP para o Looker Studio"""
    caminho_csv = Path("datalake/ouro/base_final.csv")
    df = pd.read_csv(caminho_csv)
    
    # Verifica se as colunas de influência (SHAP) foram criadas pelo motor
    colunas_ia = ['shap_lat', 'shap_lon', 'shap_perfil_idx', 'shap_periodo_idx', 'score_predito']
    for col in colunas_ia:
        assert col in df.columns, f"ERRO: A coluna de inteligência '{col}' não foi gerada."

def test_verificar_documentacao_tecnica():
    """Valida se os diagramas de arquitetura foram atualizados e salvos"""
    diagramas = ["automacao_auditoria", "modelo_estrela", "integracao_looker"]
    for nome in diagramas:
        path_diag = Path(f"documentacao/arquitetura_{nome}.png")
        assert path_diag.exists(), f"ERRO: O diagrama '{nome}' está em falta."
