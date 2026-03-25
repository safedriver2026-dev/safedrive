import pytest
import os
import pandas as pd
from autobot.autobot_engine import MotorInteligenciaLakehouse

@pytest.fixture
def motor():
    return MotorInteligenciaLakehouse()

def test_arquitetura_diretorios(motor):
    for nome_camada, caminho in motor.dirs.items():
        assert os.path.exists(caminho), f"🚨 FALHA CRÍTICA: Diretório '{nome_camada}' ausente em {caminho}"

def test_engenharia_e_modelagem_estrela(motor):
    df_mock_bronze = pd.DataFrame({
        'LATITUDE': [-23.5505, -23.5506],
        'LONGITUDE': [-46.6333, -46.6334],
        'DATA_OCORRENCIA': pd.to_datetime(['2026-03-05', '2026-03-20']),
        'DESCRICAO': ['ROUBO DE VEICULO', 'FURTO DE CELULAR']
    })
    
    df_prata = motor._engenharia_prata(df_mock_bronze)
    assert 'ID_LOCALIZACAO' in df_prata.columns, "Falha: Índice H3 não foi gerado na Prata."
    assert 'DIA_SEMANA' in df_prata.columns, "Falha: Engenharia temporal não foi aplicada."
    
    motor._modelagem_ouro(df_prata)
    
    artefatos_esperados = [
        f"{motor.dirs['estrela']}/dim_tempo.csv",
        f"{motor.dirs['estrela']}/dim_localizacao.csv",
        f"{motor.dirs['estrela']}/dim_perfil.csv",
        f"{motor.dirs['estrela']}/fato_risco.csv",
        f"{motor.dirs['ouro']}/mapa_auditavel.parquet"
    ]
    
    for artefato in artefatos_esperados:
        assert os.path.exists(artefato), f"🚨 FALHA NO MODELO: Artefato {artefato} não foi gerado."
