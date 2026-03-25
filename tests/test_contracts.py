import pytest
import os
import pandas as pd
from autobot.autobot_engine import MotorInteligenciaLakehouse

@pytest.fixture
def motor():
    return MotorInteligenciaLakehouse()

def test_arquitetura_diretorios(motor):
    for nome_camada, caminho in motor.dirs.items():
        assert os.path.exists(caminho), f"🚨 Diretório crítico {nome_camada} ausente."

def test_modelagem_multidisciplinar(motor):
    # Mock data simulando as complexas colunas da SSP
    df_mock_bronze = pd.DataFrame({
        'NUM_BO': ['123', '456'],
        'LATITUDE': [-23.5505, -23.5506],
        'LONGITUDE': [-46.6333, -46.6334],
        'DATA_OCORRENCIA_BO': pd.to_datetime(['2026-03-05 14:30:00', '2026-03-20 02:15:00']),
        'DESC_PERIODO': ['A TARDE', 'DE MADRUGADA'],
        'RUBRICA': ['ROUBO', 'FURTO'],
        'DESCR_CONDUTA': ['TRANSEUNTE', 'VEICULO'],
        'DESCR_TIPOLOCAL': ['VIA PUBLICA', 'POSTO DE GASOLINA'],
        'NOME_MUNICIPIO': ['SAO PAULO', 'OSASCO'],
        'BTL': ['1º BPM/M', '14º BPM/M'],
        'CIA': ['1ª CIA', '2ª CIA']
    })
    
    df_prata = motor._engenharia_prata(df_mock_bronze)
    
    # Verifica extrações cruciais
    assert 'HORA' in df_prata.columns, "Engenharia falhou na extração de horas."
    assert 'ID_LOCALIZACAO' in df_prata.columns, "Falha na geração H3."
    
    motor._modelagem_ouro(df_prata)
    
    # Valida as 6 pontas do Star Schema
    artefatos = [
        'dim_tempo.csv', 'dim_localizacao.csv', 'dim_perfil_crime.csv', 
        'dim_ambiente.csv', 'dim_jurisdicao.csv', 'fato_risco.csv'
    ]
    
    for artefato in artefatos:
        caminho_completo = f"{motor.dirs['estrela']}/{artefato}"
        assert os.path.exists(caminho_completo), f"🚨 Falha de Governança: Tabela {artefato} não gerada."
