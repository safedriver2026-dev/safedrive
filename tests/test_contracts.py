import pytest
import os
import pandas as pd
from autobot.autobot_engine import SistemaInteligenciaSafeDriver

@pytest.fixture
def robo():
    return SistemaInteligenciaSafeDriver()

def test_arquitetura_diretorios(robo):
    for d in robo.diretorios.values():
        assert os.path.exists(d), f"🚨 FALHA: DIRETÓRIO {d} INEXISTENTE."

def test_fusao_evidencias_e_limpeza_ia(robo):
    df_mock = pd.DataFrame({
        'NUM_BO': ['1', '2', '3'], 
        'LATITUDE': [-23.55, -23.56, -23.57],
        'LONGITUDE': [-46.63, -46.64, -46.65],
        'DATA_OCORRENCIA': pd.to_datetime(['2026-03-01', '2026-03-02', '2026-03-03']),
        'RUBRICA': ['ROUBO', 'FURTO', 'VIOLENCIA DOMESTICA'],
        'NATUREZA_APURADA': ['CARGA', 'VEICULO', 'AMEACA'],
        'HORA_OCORRENCIA_BO': ['14:30', '02:15', '18:00']
    })
    
    robo.telemetria['linhas_bronze'] = len(df_mock)
    df_prata = robo._processar_camada_prata(df_mock)
    
    # A IA DEVE TER EXPULSO A VIOLÊNCIA DOMÉSTICA
    assert len(df_prata) < len(df_mock), "🚨 IA FALHOU NA LIMPEZA DE RUÍDO."
    assert 'VIOLENCIA DOMESTICA' not in df_prata['DESCRICAO_CONSOLIDADA'].iloc[0], "🚨 IA FALHOU NA FUSÃO."
