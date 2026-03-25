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

def test_fluxo_inteligencia_hibrida(robo):
    df_mock = pd.DataFrame({
        'NUM_BO': ['1', '2', '3', '4'], 
        'LATITUDE': [-23.55, -23.56, -23.57, -23.58],
        'LONGITUDE': [-46.63, -46.64, -46.65, -46.66],
        'DATA_OCORRENCIA_BO': pd.to_datetime(['2026-03-01', '2026-03-02', '2026-03-03', '2026-03-04']),
        'RUBRICA': ['ROUBO', 'FURTO', 'VIOLENCIA DOMESTICA', 'LATROCINIO'],
        'DESCR_TIPOLOCAL': ['VIA PUBLICA', 'POSTO', 'CASA', 'RODOVIA']
    })
    
    robo.telemetria['linhas_bronze'] = len(df_mock)
    df_prata = robo._processar_camada_prata(df_mock)
    
    # VERIFICA SE A IA LIMPOU A VIOLÊNCIA DOMÉSTICA
    assert 'VIOLENCIA DOMESTICA' not in df_prata['RUBRICA'].values, "🚨 IA FALHOU NA LIMPEZA DE RUÍDO."
    
    robo._processar_camada_ouro(df_prata)
    assert os.path.exists(f"{robo.diretorios['estrela']}/fato_auditoria.csv"), "🚨 EVIDÊNCIA DE AUDITORIA NÃO GERADA."
