import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import pytest

# Garante que o Python encontre a pasta do Autobot
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autobot.autobot_engine import AutobotSafeDriver

def test_autobot_instancia():
    bot = AutobotSafeDriver(persistencia=False)
    assert hasattr(bot, "executar")

def test_correcao_gps():
    bot = AutobotSafeDriver(persistencia=False)
    # Testando se ele corrige a coordenada sem ponto decimal da SSP
    assert -47.0 < bot._corrigir_coordenadas(-46779670.0, latitude=False) < -46.0

def test_normalizacao_colunas():
    bot = AutobotSafeDriver(persistencia=False)
    # Testando a tradução semântica de colunas do Excel
    assert bot._normalizar("RUBRICA") == "NATUREZA_APURADA"

def test_classificacao_natureza():
    bot = AutobotSafeDriver(persistencia=False)
    # Testando se ele reconhece crimes do catálogo
    assert bot._classificar_crime("ROUBO DE CARGA") == "ROUBO DE CARGA"
    assert bot._classificar_crime("ALGO_ESTRANHO") == "OUTROS"

def test_qualificacao_fluxo():
    bot = AutobotSafeDriver(persistencia=False)
    df = pd.DataFrame({
        'LATITUDE':['-23.5'],
        'LONGITUDE':['-46.6'],
        'DATA_OCORRENCIA_BO':[pd.Timestamp(datetime.now().date())],
        'NATUREZA_APURADA':['ROUBO DE VEICULO'],
        'NUM_BO':['1'],
        'DESCR_TIPOLOCAL':['VIA PUBLICA']
    })
    t, r = bot._qualificar_dados(df)
    assert len(t) == 1
    assert len(r) == 1
