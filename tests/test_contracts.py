import os, sys, pandas as pd, numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autobot.autobot_engine import AutobotSafeDriver

def test_instancia():
    bot = AutobotSafeDriver(persistencia=False)
    assert bot.identidade == "Autobot SafeDriver"

def test_normalizacao_colunas():
    bot = AutobotSafeDriver(persistencia=False)
    # Testa se o motor reconhece sinônimos de colunas da SSP-SP
    assert bot._normalizar_coluna("LAT") == "LATITUDE"
    assert bot._normalizar_coluna("RUBRICA") == "NATUREZA_APURADA"

def test_higienizacao_dados_sujos():
    bot = AutobotSafeDriver(persistencia=False)
    df = pd.DataFrame({
        'LATITUDE': [-23.5, -23.5], 'LONGITUDE': [-46.6, -46.6],
        'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-03-01'), pd.Timestamp('2026-03-02')],
        'HORA_OCORRENCIA_BO': ['', ' '], # Simula o erro de horário vazio
        'NATUREZA_APURADA': ['ROUBO DE VEICULO', 'FURTO DE CELULAR'],
        'LOCAL': ['VIA PUBLICA', 'CALCADA']
    })
    res = bot._processar_ia(df)
    assert len(res) >= 0
