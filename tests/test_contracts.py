import os, sys, pandas as pd, numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autobot.autobot_engine import AutobotSafeDriver

def test_instancia():
    bot = AutobotSafeDriver(persistencia=False)
    assert bot.identidade == "Autobot SafeDriver"

def test_normalizacao_colunas():
    bot = AutobotSafeDriver(persistencia=False)
    assert bot._normalizar_coluna("LAT") == "LATITUDE"
    assert bot._normalizar_coluna("RUBRICA") == "NATUREZA_APURADA"

def test_higienizacao_dados_sujos():
    bot = AutobotSafeDriver(persistencia=False)
    df = pd.DataFrame({
        'LATITUDE': [-23.5, -23.51, -23.52], 
        'LONGITUDE': [-46.6, -46.61, -46.62],
        'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-03-01'), pd.Timestamp('2026-03-02'), pd.Timestamp('2026-03-03')],
        'HORA_OCORRENCIA_BO': ['', ' ', '14:00'], 
        'NATUREZA_APURADA': ['ROUBO DE VEICULO', 'FURTO DE CELULAR', 'LATROCINIO'],
        'LOCAL': ['VIA PUBLICA', 'CALCADA', 'VEICULO']
    })
    res = bot._processar_ia(df)
    assert not res.empty

def test_fallback_colunas_ausentes():
    bot = AutobotSafeDriver(persistencia=False)
    df = pd.DataFrame({'LATITUDE': [-23.5], 'LONGITUDE': [-46.6]})
    res = bot._processar_ia(df)
    assert res.empty
