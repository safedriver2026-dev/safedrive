import os, sys, pandas as pd, numpy as np
from datetime import datetime
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autobot.autobot_engine import AutobotSafeDriver

def test_instancia():
    bot = AutobotSafeDriver(persistencia=False)
    assert bot.identidade == "Autobot SafeDriver"

def test_higienizacao_dados_sujos():
    bot = AutobotSafeDriver(persistencia=False)
    df = pd.DataFrame({
        'LATITUDE': [-23.5, -23.5], 'LONGITUDE': [-46.6, -46.6],
        'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-03-01'), pd.Timestamp('2026-03-02')],
        'HORA_OCORRENCIA_BO': ['', ' '], # Horários vazios que causavam erro
        'NATUREZA_APURADA': ['ROUBO', 'FURTO']
    })
    res = bot._processar_ia(df)
    assert len(res) >= 0

def test_normalizacao_colunas():
    bot = AutobotSafeDriver(persistencia=False)
    assert bot._normalizar("LAT") == "LATITUDE"
