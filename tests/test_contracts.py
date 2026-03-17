import os, sys, pandas as pd, numpy as np
from datetime import datetime
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autobot.autobot_engine import AutobotSafeDriver

def test_autobot_instancia():
    bot = AutobotSafeDriver(persistencia=False)
    assert bot.identidade.startswith("Autobot")

def test_limpeza_horario_vazio():
    bot = AutobotSafeDriver(persistencia=False)
    df = pd.DataFrame({
        'LATITUDE': [-23.5], 'LONGITUDE': [-46.6],
        'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-03-01')],
        'HORA_OCORRENCIA_BO': [''], 'NATUREZA_APURADA': ['ROUBO DE VEICULO'],
        'NUM_BO': ['1'], 'DESCR_TIPOLOCAL': ['VIA PUBLICA']
    })
    # O teste deve passar sem ValueError: invalid literal
    malha = bot._processar_ia(df)
    assert len(malha) >= 0

def test_normalizacao_colunas():
    bot = AutobotSafeDriver(persistencia=False)
    assert bot._normalizar("RUBRICA") == "NATUREZA_APURADA"
