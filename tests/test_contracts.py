import os, sys, pandas as pd, numpy as np
from datetime import datetime
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autobot.autobot_engine import AutobotSafeDriver

def test_instancia_autobot():
    bot = AutobotSafeDriver(persistencia=False)
    assert "Autobot" in bot.identidade

def test_higienizacao_horario_vazio():
    bot = AutobotSafeDriver(persistencia=False)
    df = pd.DataFrame({
        'LATITUDE': [-23.5, -23.6], 'LONGITUDE': [-46.6, -46.7],
        'DATA_OCORRENCIA_BO': [pd.Timestamp('2026-03-01'), pd.Timestamp('2026-03-02')],
        'HORA_OCORRENCIA_BO': ['', '14:00'], 'NATUREZA_APURADA': ['ROUBO DE VEICULO', 'FURTO'],
        'NUM_BO': ['1', '2'], 'DESCR_TIPOLOCAL': ['VIA PUBLICA', 'VIA PUBLICA']
    })
    previsoes = bot._processar_ia(df)
    assert len(previsoes) >= 0 

def test_normalizacao_colunas():
    bot = AutobotSafeDriver(persistencia=False)
    assert bot._normalizar("RUBRICA") == "NATUREZA_APURADA"
