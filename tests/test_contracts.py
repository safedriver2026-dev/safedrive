import os, sys, pandas as pd, numpy as np
from datetime import datetime, timedelta
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autobot.autobot_engine import AutobotSafeDriver

def test_autobot_sequencia_camadas():
    bot = AutobotSafeDriver(persistencia=False)
    # Simula dados Brutos
    df_raw = pd.DataFrame({
        'NUM_BO': ['1'], 'LATITUDE': ['-23.5'], 'LONGITUDE': ['-46.6'],
        'NATUREZA_APURADA': ['ROUBO DE VEICULO'], 'DATA_OCORRENCIA_BO': ['2026-03-01'],
        'HORA_OCORRENCIA_BO': ['14:00']
    })
    # Constrói a Camada Confiável (Silver)
    conf, ref = bot._qualificar_dados(df_raw)
    assert len(conf) == 1
    assert 'ANO_BASE' in conf.columns

def test_bloqueio_ia_sem_dados():
    bot = AutobotSafeDriver(persistencia=False)
    df_vazio = pd.DataFrame()
    # O motor não deve travar, deve retornar vazio para não processar IA
    malha = bot._processar_ia(df_vazio)
    assert malha.empty

def test_normalizacao_autobot():
    bot = AutobotSafeDriver(persistencia=False)
    assert bot._normalizar("LAT") == "LATITUDE"
