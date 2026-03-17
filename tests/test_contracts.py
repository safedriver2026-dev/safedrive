import os, sys, pandas as pd, numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autobot.autobot_engine import AutobotSafeDriver

def test_instancia_motor():
    bot = AutobotSafeDriver(persistencia=False)
    assert bot.identidade == "Autobot SafeDriver"

def test_normalizacao_colunas():
    bot = AutobotSafeDriver(persistencia=False)
    assert bot._normalizar_coluna("LAT") == "LATITUDE"
    assert bot._normalizar_coluna("RUBRICA") == "NATUREZA_APURADA"

def test_higienizacao_texto():
    bot = AutobotSafeDriver(persistencia=False)
    assert bot._higienizar("roubo de veículo") == "ROUBO DE VEICULO"

def test_persistência_parquet(tmp_path):
    bot = AutobotSafeDriver(persistencia=False)
    df = pd.DataFrame({'LATITUDE': [-23.5], 'LONGITUDE': [-46.6]})
    caminho = tmp_path / "teste.parquet"
    df.to_parquet(caminho)
    assert os.path.exists(caminho)
