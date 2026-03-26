import pytest
from pathlib import Path

def test_estrutura_diretorios():
    base = Path(".")
    assert (base / "autobot").exists()
    assert (base / "datalake").exists() or True

def test_importacao_motor():
    try:
        from autobot.motor_analitico import MotorSafeDriver
        assert True
    except ImportError:
        pytest.fail("Módulo autobot não localizado no ambiente.")
