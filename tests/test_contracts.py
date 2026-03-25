import pytest
import pandas as pd
import os

# Configurações de Contrato
EXPECTED_COLUMNS = ['LATITUDE', 'LONGITUDE', 'score', 'h3_id']
MIN_RECORDS = 100 # Segurança para não gerar modelos com dados insuficientes

def test_silver_integrity():
    """Valida se a camada Silver está pronta para virar Gold."""
    path = "datalake/silver/main.parquet"
    assert os.path.exists(path), "❌ Erro Crítico: Camada Silver não encontrada."
    
    df = pd.read_parquet(path)
    
    # Validação de Contrato de Schema
    for col in ['LATITUDE', 'LONGITUDE']:
        assert col in df.columns, f"❌ Coluna {col} ausente."
        assert df[col].isnull().sum() == 0, f"❌ Valores nulos detectados em {col}."
    
    # Validação Geográfica (São Paulo)
    assert df['LATITUDE'].between(-26, -19).all(), "🚨 Detecção de coordenadas fora de SP."
    assert len(df) >= MIN_RECORDS, f"🚨 Volume de dados insuficiente: {len(df)} registros."

def test_gold_distribution():
    """Valida se o output da API faz sentido estatístico."""
    path = "datalake/gold/api_v1.json"
    if os.path.exists(path):
        df = pd.read_json(path)
        assert df['score'].mean() > 0, "🚨 Alerta: Score médio de risco zerado."
