import pytest
import json
import polars as pl
from pathlib import Path

def test_pipeline_integridade_e_metricas():
    caminho_auditoria = Path("datalake/auditoria/auditoria.json")
    assert caminho_auditoria.exists(), "[FALHA] Arquivo de auditoria nao localizado no diretorio esperado."
    
    with open(caminho_auditoria, "r") as f:
        log = json.load(f)
        
    assert log.get("r2_final", 0) > 0.35, "[FALHA] Metrica R2 abaixo do limite minimo aceitavel para operacao."
    assert len(log.get("hashes_seguranca", {})) > 0, "[FALHA] Ausencia de hashes SHA-256 no manifesto de seguranca."
    assert "timestamp" in log, "[FALHA] Registro de tempo ausente no manifesto."
    assert log.get("anomalias_estatisticas_z3", 0) >= 0, "[FALHA] Valor de anomalias estatisticas invalido."

def test_deduplicacao_bo():
    caminho_prata = Path("datalake/prata/camada_prata.parquet")
    if caminho_prata.exists():
        df = pl.read_parquet(caminho_prata)
        assert df.height == df.select("NUM_BO").unique().height, "[FALHA] Duplicatas detectadas na chave primaria (NUM_BO)."

def test_antifraude_geografica():
    caminho_prata = Path("datalake/prata/camada_prata.parquet")
    if caminho_prata.exists():
        df = pl.read_parquet(caminho_prata)
        max_lat = df.select(pl.col("LAT").max()).to_numpy()[0][0]
        min_lat = df.select(pl.col("LAT").min()).to_numpy()[0][0]
        max_lon = df.select(pl.col("LON").max()).to_numpy()[0][0]
        min_lon = df.select(pl.col("LON").min()).to_numpy()[0][0]
        
        assert max_lat <= -19.5, "[FALHA] Coordenada de latitude excede limite superior definido."
        assert min_lat >= -25.5, "[FALHA] Coordenada de latitude excede limite inferior definido."
        assert max_lon <= -44.0, "[FALHA] Coordenada de longitude excede limite superior definido."
        assert min_lon >= -53.5, "[FALHA] Coordenada de longitude excede limite inferior definido."

def test_qualidade_geospatial():
    caminho_ouro = Path("datalake/ouro/dashboard_final.parquet")
    if caminho_ouro.exists():
        df = pl.read_parquet(caminho_ouro)
        assert df.filter(pl.col("H3").is_null()).height == 0, "[FALHA] Detectados registros sem indexacao H3 valida."
        assert "PREVISAO_FINAL" in df.columns, "[FALHA] Coluna de previsao ausente no dataset final."
        assert df.select("PREVISAO_FINAL").mean().to_numpy()[0] > 0, "[FALHA] Media preditiva zerada. Modelo requer recalibragem."
