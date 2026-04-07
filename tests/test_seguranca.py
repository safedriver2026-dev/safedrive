import pytest
import json
import polars as pl
from pathlib import Path

def test_pipeline_integridade_e_metricas():
    caminho_auditoria = Path("datalake/auditoria/auditoria.json")
    assert caminho_auditoria.exists(), "[FALHA] Arquivo de auditoria nao localizado."
    
    with open(caminho_auditoria, "r") as f:
        log = json.load(f)
        
    assert log.get("precisao_modelo", 0) > 0.35, "[FALHA] Precisao (R2) abaixo do limite operacional."
    assert "assinaturas_seguranca" in log, "[FALHA] Ausencia de assinaturas criptograficas SHA-256."
    assert log.get("volumetria_bruta", 0) > 0, "[FALHA] Volume de entrada zerado."

def test_deduplicacao_e_limpeza():
    caminho_prata = Path("datalake/prata/camada_prata.parquet")
    if caminho_prata.exists():
        df = pl.read_parquet(caminho_prata)
        assert df.height == df.select("NUM_BO").unique().height, "[FALHA] Duplicatas detectadas na Camada Prata."
        assert df.filter((pl.col("LAT") > -19.5) | (pl.col("LAT") < -25.5)).height == 0, "[FALHA] Dados fora de SP detectados."

def test_engenharia_de_features():
    caminho_ouro = Path("datalake/ouro/dashboard_final.parquet")
    if caminho_ouro.exists():
        df = pl.read_parquet(caminho_ouro)
        colunas_obrigatorias = ["MES", "IS_FIM_SEMANA", "IS_FERIADO", "QTD_VIA_PUBLICA", "PREVISAO_FINAL"]
        for col in colunas_obrigatorias:
            assert col in df.columns, f"[FALHA] Variavel critica ausente: {col}"
        
    
        assert df.select("IS_FERIADO").unique().height <= 2
        assert df.select("IS_FIM_SEMANA").unique().height <= 2

def test_qualidade_h3():
    caminho_ouro = Path("datalake/ouro/dashboard_final.parquet")
    if caminho_ouro.exists():
        df = pl.read_parquet(caminho_ouro)
        assert df.filter(pl.col("H3").is_null()).height == 0, "[FALHA] Falha na indexacao hexagonal H3."
        assert "RISCO_GEO" in df.columns, "[FALHA] Suavizacao de vizinhanca nao processada."
