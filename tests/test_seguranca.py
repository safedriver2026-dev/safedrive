import pytest
import json
import polars as pl
from pathlib import Path

def test_pipeline_integridade_e_metricas():
    caminho_auditoria = Path("datalake/auditoria/auditoria.json")
    assert caminho_auditoria.exists(), "[FALHA] Ficheiro de auditoria não localizado."
    
    with open(caminho_auditoria, "r") as f:
        log = json.load(f)
        
    assert log.get("precisao_modelo", 0) > 0.35, "[FALHA] Precisão (R2) abaixo do limite operacional de segurança."
    assert "assinaturas_seguranca" in log, "[FALHA] Ausência de assinaturas criptográficas SHA-256."
    assert log.get("volumetria_bruta", 0) > 0, "[FALHA] Volume de entrada zerado."

def test_deduplicacao_e_limpeza():
    caminho_prata = Path("datalake/prata/camada_prata.parquet")
    if caminho_prata.exists():
        df = pl.read_parquet(caminho_prata)
       
        assert df.height == df.select("ID_ANONIMO").unique().height, "[FALHA] Duplicados detetados na Camada Prata."
     
        assert df.filter((pl.col("LAT") > -19.5) | (pl.col("LAT") < -25.5)).height == 0, "[FALHA] Dados fora do perímetro de SP detetados."

def test_engenharia_de_features_e_lgpd():
    caminho_ouro = Path("datalake/ouro/dashboard_final.parquet")
    if caminho_ouro.exists():
        df = pl.read_parquet(caminho_ouro)
        
       
        colunas_obrigatorias = ["MES", "IS_FIM_SEMANA", "IS_FERIADO", "QTD_VIA_PUBLICA", "PREVISAO_FINAL"]
        for col in colunas_obrigatorias:
            assert col in df.columns, f"[FALHA] Variável preditiva crítica ausente: {col}"
        
        assert "NUM_BO" not in df.columns, "[FALHA CRÍTICA - LGPD] Identificador NUM_BO não foi destruído do dashboard final."

def test_integridade_temporal_multi_aba():
    caminho_ouro = Path("datalake/ouro/dashboard_final.parquet")
    if caminho_ouro.exists():
        df = pl.read_parquet(caminho_ouro)
        
    
        meses_capturados = df.select("MES").unique().height
        assert meses_capturados > 1, "[FALHA] O modelo capturou apenas 1 mês de dados. O Scanner Multi-Aba pode ter falhado."
        
       
        assert df.height > 5000, "[FALHA] Volumetria suspeitamente baixa. Falha na concatenação diagonal."

def test_qualidade_h3():
    caminho_ouro = Path("datalake/ouro/dashboard_final.parquet")
    if caminho_ouro.exists():
        df = pl.read_parquet(caminho_ouro)
        assert df.filter(pl.col("H3").is_null()).height == 0, "[FALHA] Falha na indexação hexagonal H3."
        assert "RISCO_GEO" in df.columns, "[FALHA] Suavização espacial de vizinhança não processada."
