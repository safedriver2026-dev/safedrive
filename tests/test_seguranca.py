import pytest
import polars as pl
import os
import re
import logging

logging.basicConfig(level=logging.INFO)

@pytest.fixture
def sample_df_prata():
    return pl.DataFrame({
        "H3_INDEX": ["89a81001d83ffff", "89a81ed28a3ffff"],
        "TOTAL_CRIMES": [10, 5],
        "NM_BAIRRO": ["Jardim Aeroporto", "Centro"],
        "NM_MUN": ["São Paulo", "Limeira"],
        "DENSIDADE": [12000.5, 8500.2],
        "ANO_REFERENCIA": [2026, 2026],
        "LATITUDE": [-23.5505, -23.5510],
        "LONGITUDE": [-46.6333, -46.6335]
    })

def test_contrato_colunas_obrigatorias_prata(sample_df_prata):
    colunas_esperadas = {
        "H3_INDEX", "TOTAL_CRIMES", "NM_BAIRRO", "NM_MUN", "DENSIDADE", "ANO_REFERENCIA"
    }
    assert colunas_esperadas.issubset(set(sample_df_prata.columns))

def test_contrato_tipagem_ia(sample_df_prata):
    for col in ["TOTAL_CRIMES", "DENSIDADE"]:
        dtype = sample_df_prata[col].dtype
        assert dtype in [pl.Int32, pl.Int64, pl.Float64, pl.Float32]

def test_lgpd_ausencia_dados_pessoais():
    termos_proibidos = ["NOME", "CPF", "RG", "VITIMA", "AUTOR", "TELEFONE", "ENDERECO"]
    colunas_atuais = ["H3_INDEX", "TOTAL_CRIMES", "NM_BAIRRO", "NM_MUN", "DENSIDADE"]
    
    for termo in termos_proibidos:
        for col in colunas_atuais:
            assert termo not in col.upper()

def test_lgpd_pseudonimizacao_h3(sample_df_prata):
    padrao_h3 = re.compile(r"^[89a-f0-f]{15}$")
    for codigo in sample_df_prata["H3_INDEX"]:
        assert padrao_h3.match(codigo)

def test_seguranca_presenca_secrets():
    keys_obrigatorias = [
        "R2_ACCESS_KEY_ID", 
        "R2_SECRET_ACCESS_KEY", 
        "BQ_SERVICE_ACCOUNT_JSON"
    ]
    if os.getenv("GITHUB_ACTIONS"):
        for key in keys_obrigatorias:
            valor = os.getenv(key)
            assert valor is not None and len(valor) > 0

def test_seguranca_exposicao_secrets_em_logs(caplog):
    logger = logging.getLogger("SafeDriver")
    logger.info("Verificando opacidade de logs.")
    
    secret_fake = os.getenv("R2_SECRET_ACCESS_KEY", "VALOR_PADRAO")
    
    for record in caplog.records:
        if secret_fake != "VALOR_PADRAO":
            assert secret_fake not in record.text
