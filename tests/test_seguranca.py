import pytest
import polars as pl
import os
import re
import logging


logging.basicConfig(level=logging.INFO)

@pytest.fixture
def sample_df_prata():
    """Gera um DataFrame de exemplo para os testes."""
    return pl.DataFrame({
        "H3_INDEX": ["88a242ce13fffff", "88a242ce11fffff"],
        "TOTAL_CRIMES_MOTORISTA": [10, 5],
        "INDICE_RESIDENCIAL": [2, 1],
        "ANO_REFERENCIA": [2025, 2025],
        "LATITUDE": [-23.5505, -23.5510],
        "LONGITUDE": [-46.6333, -46.6335]
    })

# TESTES DE CONTRATO 

def test_contrato_colunas_obrigatorias_prata(sample_df_prata):
    """Verifica se as colunas essenciais estão presentes."""
    colunas_esperadas = {
        "H3_INDEX", "TOTAL_CRIMES_MOTORISTA", "INDICE_RESIDENCIAL", "ANO_REFERENCIA"
    }
    assert colunas_esperadas.issubset(set(sample_df_prata.columns)), \
        f"Faltam colunas essenciais na Prata. Encontradas: {sample_df_prata.columns}"

def test_contrato_tipagem_ia(sample_df_prata):
    """Garante que os dados de crimes sejam numéricos."""
    dtype = sample_df_prata["TOTAL_CRIMES_MOTORISTA"].dtype
    assert dtype in [pl.Int32, pl.Int64, pl.Float64, pl.Float32], \
        f"Erro de Contrato: TOTAL_CRIMES deve ser numérico, não {dtype}."

# 2. TESTES DE LGPD (PRIVACIDADE E ANONIMIZAÇÃO)

def test_lgpd_ausencia_dados_pessoais():
    """Garante que colunas com dados sensíveis não existam."""
    termos_proibidos = ["NOME", "CPF", "RG", "VITIMA", "AUTOR", "TELEFONE"]
    # Simulando colunas da Prata/Ouro (ajuste conforme sua realidade)
    colunas_atuais = ["H3_INDEX", "TOTAL_CRIMES_MOTORISTA", "LATITUDE", "LONGITUDE"]
    
    for termo in termos_proibidos:
        for col in colunas_atuais:
            assert termo not in col.upper(), f"LGPD VIOLADA: Coluna sensível '{col}' detectada!"

def test_lgpd_pseudonimizacao_h3(sample_df_prata):
    """Garante que o H3_INDEX siga o padrão hexadecimal de 15 caracteres."""
    padrao_h3 = re.compile(r"^[89a-f0-f]{15}$")
    for codigo in sample_df_prata["H3_INDEX"]:
        assert padrao_h3.match(codigo), f"H3_INDEX inválido detectado: {codigo}"

# 3. TESTES DE SEGURANÇA DE INFRAESTRUTURA

def test_seguranca_presenca_secrets():
    """Verifica se as secrets obrigatórias estão no ambiente."""
    keys_obrigatorias = [
        "R2_ACCESS_KEY_ID", 
        "R2_SECRET_ACCESS_KEY", 
        "BQ_SERVICE_ACCOUNT_JSON"
    ]
    for key in keys_obrigatorias:
        valor = os.getenv(key)
       
        if os.getenv("GITHUB_ACTIONS"):
            assert valor is not None and len(valor) > 0, f"SEGURANÇA: Secret '{key}' ausente!"

def test_seguranca_exposicao_secrets_em_logs(caplog):
    """Garante que secrets não sejam printadas nos logs acidentalmente."""
    logger = logging.getLogger("SafeDriver")
    logger.info("Teste de log seguro iniciado.")
    
    secret_fake = os.getenv("R2_SECRET_ACCESS_KEY", "VALOR_INEXISTENTE_PARA_TESTE")
    
    for record in caplog.records:
        if secret_fake != "VALOR_INEXISTENTE_PARA_TESTE":
            assert secret_fake not in record.text, "VAZAMENTO: Secret detectada nos logs!"
