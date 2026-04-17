import pytest
import polars as pl
import os
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(module)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfiguracaoTestes:
    COLUNAS_ESSENCIAIS_PRATA = {
        "H3_INDEX", "INDICE_GRAVIDADE", "ANO_REF", 
        "CONTAGIO_PONDERADO", "PRESSAO_RISCO_LOCAL",
        "NM_MUN", "PERIODO_DIA"
    }
    
    TERMOS_PROIBIDOS_PRIVACIDADE = [
        "NOME", "CPF", "RG", "VITIMA", "AUTOR", 
        "TELEFONE", "ENDERECO", "LATITUDE", "LONGITUDE"
    ]
    
    COLUNAS_SIMULACAO_OURO = [
        "H3_INDEX", "SCORE_RISCO_FINAL", "NM_BAIRRO", 
        "NM_MUN", "PERIODO_DIA", "DT_REF"
    ]
    
    PADRAO_VALIDACAO_H3 = re.compile(r"^[89a-f0-f]{15}$")
    
    CHAVES_OBRIGATORIAS_AMBIENTE = [
        "R2_ACCESS_KEY_ID", 
        "R2_SECRET_ACCESS_KEY", 
        "BQ_SERVICE_ACCOUNT_JSON"
    ]

@pytest.fixture
def amostra_dados_prata() -> pl.DataFrame:
    return pl.DataFrame({
        "H3_INDEX": ["89a81001d83ffff", "89a81ed28a3ffff"],
        "TOTAL_CRIMES": [10, 5],
        "INDICE_GRAVIDADE": [45.0, 12.5],
        "NM_BAIRRO": ["Jardim Aeroporto", "Centro"],
        "NM_MUN": ["SAO PAULO", "LIMEIRA"],
        "DENSIDADE": [12000.5, 8500.2],
        "ANO_REF": [2026, 2026],
        "CONTAGIO_PONDERADO": [1.5, 0.8],
        "PRESSAO_RISCO_LOCAL": [0.00012, 0.00009],
        "PERIODO_DIA": ["NOITE", "MANHA"],
        "IS_PAGAMENTO": [1, 0]
    })

def test_validacao_contrato_dados(amostra_dados_prata: pl.DataFrame):
    colunas_atuais = set(amostra_dados_prata.columns)
    colunas_em_falta = ConfiguracaoTestes.COLUNAS_ESSENCIAIS_PRATA - colunas_atuais
    assert not colunas_em_falta, f"Falha de integridade. Colunas em falta: {colunas_em_falta}"

def test_consistencia_matriz_gravidade(amostra_dados_prata: pl.DataFrame):
    assert amostra_dados_prata["INDICE_GRAVIDADE"].min() >= 0
    assert (amostra_dados_prata["INDICE_GRAVIDADE"] >= amostra_dados_prata["TOTAL_CRIMES"]).all()

def test_conformidade_privacidade_dados():
    for termo in ConfiguracaoTestes.TERMOS_PROIBIDOS_PRIVACIDADE:
        for coluna in ConfiguracaoTestes.COLUNAS_SIMULACAO_OURO:
            assert termo not in coluna.upper(), f"Infracao de privacidade detectada na coluna: '{coluna}'"

def test_integridade_identificadores_geograficos(amostra_dados_prata: pl.DataFrame):
    for codigo in amostra_dados_prata["H3_INDEX"]:
        assert ConfiguracaoTestes.PADRAO_VALIDACAO_H3.match(codigo), f"Identificador geografico invalido: {codigo}"

def test_disponibilidade_credenciais_ambiente():
    if os.getenv("GITHUB_ACTIONS"):
        for chave in ConfiguracaoTestes.CHAVES_OBRIGATORIAS_AMBIENTE:
            valor = os.getenv(chave)
            assert valor, f"Credencial de ambiente nao encontrada: {chave}"

def test_opacidade_registos_auditoria(caplog):
    chave_secreta = os.getenv("R2_SECRET_ACCESS_KEY")
    if chave_secreta:
        for registo in caplog.records:
            assert chave_secreta not in registo.text, "Exposicao de credencial nos registos de auditoria"
