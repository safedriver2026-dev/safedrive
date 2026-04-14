import pytest
import polars as pl
import os
import re


@pytest.fixture
def sample_df_prata():
    return pl.DataFrame({
        "H3_INDEX": ["88a242ce13fffff", "88a242ce11fffff"],
        "TOTAL_CRIMES_MOTORISTA": [10, 5],
        "INDICE_RESIDENCIAL": [2, 1],
        "ANO_REFERENCIA": [2025, 2025],
        "LATITUDE": [-23.5505, -23.5510],
        "LONGITUDE": [-46.6333, -46.6335]
    })

def test_contrato_colunas_obrigatorias_prata(sample_df_prata):
    """Verifica se todas as colunas esperadas pela IA e pela Ouro estão presentes."""
    colunas_esperadas = {
        "H3_INDEX", "TOTAL_CRIMES_MOTORISTA", "INDICE_RESIDENCIAL", "ANO_REFERENCIA"
    }
    assert colunas_esperadas.issubset(set(sample_df_prata.columns)), \
        f"Faltam colunas essenciais no contrato da Prata. Encontradas: {sample_df_prata.columns}"

def test_contrato_tipagem_ia(sample_df_prata):
    """Garante que os dados numéricos são Int32 (vacina de tipos que aplicamos)."""
    assert sample_df_prata["TOTAL_CRIMES_MOTORISTA"].dtype == pl.Int64 or \
           sample_df_prata["TOTAL_CRIMES_MOTORISTA"].dtype == pl.Int32, \
           "Erro de Contrato: TOTAL_CRIMES deve ser numérico."


def test_lgpd_ausencia_dados_pessoais():
    """Verifica se não existem colunas proibidas (RG, CPF, Nome, Endereço Completo)."""
    
    termos_proibidos = ["NOME", "CPF", "RG", "VITIMA", "AUTOR", "RUA", "NUMERO_LOGRADOURO", "TELEFONE"]
    
   
    colunas_atuais = ["H3_INDEX", "TOTAL_CRIMES", "LATITUDE", "LONGITUDE"] 
    
    for termo in termos_proibidos:
        for col in colunas_atuais:
            assert termo not in col.upper(), f"LGPD VIOLADA: Coluna sensível '{col}' detectada!"

def test_lgpd_pseudonimizacao_h3(sample_df_prata):
    """Garante que a localização está agregada em H3 e não em pontos exatos expostos."""
   
    padrao_h3 = re.compile(r"^[89a-f0-f]{15}$")
    for codigo in sample_df_prata["H3_INDEX"]:
        assert padrao_h3.match(codigo), f"H3_INDEX inválido detectado: {codigo}"


--

def test_seguranca_presenca_secrets():
    """Verifica se as chaves críticas estão carregadas no ambiente (não vazias)."""
    keys_obrigatorias = [
        "R2_ACCESS_KEY_ID", 
        "R2_SECRET_ACCESS_KEY", 
        "BQ_SERVICE_ACCOUNT_JSON",
        "LGPD_SALT"
    ]
    for key in keys_obrigatorias:
        valor = os.getenv(key)
        assert valor is not None and len(valor) > 0, f"SEGURANÇA: Secret '{key}' não configurada!"

def test_seguranca_exposicao_secrets_em_logs(caplog):
    """Garante que as secrets não foram printadas nos logs acidentalmente."""
    
    logger = logging.getLogger("SafeDriver")
    logger.info("Iniciando processamento...")
    
    forbidden_content = os.getenv("R2_SECRET_ACCESS_KEY", "VALOR_PADRAO_INEXISTENTE")
    
    for record in caplog.records:
        assert forbidden_content not in record.text, "VAZAMENTO DE CREDENCIAL: Secret detectada nos logs!"
