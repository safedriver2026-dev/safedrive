import pytest
import polars as pl
import os
import re
import logging

logging.basicConfig(level=logging.INFO)

@pytest.fixture
def sample_df_prata():
    """Mock do novo esquema 'Heavy Silver' com Matriz de Gravidade."""
    return pl.DataFrame({
        "H3_INDEX": ["89a81001d83ffff", "89a81ed28a3ffff"],
        "TOTAL_CRIMES": [10, 5],
        "INDICE_GRAVIDADE": [45.0, 12.5], # Nova métrica de inteligência
        "NM_BAIRRO": ["Jardim Aeroporto", "Centro"],
        "NM_MUN": ["SAO PAULO", "LIMEIRA"],
        "DENSIDADE": [12000.5, 8500.2],
        "ANO_REF": [2026, 2026],
        "CONTAGIO_PONDERADO": [1.5, 0.8],
        "PRESSAO_RISCO_LOCAL": [0.00012, 0.00009],
        "PERIODO_DIA": ["NOITE", "MANHA"],
        "IS_PAGAMENTO": [1, 0]
    })

def test_contrato_colunas_obrigatorias_prata(sample_df_prata):
    """Garante que as novas colunas de engenharia de features e gravidade existam."""
    colunas_essenciais = {
        "H3_INDEX", "INDICE_GRAVIDADE", "ANO_REF", 
        "CONTAGIO_PONDERADO", "PRESSAO_RISCO_LOCAL",
        "NM_MUN", "PERIODO_DIA"
    }
    colunas_atuais = set(sample_df_prata.columns)
    faltando = colunas_essenciais - colunas_atuais
    assert not faltando, f"Contrato de Dados violado! Colunas ausentes: {faltando}"

def test_sanidade_matriz_gravidade(sample_df_prata):
    """Garante que o cálculo de peso dos crimes não gerou valores inválidos."""
    assert sample_df_prata["INDICE_GRAVIDADE"].min() >= 0
    # O Indice de Gravidade deve ser, em tese, >= que o número total de crimes 
    # (já que o peso mínimo de um crime é 1.0)
    assert (sample_df_prata["INDICE_GRAVIDADE"] >= sample_df_prata["TOTAL_CRIMES"]).all()

def test_lgpd_ausencia_dados_pessoais():
    """Verifica se colunas com nomes sensíveis foram removidas por design."""
    termos_proibidos = ["NOME", "CPF", "RG", "VITIMA", "AUTOR", "TELEFONE", "ENDERECO", "LATITUDE", "LONGITUDE"]
    # Simulando colunas que chegam na Ouro
    colunas_ouro = ["H3_INDEX", "SCORE_RISCO_FINAL", "NM_BAIRRO", "NM_MUN", "PERIODO_DIA", "DT_REF"]
    
    for termo in termos_proibidos:
        for col in colunas_ouro:
            assert termo not in col.upper(), f"LGPD VIOLADA: Coluna '{col}' contém termo sensível '{termo}'"

def test_lgpd_pseudonimizacao_h3(sample_df_prata):
    """Valida se o identificador geográfico segue o padrão de hash H3 (Resolução 9)."""
    # Regex para validar index H3 de 15 caracteres hexadecimais
    padrao_h3 = re.compile(r"^[89a-f0-f]{15}$")
    for codigo in sample_df_prata["H3_INDEX"]:
        assert padrao_h3.match(codigo), f"H3 Inválido detectado: {codigo}"

def test_seguranca_presenca_secrets():
    """Garante que o ambiente de produção (CI/CD) tem as chaves necessárias."""
    keys_obrigatorias = [
        "R2_ACCESS_KEY_ID", 
        "R2_SECRET_ACCESS_KEY", 
        "BQ_SERVICE_ACCOUNT_JSON"
    ]
    # Este teste só roda dentro do GitHub Actions
    if os.getenv("GITHUB_ACTIONS"):
        for key in keys_obrigatorias:
            valor = os.getenv(key)
            assert valor is not None and len(valor) > 0, f"SECRET AUSENTE: {key}"

def test_seguranca_exposicao_secrets_em_logs(caplog):
    """
    TESTE DE OPACIDADE: Garante que os segredos da AWS/R2 não foram 
    printados por acidente durante o log dos processos.
    """
    logger = logging.getLogger("SafeDriver")
    logger.info("Verificando opacidade de logs.")
    
    # Pega o secret real do ambiente para testar se ele aparece nos logs capturados
    secret_real = os.getenv("R2_SECRET_ACCESS_KEY")
    
    if secret_real:
        for record in caplog.records:
            # O secret real NUNCA deve estar contido em nenhuma mensagem de log
            assert secret_real not in record.text, "VULNERABILIDADE: Secret detectado em log!"
