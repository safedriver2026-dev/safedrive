import pytest
import pandas as pd
import os
# Importa o motor unificado e estudado
from autobot.autobot_engine import MotorSeguranca

def test_geracao_camada_ouro():
    """Valida se a transformação da Camada Bruta para a Ouro gera os arquivos do Esquema Estrela"""
    motor = MotorSeguranca(persistencia=False)
    dados_exemplo = pd.DataFrame({
        'LATITUDE': ['-23.5505'], 
        'LONGITUDE': ['-46.6333'],
        'HORA_OCORRENCIA_BO': ['19:00'], 
        'NATUREZA_APURADA': ['ROUBO DE CARGA'],
        'DATA_OCORRENCIA_BO': ['2026-03-18']
    })
    
    # Executa a lógica de inteligência
    resultado = motor._gerar_camada_ouro(dados_exemplo)
    
    # Verifica se a tabela de fatos foi criada na arquitetura de medalhão
    assert not resultado.empty
    assert os.path.exists('datalake/camada_ouro_refinada/esquema_estrela/fato_risco.csv')

def test_validacao_pesos_ia():
    """Garante que a inteligência de captura de crimes está atribuindo a gravidade correta"""
    motor = MotorSeguranca(persistencia=False)
    
    # Testa a captura de crime de altíssima gravidade (Latrocínio)
    serie_teste = pd.Series({'NATUREZA': 'LATROCINIO NA RUA'})
    
    # Chamada corrigida para o nome atual: _definir_peso
    assert motor._definir_peso(serie_teste) == 10.0
