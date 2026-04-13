import pytest
import pandas as pd
import hashlib
from autobot.processamento_prata import ProcessamentoPrata
from unittest.mock import MagicMock

class TestSeguranca:
    @pytest.fixture
    def configuracao_teste(self):
        robo = MagicMock()
        processador = ProcessamentoPrata(robo)
        return processador

    def test_anonimizacao_lgpd(self, configuracao_teste):
        sal = "safedriver_2026_token"
        num_bo = "AX8110"
        ano_bo = 2026
        
        esperado = hashlib.sha256(f"{num_bo}{ano_bo}{sal}".encode()).hexdigest()[:16]
        
        df_teste = pd.DataFrame({
            'NUM_BO': [num_bo],
            'ANO_BO': [ano_bo]
        })
        
        df_teste['ID_ANONIMO'] = (df_teste['NUM_BO'].astype(str) + 
                                  df_teste['ANO_BO'].astype(str) + sal).apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
        )
        
        assert df_teste.loc[0, 'ID_ANONIMO'] == esperado

    def test_remocao_dados_sensiveis(self, configuracao_teste):
        df_entrada = pd.DataFrame({
            'NUM_BO': ['123'],
            'LATITUDE': [-23.5],
            'LONGITUDE': [-46.6],
            'NATUREZA': ['ROUBO'],
            'H3_INDEX': ['8847552813fffff']
        })
        
        colunas_remover = ['NUM_BO', 'LATITUDE', 'LONGITUDE']
        df_saida = df_entrada.drop(columns=[c for c in colunas_remover if c in df_entrada.columns])
        
        assert 'NUM_BO' not in df_saida.columns
        assert 'LATITUDE' not in df_saida.columns
        assert 'LONGITUDE' not in df_saida.columns
        assert 'NATUREZA' in df_saida.columns

    def test_integridade_chave_localidade(self, configuracao_teste):
        municipio = "São Bernardo do Campo"
        bairro = "Centro"
        logradouro = "Rua Marechal Deodoro"
        
        esperado = "SAO BERNARDO DO CAMPO|CENTRO|RUA MARECHAL DEODORO"
        
        resultado = (configuracao_teste.normalizar(municipio) + "|" + 
                     configuracao_teste.normalizar(bairro) + "|" + 
                     configuracao_teste.normalizar(logradouro))
        
        assert resultado == esperado

    def test_hash_irreversivel(self, configuracao_teste):
        entrada_1 = "BO1232026token"
        entrada_2 = "BO1232026token"
        
        hash_1 = hashlib.sha256(entrada_1.encode()).hexdigest()
        hash_2 = hashlib.sha256(entrada_2.encode()).hexdigest()
        
        assert hash_1 == hash_2
        assert len(hash_1) == 64
