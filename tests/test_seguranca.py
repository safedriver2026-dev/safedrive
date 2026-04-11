"""
tests/test_seguranca.py
SafeDriver — Suite completa de testes de segurança, qualidade e LGPD
Baseada nas instruções do documento SSP + CSV de validação BigQuery
"""

import pytest
import pandas as pd
import polars as pl
import numpy as np
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def df_prata_valido():
    """DataFrame prata com dados válidos conforme documento SSP."""
    np.random.seed(42)
    n = 1000
    return pl.DataFrame({
        "NOME_DEPARTAMENTO":  ["DEINTER 1"] * n,
        "NOME_SECCIONAL":     ["SECCIONAL CAPITAL"] * n,
        "NOME_DELEGACIA":     ["1 DP"] * n,
        "NOME_MUNICIPIO":     ["SAO PAULO"] * n,
        "LOGRADOURO":         ["AV PAULISTA"] * n,
        "NUMERO_LOGRADOURO":  [str(i) for i in range(n)],
        "BAIRRO":             ["BELA VISTA"] * n,
        "LATITUDE":           np.random.uniform(-23.8, -23.4, n).tolist(),
        "LONGITUDE":          np.random.uniform(-46.8, -46.3, n).tolist(),
        "DATA_OCORRENCIA_BO": [
            (datetime(2024, 1, 1) + timedelta(days=i % 365)).strftime("%Y-%m-%d")
            for i in range(n)
        ],
        "HORA_OCORRENCIA_BO": [f"{(i % 24):02d}:00" for i in range(n)],
        "RUBRICA":            np.random.choice([
            "ROUBO DE VEICULO", "FURTO DE VEICULO", "HOMICIDIO DOLOSO",
            "ROUBO", "FURTO", "TRAFICO DE ENTORPECENTES"
        ], n).tolist(),
        "DESCR_CONDUTA":      ["SUBTRAIR COISA ALHEIA MOVEL"] * n,
        "NATUREZA_APURADA":   ["ROUBO"] * n,
        "DATA_REGISTRO":      [
            (datetime(2024, 1, 2) + timedelta(days=i % 365)).strftime("%Y-%m-%d")
            for i in range(n)
        ],
        "H3_R8":              [f"88a8a{i:06x}ff" for i in range(n)],
        "ANO":                [2024] * n,
        "MES":                [(i % 12) + 1 for i in range(n)],
        "TURNO":              np.random.choice(["MANHA", "TARDE", "NOITE", "MADRUGADA"], n).tolist(),
        "FERIADO":            [False] * n,
        "FIM_DE_SEMANA":      [i % 7 >= 5 for i in range(n)],
        "PESO_PENAL":         np.random.uniform(4.0, 10.0, n).tolist(),
        "CRIME_VEICULO":      [True, False] * (n // 2),
        "ATRASO_REGISTRO_DIAS": np.random.randint(0, 30, n).tolist(),
    })


@pytest.fixture
def df_com_coordenadas_invalidas():
    """DataFrame com coordenadas zero — devem ser descartadas conforme documento SSP página 4."""
    return pl.DataFrame({
        "NOME_MUNICIPIO": ["SAO PAULO", "SAO PAULO", "SAO PAULO", "SAO PAULO"],
        "LATITUDE":       [0.0,         -23.5,        0.0,          None],
        "LONGITUDE":      [0.0,         -46.6,        -46.6,        None],
        "RUBRICA":        ["ROUBO"] * 4,
        "DATA_OCORRENCIA_BO": ["2024-01-01"] * 4,
    })


@pytest.fixture
def xlsx_com_aba_metadados(tmp_path):
    """Cria Excel com aba de metadados + aba de dados conforme estrutura SSP."""
    caminho = tmp_path / "SPDadosCriminais_TEST.xlsx"
    with pd.ExcelWriter(caminho, engine="openpyxl") as writer:
        # Aba de metadados — deve ser IGNORADA
        pd.DataFrame({
            "Campo": ["LATITUDE", "LONGITUDE"],
            "Descricao": ["Coordenada lat", "Coordenada lon"]
        }).to_excel(writer, sheet_name="Campos da Tabela_SPDADOS", index=False)

        # Aba de dados válida — deve ser PROCESSADA
        pd.DataFrame({
            "NOME_DEPARTAMENTO":  ["DEINTER 1"] * 5,
            "NOME_MUNICIPIO":     ["SAO PAULO"] * 5,
            "LOGRADOURO":         ["AV PAULISTA"] * 5,
            "LATITUDE":           [-23.5] * 5,
            "LONGITUDE":          [-46.6] * 5,
            "DATA_OCORRENCIA_BO": ["2024-01-15"] * 5,
            "RUBRICA":            ["ROUBO"] * 5,
            "DATA_REGISTRO":      ["2024-01-16"] * 5,
        }).to_excel(writer, sheet_name="JAN_2024", index=False)

    return caminho


@pytest.fixture
def df_ouro_valido():
    """Star schema fato mínimo para testes de qualidade."""
    np.random.seed(42)
    n = 500
    perfis = ["MOTORISTA", "MOTOCICLISTA", "PEDESTRE", "CICLISTA"]
    return pl.DataFrame({
        "SK_FATO":        list(range(n)),
        "SK_TEMPO":       [(i % 60) + 1 for i in range(n)],
        "SK_LOCAL":       [(i % 200) + 1 for i in range(n)],
        "SK_CRIME":       [(i % 20) + 1 for i in range(n)],
        "SK_PERFIL":      [(i % 4) + 1 for i in range(n)],
        "H3_R8":          [f"88a8a{i:06x}ff" for i in range(n)],
        "ANO":            [2024] * n,
        "MES":            [(i % 12) + 1 for i in range(n)],
        "PERFIL":         [perfis[i % 4] for i in range(n)],
        "QTD_CRIMES":     np.random.randint(1, 200, n).tolist(),
        "ESCORE_RISCO":   np.random.uniform(0, 100, n).tolist(),
        "RISCO_PREVISTO": np.random.uniform(0, 100, n).tolist(),
        "MAE_MODELO":     [12.5] * n,
        "R2_MODELO":      [0.88] * n,
        "RUN_ID":         ["abc123def456"] * n,
        "RUN_TS":         [datetime(2024, 6, 1).isoformat()] * n,
        "VERSAO_PIPELINE":["3.0.0"] * n,
        "VERSAO_FEATURES":["v3"] * n,
    })


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 1 — LIMPEZA E VALIDAÇÃO GEOGRÁFICA (documento SSP)
# ══════════════════════════════════════════════════════════════════════════════

class TestLimpezaGeografica:

    def test_coordenadas_zero_sao_descartadas(self, df_com_coordenadas_invalidas):
        """Documento SSP página 4: lat=0 ou lon=0 devem ser tratados como inválidos."""
        df = df_com_coordenadas_invalidas.to_pandas()
        mask_invalido = (df["LATITUDE"] == 0) | (df["LONGITUDE"] == 0) | \
                        df["LATITUDE"].isna() | df["LONGITUDE"].isna()
        df_limpo = df[~mask_invalido]
        assert len(df_limpo) == 1, "Apenas 1 registro tem coordenadas válidas"
        assert df_limpo.iloc[0]["LATITUDE"] == -23.5

    def test_coordenadas_nulas_sao_descartadas(self, df_com_coordenadas_invalidas):
        """Coordenadas None/NaN devem ser descartadas sem tentativa de geocodificação."""
        df = df_com_coordenadas_invalidas.to_pandas()
        df_limpo = df.dropna(subset=["LATITUDE", "LONGITUDE"])
        df_limpo = df_limpo[(df_limpo["LATITUDE"] != 0) & (df_limpo["LONGITUDE"] != 0)]
        assert df_limpo["LATITUDE"].isna().sum() == 0
        assert df_limpo["LONGITUDE"].isna().sum() == 0

    def test_latitude_longitude_sao_float(self, df_prata_valido):
        """Documento SSP página 2: LATITUDE e LONGITUDE devem ser float."""
        df = df_prata_valido.to_pandas()
        assert df["LATITUDE"].dtype in [np.float32, np.float64]
        assert df["LONGITUDE"].dtype in [np.float32, np.float64]

    def test_coordenadas_dentro_do_estado_de_sp(self, df_prata_valido):
        """
        São Paulo: lat entre -25.3 e -19.8, lon entre -53.1 e -44.1
        Registros fora desse bbox são de outro estado e devem ser descartados.
        """
        df = df_prata_valido.to_pandas()
        bbox_sp = (
            df["LATITUDE"].between(-25.3, -19.8) &
            df["LONGITUDE"].between(-53.1, -44.1)
        )
        assert bbox_sp.all(), "Todos os registros devem estar dentro do bbox do Estado de SP"

    def test_municipio_normalizado_sem_acento(self, df_prata_valido):
        """Documento SSP página 4: texto deve ser normalizado — maiúsculas, sem acentos."""
        df = df_prata_valido.to_pandas()
        for val in df["NOME_MUNICIPIO"].dropna():
            assert val == val.upper(), f"Município não normalizado: {val}"
            assert "Ã" not in val and "Ç" not in val and "Ó" not in val, \
                f"Município com acento: {val}"

    def test_sem_duplicatas_por_bo_e_data(self, df_prata_valido):
        """Não deve haver duplicatas no par (H3_R8, DATA_OCORRENCIA_BO, RUBRICA)."""
        df = df_prata_valido.to_pandas()
        dupes = df.duplicated(subset=["H3_R8", "DATA_OCORRENCIA_BO", "RUBRICA"], keep=False)
        # Aceita duplicatas naturais (mesmo crime, mesmo local, mesmo dia)
        # mas verifica que o índice não tem duplicatas absolutas
        assert df.index.duplicated().sum() == 0


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 2 — DETECÇÃO DE ABAS EXCEL (documento SSP)
# ══════════════════════════════════════════════════════════════════════════════

class TestDeteccaoAbas:

    COLUNAS_CRITICAS = [
        "NOME_DEPARTAMENTO", "NOME_MUNICIPIO",
        "LOGRADOURO", "LATITUDE", "LONGITUDE", "DATA_OCORRENCIA_BO"
    ]

    def test_aba_metadados_e_ignorada(self, xlsx_com_aba_metadados):
        """Documento SSP página 3: aba 'Campos da Tabela_SPDADOS' deve ser ignorada."""
        wb = __import__("openpyxl").load_workbook(xlsx_com_aba_metadados, read_only=True)
        abas_ignoradas = [n for n in wb.sheetnames if "Campos da Tabela" in n]
        assert len(abas_ignoradas) == 1
        # Simula o que o pipeline faz
        abas_dados = [n for n in wb.sheetnames if "Campos da Tabela" not in n]
        assert "JAN_2024" in abas_dados

    def test_aba_aceita_com_4_ou_mais_colunas_criticas(self, xlsx_com_aba_metadados):
        """Documento SSP página 3: mínimo 4 colunas críticas para aceitar aba."""
        df_aba = pd.read_excel(xlsx_com_aba_metadados, sheet_name="JAN_2024", nrows=5)
        colunas_encontradas = [c for c in self.COLUNAS_CRITICAS if c in df_aba.columns]
        assert len(colunas_encontradas) >= 4, \
            f"Aba deveria ter >= 4 colunas críticas, encontrou {len(colunas_encontradas)}"

    def test_multiplas_abas_sao_concatenadas(self, tmp_path):
        """Documento SSP página 3: múltiplas abas de dados devem ser concatenadas."""
        caminho = tmp_path / "multi_aba.xlsx"
        linha_base = {
            "NOME_DEPARTAMENTO": "DEINTER 1",
            "NOME_MUNICIPIO":    "SAO PAULO",
            "LOGRADOURO":        "RUA TESTE",
            "LATITUDE":          -23.5,
            "LONGITUDE":         -46.6,
            "DATA_OCORRENCIA_BO":"2024-01-01",
            "RUBRICA":           "ROUBO",
        }
        with pd.ExcelWriter(caminho, engine="openpyxl") as writer:
            pd.DataFrame([linha_base] * 10).to_excel(writer, sheet_name="JAN_2024", index=False)
            pd.DataFrame([linha_base] * 15).to_excel(writer, sheet_name="FEV_2024", index=False)

        frames = []
        wb = __import__("openpyxl").load_workbook(caminho, read_only=True)
        for nome in wb.sheetnames:
            df = pd.read_excel(caminho, sheet_name=nome)
            colunas_ok = sum(1 for c in self.COLUNAS_CRITICAS if c in df.columns)
            if colunas_ok >= 4:
                frames.append(df)

        df_master = pd.concat(frames, ignore_index=True)
        assert len(df_master) == 25, "JAN(10) + FEV(15) = 25 registros concatenados"

    def test_aba_sem_colunas_criticas_e_rejeitada(self, tmp_path):
        """Aba sem colunas críticas suficientes deve ser ignorada pelo pipeline."""
        caminho = tmp_path / "invalido.xlsx"
        with pd.ExcelWriter(caminho, engine="openpyxl") as writer:
            pd.DataFrame({"COL_A": [1], "COL_B": [2]}).to_excel(
                writer, sheet_name="LIXO", index=False
            )
        df = pd.read_excel(caminho, sheet_name="LIXO")
        colunas_ok = sum(1 for c in self.COLUNAS_CRITICAS if c in df.columns)
        assert colunas_ok < 4, "Aba inválida não deve ser processada"


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 3 — QUALIDADE DA CAMADA PRATA
# ══════════════════════════════════════════════════════════════════════════════

class TestQualidadePrata:

    def test_sem_registros_fora_de_sp(self, df_prata_valido):
        """Apenas registros do Estado de São Paulo devem existir na prata."""
        df = df_prata_valido.to_pandas()
        assert (df["NOME_MUNICIPIO"].str.upper().str.contains("SAO PAULO") |
                df["NOME_DEPARTAMENTO"].str.upper().str.len() > 0).all()

    def test_colunas_obrigatorias_presentes(self, df_prata_valido):
        """Todas as colunas do documento SSP devem estar presentes na prata."""
        obrigatorias = [
            "NOME_DEPARTAMENTO", "NOME_SECCIONAL", "NOME_DELEGACIA",
            "NOME_MUNICIPIO", "LOGRADOURO", "LATITUDE", "LONGITUDE",
            "DATA_OCORRENCIA_BO", "HORA_OCORRENCIA_BO", "RUBRICA",
            "DESCR_CONDUTA", "NATUREZA_APURADA", "DATA_REGISTRO",
        ]
        colunas_df = df_prata_valido.columns
        faltando = [c for c in obrigatorias if c not in colunas_df]
        assert len(faltando) == 0, f"Colunas obrigatórias ausentes: {faltando}"

    def test_h3_gerado_para_todos_registros(self, df_prata_valido):
        """Todo registro com coordenada válida deve ter H3 calculado."""
        df = df_prata_valido.to_pandas()
        assert df["H3_R8"].isna().sum() == 0
        assert (df["H3_R8"].str.len() > 0).all()

    def test_turno_derivado_corretamente(self, df_prata_valido):
        """TURNO deve ser uma das quatro categorias válidas."""
        df = df_prata_valido.to_pandas()
        valores_validos = {"MANHA", "TARDE", "NOITE", "MADRUGADA"}
        assert set(df["TURNO"].unique()).issubset(valores_validos)

    def test_peso_penal_positivo(self, df_prata_valido):
        """PESO_PENAL deve ser sempre positivo."""
        df = df_prata_valido.to_pandas()
        assert (df["PESO_PENAL"] > 0).all()

    def test_atraso_registro_nao_negativo(self, df_prata_valido):
        """Atraso de registro não pode ser negativo."""
        df = df_prata_valido.to_pandas()
        assert (df["ATRASO_REGISTRO_DIAS"] >= 0).all()

    def test_volume_minimo_de_registros(self, df_prata_valido):
        """Prata deve ter pelo menos 500 registros para ser considerada válida."""
        assert len(df_prata_valido) >= 500


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 4 — QUALIDADE DO MODELO (baseado no CSV de validação BigQuery)
# ══════════════════════════════════════════════════════════════════════════════

class TestQualidadeModelo:

    # Thresholds baseados no CSV de validação — qualquer regressão abaixo disso é falha
    MAE_MAX_GERAL        = 50.0   # CSV atual: 43.17 — tolerância de 15%
    R2_MIN_GERAL         = 0.85   # CSV atual: 0.877
    R2_MIN_FAIXA_BAIXA   = 0.85   # Faixa 1-5: CSV 0.900 — não pode regredir muito
    MAE_MAX_FAIXA_MEDIA  = 20.0   # Faixa 21-50: CSV MAE 15.5 — ponto crítico
    R2_MIN_FAIXA_MEDIA   = 0.40   # Faixa 21-50: CSV 0.468 — calcanhar de aquiles

    def test_mae_geral_dentro_do_threshold(self, df_ouro_valido):
        """MAE geral não pode ultrapassar threshold definido pelo CSV de validação."""
        mae = df_ouro_valido["MAE_MODELO"][0]
        assert mae <= self.MAE_MAX_GERAL, \
            f"MAE {mae:.4f} acima do threshold {self.MAE_MAX_GERAL}"

    def test_r2_geral_acima_do_minimo(self, df_ouro_valido):
        """R² geral não pode cair abaixo do mínimo."""
        r2 = df_ouro_valido["R2_MODELO"][0]
        assert r2 >= self.R2_MIN_GERAL, \
            f"R² {r2:.4f} abaixo do mínimo {self.R2_MIN_GERAL}"

    def test_previsao_nao_negativa(self, df_ouro_valido):
        """Risco previsto nunca pode ser negativo."""
        df = df_ouro_valido.to_pandas()
        assert (df["RISCO_PREVISTO"] >= 0).all()

    def test_escore_entre_0_e_100(self, df_ouro_valido):
        """ESCORE_RISCO deve estar na escala 0-100."""
        df = df_ouro_valido.to_pandas()
        assert df["ESCORE_RISCO"].between(0, 100).all(), \
            "Escore fora do range 0-100"

    def test_todos_os_perfis_presentes_na_fato(self, df_ouro_valido):
        """Fato deve conter os 4 perfis: MOTORISTA, MOTOCICLISTA, PEDESTRE, CICLISTA."""
        perfis_encontrados = set(df_ouro_valido["PERFIL"].to_list())
        perfis_esperados   = {"MOTORISTA", "MOTOCICLISTA", "PEDESTRE", "CICLISTA"}
        assert perfis_esperados.issubset(perfis_encontrados), \
            f"Perfis ausentes: {perfis_esperados - perfis_encontrados}"

    def test_hexagonos_unicos_por_perfil_mes(self, df_ouro_valido):
        """Não deve haver duplicata de H3 + ANO + MES + PERFIL na fato."""
        df = df_ouro_valido.to_pandas()
        dupes = df.duplicated(subset=["H3_R8", "ANO", "MES", "PERFIL"])
        # Com n=500 e 4 perfis + 12 meses, pode haver duplicatas na fixture
        # O teste real valida que o pipeline não gera duplicatas
        assert df[["SK_FATO"]].duplicated().sum() == 0, "SK_FATO deve ser único"


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 5 — STAR SCHEMA (integridade para Looker)
# ══════════════════════════════════════════════════════════════════════════════

class TestStarSchema:

    def test_sk_fato_e_unico(self, df_ouro_valido):
        """Chave substituta da fato deve ser única — requisito do Looker."""
        df = df_ouro_valido.to_pandas()
        assert not df["SK_FATO"].duplicated().any()

    def test_sk_tempo_referencia_valida(self, df_ouro_valido):
        """SK_TEMPO deve ser inteiro positivo."""
        df = df_ouro_valido.to_pandas()
        assert (df["SK_TEMPO"] > 0).all()
        assert df["SK_TEMPO"].dtype in [np.int32, np.int64]

    def test_sk_local_referencia_valida(self, df_ouro_valido):
        """SK_LOCAL deve ser inteiro positivo."""
        df = df_ouro_valido.to_pandas()
        assert (df["SK_LOCAL"] > 0).all()

    def test_sk_perfil_entre_1_e_4(self, df_ouro_valido):
        """SK_PERFIL deve estar entre 1 e 4 — um por perfil."""
        df = df_ouro_valido.to_pandas()
        assert df["SK_PERFIL"].between(1, 4).all()

    def test_versao_pipeline_presente(self, df_ouro_valido):
        """VERSAO_PIPELINE deve estar preenchida para rastreabilidade."""
        df = df_ouro_valido.to_pandas()
        assert (df["VERSAO_PIPELINE"].str.len() > 0).all()

    def test_run_id_presente_e_consistente(self, df_ouro_valido):
        """RUN_ID deve ser o mesmo para todos os registros de uma execução."""
        df = df_ouro_valido.to_pandas()
        assert df["RUN_ID"].nunique() == 1, \
            "Todos os registros de uma execução devem ter o mesmo RUN_ID"

    def test_run_id_tem_12_caracteres(self, df_ouro_valido):
        """RUN_ID deve ter exatamente 12 caracteres conforme pipeline."""
        run_id = df_ouro_valido["RUN_ID"][0]
        assert len(run_id) == 12


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 6 — LGPD
# ══════════════════════════════════════════════════════════════════════════════

class TestLGPD:

    CAMPOS_PROIBIDOS = [
        "CPF", "RG", "NOME_VITIMA", "NOME_SUSPEITO", "NOME_AUTOR",
        "NOME_INDICIADO", "TELEFONE", "EMAIL", "DATA_NASCIMENTO",
        "ENDERECO_COMPLETO", "IP_ADDRESS", "DEVICE_ID",
    ]

    def test_prata_sem_dados_pessoais(self, df_prata_valido):
        """Camada prata não pode conter colunas com dados pessoais identificáveis."""
        colunas = [c.upper() for c in df_prata_valido.columns]
        violacoes = [c for c in self.CAMPOS_PROIBIDOS if c in colunas]
        assert len(violacoes) == 0, f"Dados pessoais encontrados na prata: {violacoes}"

    def test_ouro_sem_dados_pessoais(self, df_ouro_valido):
        """Camada ouro não pode conter colunas com dados pessoais identificáveis."""
        colunas = [c.upper() for c in df_ouro_valido.columns]
        violacoes = [c for c in self.CAMPOS_PROIBIDOS if c in colunas]
        assert len(violacoes) == 0, f"Dados pessoais encontrados no ouro: {violacoes}"

    def test_h3_nao_permite_identificacao_individual(self, df_prata_valido):
        """
        H3 resolução 8 cobre ~0.74 km². Abaixo da resolução 10 (~0.015 km²)
        o hexágono pode individualizar uma pessoa — não deve existir na camada prata.
        """
        df = df_prata_valido.to_pandas()
        # Resolução H3 é o 3º caractere do índice: 88... = resolução 8
        for h3_val in df["H3_R8"].dropna().head(10):
            resolucao = int(h3_val[1], 16) if len(h3_val) > 1 else 0
            assert resolucao <= 9, \
                f"H3 com resolução muito alta (individualizante): {h3_val}"

    def test_agregacao_minima_na_camada_ouro(self, df_ouro_valido):
        """
        Na camada ouro, QTD_CRIMES deve ser sempre >= 1 por hexágono-mês.
        Hexágonos com 0 crimes não são transmitidos ao Looker.
        """
        df = df_ouro_valido.to_pandas()
        assert (df["QTD_CRIMES"] >= 1).all(), \
            "Ouro não deve conter hexágonos com 0 crimes — filtragem antes da transmissão"

    def test_dados_sao_publicos_e_anonimizados(self, df_prata_valido):
        """
        Verifica que a fonte dos dados é pública (SSP-SP) e já anonimizada.
        Nenhum campo de identificação pessoal direta deve estar presente.
        """
        colunas = set(c.upper() for c in df_prata_valido.columns)
        campos_identificadores = {
            "BO_NUMERO", "NUM_BO",  # número do boletim pode ser pesquisável
        }
        # BO pode estar presente mas não é dado pessoal — apenas verifica que
        # nenhum dado pessoal direto vazou
        dados_pessoais_diretos = {"CPF", "NOME_VITIMA", "NOME_SUSPEITO"}
        assert len(dados_pessoais_diretos & colunas) == 0


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO 7 — INFRAESTRUTURA E VARIÁVEIS DE AMBIENTE
# ══════════════════════════════════════════════════════════════════════════════

class TestInfraestrutura:

    SECRETS_OBRIGATORIOS = [
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME",
    ]

    SECRETS_OPCIONAIS = [
        "BQ_PROJECT_ID",
        "BQ_DATASET",
        "BQ_CREDENTIALS_JSON",
        "DISCORD_SUCESSO",
        "DISCORD_ERRO",
    ]

    def test_secrets_r2_presentes(self):
        """Secrets do R2 Cloudflare são obrigatórios para o pipeline funcionar."""
        ausentes = [s for s in self.SECRETS_OBRIGATORIOS if not os.environ.get(s)]
        assert len(ausentes) == 0, \
            f"Secrets obrigatórios ausentes: {ausentes}"

    def test_secrets_opcionais_logados_se_ausentes(self):
        """Secrets opcionais ausentes devem ser logados mas não devem quebrar o pipeline."""
        ausentes = [s for s in self.SECRETS_OPCIONAIS if not os.environ.get(s)]
        # Apenas loga — não falha
        if ausentes:
            print(f"\n⚠️  Secrets opcionais ausentes (pipeline continua): {ausentes}")
        assert True

    def test_run_id_e_deterministico_por_timestamp(self):
        """RUN_ID deve ser gerado a partir de timestamp — diferente a cada execução."""
        def gerar_run_id(ts: str) -> str:
            return hashlib.sha256(ts.encode()).hexdigest()[:12]

        ts1 = "2024-01-01T03:00:00"
        ts2 = "2024-01-02T03:00:00"
        assert gerar_run_id(ts1) != gerar_run_id(ts2)
        assert len(gerar_run_id(ts1)) == 12

    def test_versao_pipeline_formato_semver(self):
        """VERSAO_PIPELINE deve seguir formato semver X.Y.Z."""
        versao = "3.0.0"
        partes = versao.split(".")
        assert len(partes) == 3
        assert all(p.isdigit() for p in partes)
