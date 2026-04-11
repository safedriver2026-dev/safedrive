# tests/test_seguranca.py
"""
SafeDriver_Motor_V1.0.0 — Suite de Testes
Cobre: LGPD, contratos de schema, escores, períodos e integração R2.
"""
import sys
import os
import json
import hashlib
import pytest
import pandas as pd
import numpy as np
import polars as pl

# Garante que o módulo principal é encontrado
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autobot.motor_analise_preditiva import (
    calcular_escore,
    classificar_periodo,
    fator_periodo,
    anonimizar_campo,
    normalizar_texto,
    renomear_sinonimos,
    PESO_PENAL_BASE,
    MULTIPLICADOR_PERFIL,
    FATOR_PERIODO,
    COLUNAS_CRITICAS,
    SP_LAT_MIN, SP_LAT_MAX,
    SP_LON_MIN, SP_LON_MAX,
    NOME_SISTEMA,
    VERSAO_PIPELINE,
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def salt_teste():
    return "salt_teste_unitario_safedriver"

@pytest.fixture
def df_prata_minimo():
    """DataFrame mínimo válido para testar contratos."""
    return pd.DataFrame({
        "H3_INDEX":           ["8a2a100d2c37fff", "8a2a100d2c3ffff", "8a2a100d2c47fff"] * 200,
        "ESCORE":             np.random.uniform(1.0, 10.0, 600),
        "LATITUDE_F":         np.random.uniform(-23.6, -23.4, 600),
        "LONGITUDE_F":        np.random.uniform(-46.7, -46.5, 600),
        "RUBRICA":            ["ROUBO DE VEICULO", "FURTO", "HOMICIDIO DOLOSO"] * 200,
        "PERIODO_DIA":        ["NOITE", "MANHA", "MADRUGADA"] * 200,
        "IS_NOITE_MADRUGADA": [1, 0, 1] * 200,
        "IS_PATRIMONIO":      [1, 1, 0] * 200,
        "IS_VIOLENCIA_PESSOA":[0, 0, 1] * 200,
        "NOME_MUNICIPIO":     ["SAO PAULO", "CAMPINAS", "GUARULHOS"] * 200,
        "DATA_OCORRENCIA_BO": pd.date_range("2024-01-01", periods=600, freq="h"),
        "ANO_MES":            ["2024-01"] * 600,
    })

@pytest.fixture
def df_ouro_minimo(df_prata_minimo):
    """Agregação mínima simulando saída do construir_ouro."""
    agg = df_prata_minimo.groupby("H3_INDEX").agg(
        QTD_CRIMES            =("ESCORE",             "count"),
        ESCORE_TOTAL          =("ESCORE",             "sum"),
        ESCORE_MEDIO          =("ESCORE",             "mean"),
        ESCORE_GRAVIDADE_MAX  =("ESCORE",             "max"),
        LATITUDE_MEDIA        =("LATITUDE_F",         "mean"),
        LONGITUDE_MEDIA       =("LONGITUDE_F",        "mean"),
        PROP_NOITE_MADRUGADA  =("IS_NOITE_MADRUGADA", "mean"),
        PROP_PATRIMONIO       =("IS_PATRIMONIO",      "mean"),
        PROP_VIOLENCIA_PESSOA =("IS_VIOLENCIA_PESSOA","mean"),
    ).reset_index()
    agg["ESCORE_LAG2"]    = agg["ESCORE_TOTAL"] * 0.9
    agg["QTD_LAG2"]       = agg["QTD_CRIMES"]
    agg["ESCORE_VIZ_1"]   = agg["ESCORE_MEDIO"] * 0.95
    agg["ESCORE_VIZ_2"]   = agg["ESCORE_MEDIO"] * 0.85
    agg["QTD_CRIMES_VIZ"] = agg["QTD_CRIMES"] * 0.9
    agg["ESCORE_PREDITO"] = agg["ESCORE_TOTAL"] * 1.05
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# 1. IDENTIDADE DO SISTEMA
# ══════════════════════════════════════════════════════════════════════════════

class TestIdentidade:
    def test_nome_sistema(self):
        assert NOME_SISTEMA == "SafeDriver_Motor_V1.0.0", \
            f"Nome incorreto: {NOME_SISTEMA}"

    def test_versao_pipeline(self):
        assert VERSAO_PIPELINE == "5.0.0", \
            f"Versão incorreta: {VERSAO_PIPELINE}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. LGPD — ANONIMIZAÇÃO
# ══════════════════════════════════════════════════════════════════════════════

class TestLGPD:
    def test_anonimizar_retorna_sha256(self, salt_teste):
        resultado = anonimizar_campo("Rua das Flores, 123", salt_teste)
        assert len(resultado) == 64, "SHA-256 deve ter 64 caracteres hex"
        assert all(c in "0123456789abcdef" for c in resultado)

    def test_anonimizar_determinista(self, salt_teste):
        r1 = anonimizar_campo("teste", salt_teste)
        r2 = anonimizar_campo("teste", salt_teste)
        assert r1 == r2, "Mesmo input deve gerar mesmo hash"

    def test_anonimizar_salts_diferentes(self, salt_teste):
        r1 = anonimizar_campo("teste", salt_teste)
        r2 = anonimizar_campo("teste", "outro_salt")
        assert r1 != r2, "Salts diferentes devem gerar hashes diferentes"

    def test_anonimizar_valores_diferentes(self, salt_teste):
        r1 = anonimizar_campo("Rua A", salt_teste)
        r2 = anonimizar_campo("Rua B", salt_teste)
        assert r1 != r2, "Valores diferentes devem gerar hashes diferentes"

    def test_anonimizar_campo_vazio(self, salt_teste):
        resultado = anonimizar_campo("", salt_teste)
        assert len(resultado) == 64

    def test_anonimizar_nao_expoe_original(self, salt_teste):
        original  = "João da Silva"
        resultado = anonimizar_campo(original, salt_teste)
        assert "JOAO" not in resultado.upper()
        assert "SILVA" not in resultado.upper()


# ══════════════════════════════════════════════════════════════════════════════
# 3. PERÍODO DO DIA
# ══════════════════════════════════════════════════════════════════════════════

class TestPeriodoDia:
    @pytest.mark.parametrize("hora,esperado", [
        ("00:00", "MADRUGADA"),
        ("03:30", "MADRUGADA"),
        ("05:59", "MADRUGADA"),
        ("06:00", "MANHA"),
        ("11:59", "MANHA"),
        ("12:00", "TARDE"),
        ("17:59", "TARDE"),
        ("18:00", "NOITE"),
        ("23:59", "NOITE"),
    ])
    def test_classificar_periodo(self, hora, esperado):
        assert classificar_periodo(hora) == esperado

    def test_periodo_hora_invalida(self):
        resultado = classificar_periodo("XX:YY")
        assert resultado in ["MADRUGADA", "MANHA", "TARDE", "NOITE"]

    @pytest.mark.parametrize("hora,esperado", [
        ("02:00", 1.4),  # madrugada — agravante forte
        ("08:00", 1.0),  # manhã — neutro
        ("15:00", 1.0),  # tarde — neutro
        ("20:00", 1.3),  # noite — agravante
    ])
    def test_fator_periodo(self, hora, esperado):
        assert fator_periodo(hora) == esperado

    def test_manha_tarde_nao_sao_agravantes(self):
        assert fator_periodo("09:00") == 1.0, "Manhã deve ser neutro"
        assert fator_periodo("14:00") == 1.0, "Tarde deve ser neutro"

    def test_noite_madrugada_sao_agravantes(self):
        assert fator_periodo("02:00") > 1.0, "Madrugada deve ser agravante"
        assert fator_periodo("21:00") > 1.0, "Noite deve ser agravante"

    def test_madrugada_mais_agravante_que_noite(self):
        assert fator_periodo("03:00") > fator_periodo("21:00"), \
            "Madrugada deve ter fator maior que noite"


# ══════════════════════════════════════════════════════════════════════════════
# 4. ESCORE
# ══════════════════════════════════════════════════════════════════════════════

class TestEscore:
    def test_escore_positivo(self):
        assert calcular_escore("ROUBO DE VEICULO", "08:00") > 0

    def test_escore_noite_maior_que_manha(self):
        escore_manha  = calcular_escore("ROUBO", "09:00", "MOTORISTA")
        escore_noite  = calcular_escore("ROUBO", "21:00", "MOTORISTA")
        assert escore_noite > escore_manha, \
            "Crime à noite deve ter escore maior"

    def test_escore_madrugada_maior_que_noite(self):
        escore_noite     = calcular_escore("ROUBO", "20:00", "MOTORISTA")
        escore_madrugada = calcular_escore("ROUBO", "02:00", "MOTORISTA")
        assert escore_madrugada > escore_noite, \
            "Madrugada deve ter escore maior que noite"

    def test_escore_manha_igual_tarde(self):
        escore_manha = calcular_escore("FURTO", "09:00", "MOTORISTA")
        escore_tarde = calcular_escore("FURTO", "15:00", "MOTORISTA")
        assert escore_manha == escore_tarde, \
            "Manhã e tarde devem ter o mesmo escore (neutros)"

    def test_escore_homicidio_maior_que_furto(self):
        escore_hom  = calcular_escore("HOMICIDIO DOLOSO", "08:00")
        escore_furt = calcular_escore("FURTO",            "08:00")
        assert escore_hom > escore_furt

    def test_escore_latrocinio_maximo(self):
        escore = calcular_escore("LATROCINIO", "03:00", "MOTORISTA")
        assert escore >= 10.0

    def test_escore_crime_desconhecido(self):
        escore = calcular_escore("CRIME_DESCONHECIDO_XYZ", "10:00")
        assert escore > 0, "Crime desconhecido deve retornar escore base > 0"

    def test_escore_perfil_motociclista(self):
        escore_mot  = calcular_escore("ROUBO DE MOTOCICLETA", "21:00", "MOTOCICLISTA")
        escore_ped  = calcular_escore("ROUBO DE MOTOCICLETA", "21:00", "PEDESTRE")
        assert escore_mot > escore_ped, \
            "Motociclista deve ter escore maior para roubo de moto"

    def test_escore_retorna_float(self):
        resultado = calcular_escore("ROUBO", "20:00")
        assert isinstance(resultado, float)

    def test_escore_arredondado_4_casas(self):
        resultado = calcular_escore("ROUBO", "20:00")
        assert resultado == round(resultado, 4)


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONTRATOS DE SCHEMA — OURO
# ══════════════════════════════════════════════════════════════════════════════

class TestContratos:
    COLUNAS_OBRIGATORIAS_OURO = [
        "H3_INDEX",
        "QTD_CRIMES",
        "ESCORE_TOTAL",
        "ESCORE_MEDIO",
        "ESCORE_GRAVIDADE_MAX",
        "LATITUDE_MEDIA",
        "LONGITUDE_MEDIA",
        "PROP_NOITE_MADRUGADA",
        "PROP_PATRIMONIO",
        "PROP_VIOLENCIA_PESSOA",
        "ESCORE_LAG2",
        "QTD_LAG2",
        "ESCORE_VIZ_1",
        "ESCORE_VIZ_2",
        "QTD_CRIMES_VIZ",
        "ESCORE_PREDITO",
    ]

    def test_colunas_obrigatorias_presentes(self, df_ouro_minimo):
        faltando = [
            c for c in self.COLUNAS_OBRIGATORIAS_OURO
            if c not in df_ouro_minimo.columns
        ]
        assert not faltando, f"Colunas faltando no ouro: {faltando}"

    def test_h3_index_nao_nulo(self, df_ouro_minimo):
        assert df_ouro_minimo["H3_INDEX"].notna().all(), \
            "H3_INDEX não pode ter nulos"

    def test_h3_index_unico(self, df_ouro_minimo):
        assert df_ouro_minimo["H3_INDEX"].nunique() == len(df_ouro_minimo), \
            "H3_INDEX deve ser único por linha no ouro"

    def test_escore_total_positivo(self, df_ouro_minimo):
        assert (df_ouro_minimo["ESCORE_TOTAL"] > 0).all(), \
            "ESCORE_TOTAL deve ser positivo"

    def test_qtd_crimes_positivo(self, df_ouro_minimo):
        assert (df_ouro_minimo["QTD_CRIMES"] > 0).all(), \
            "QTD_CRIMES deve ser positivo"

    def test_prop_noite_madrugada_entre_0_1(self, df_ouro_minimo):
        col = df_ouro_minimo["PROP_NOITE_MADRUGADA"]
        assert (col >= 0).all() and (col <= 1).all(), \
            "PROP_NOITE_MADRUGADA deve estar entre 0 e 1"

    def test_prop_patrimonio_entre_0_1(self, df_ouro_minimo):
        col = df_ouro_minimo["PROP_PATRIMONIO"]
        assert (col >= 0).all() and (col <= 1).all()

    def test_prop_violencia_entre_0_1(self, df_ouro_minimo):
        col = df_ouro_minimo["PROP_VIOLENCIA_PESSOA"]
        assert (col >= 0).all() and (col <= 1).all()

    def test_latitude_dentro_sp(self, df_ouro_minimo):
        lat = df_ouro_minimo["LATITUDE_MEDIA"]
        assert (lat >= SP_LAT_MIN).all() and (lat <= SP_LAT_MAX).all(), \
            f"Latitude fora do Estado de SP: {lat.min():.4f} – {lat.max():.4f}"

    def test_longitude_dentro_sp(self, df_ouro_minimo):
        lon = df_ouro_minimo["LONGITUDE_MEDIA"]
        assert (lon >= SP_LON_MIN).all() and (lon <= SP_LON_MAX).all(), \
            f"Longitude fora do Estado de SP: {lon.min():.4f} – {lon.max():.4f}"

    def test_escore_gravidade_max_gte_medio(self, df_ouro_minimo):
        assert (
            df_ouro_minimo["ESCORE_GRAVIDADE_MAX"] >= df_ouro_minimo["ESCORE_MEDIO"]
        ).all(), "ESCORE_GRAVIDADE_MAX deve ser >= ESCORE_MEDIO"

    def test_escore_predito_nao_nulo(self, df_ouro_minimo):
        assert df_ouro_minimo["ESCORE_PREDITO"].notna().all(), \
            "ESCORE_PREDITO não pode ter NaN"

    def test_sem_infinitos(self, df_ouro_minimo):
        numericas = df_ouro_minimo.select_dtypes(include=[np.number])
        assert not np.isinf(numericas.values).any(), \
            "Ouro contém valores infinitos"

    def test_escore_lag2_nao_negativo(self, df_ouro_minimo):
        assert (df_ouro_minimo["ESCORE_LAG2"] >= 0).all(), \
            "ESCORE_LAG2 não pode ser negativo"

    def test_viz_1_nao_negativo(self, df_ouro_minimo):
        assert (df_ouro_minimo["ESCORE_VIZ_1"] >= 0).all()

    def test_viz_2_nao_negativo(self, df_ouro_minimo):
        assert (df_ouro_minimo["ESCORE_VIZ_2"] >= 0).all()

    def test_schema_polars_compativel(self, df_ouro_minimo):
        """Garante que o ouro pode ser convertido para Polars sem erro."""
        try:
            ouro_pl = pl.from_pandas(df_ouro_minimo)
            assert len(ouro_pl) == len(df_ouro_minimo)
        except Exception as e:
            pytest.fail(f"Conversão pandas→polars falhou: {e}")

    def test_schema_serializavel_json(self, df_ouro_minimo):
        """Garante que o schema pode ser serializado (BigQuery load)."""
        schema = {col: str(dtype) for col, dtype in df_ouro_minimo.dtypes.items()}
        try:
            json.dumps(schema)
        except Exception as e:
            pytest.fail(f"Schema não serializável: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. NORMALIZAÇÃO E SINÔNIMOS
# ══════════════════════════════════════════════════════════════════════════════

class TestNormalizacao:
    def test_normalizar_remove_acentos(self):
        assert normalizar_texto("São Paulo") == "SAO PAULO"

    def test_normalizar_maiusculas(self):
        assert normalizar_texto("roubo de veículo") == "ROUBO DE VEICULO"

    def test_normalizar_none(self):
        assert normalizar_texto(None) == ""

    def test_normalizar_numero(self):
        resultado = normalizar_texto(123)
        assert resultado == "123"

    def test_renomear_sinonimos(self):
        df = pl.DataFrame({"MUNICIPIO": ["SP"], "LAT": [1.0], "LON": [2.0]})
        df_renomeado = renomear_sinonimos(df)
        assert "NOME_MUNICIPIO" in df_renomeado.columns
        assert "LATITUDE"       in df_renomeado.columns
        assert "LONGITUDE"      in df_renomeado.columns

    def test_renomear_sem_sinonimos_conhecidos(self):
        df = pl.DataFrame({"COLUNA_ESTRANHA": ["x"]})
        df_renomeado = renomear_sinonimos(df)
        assert "COLUNA_ESTRANHA" in df_renomeado.columns


# ══════════════════════════════════════════════════════════════════════════════
# 7. SECRETS DE AMBIENTE
# ══════════════════════════════════════════════════════════════════════════════

class TestSecrets:
    SECRETS_OBRIGATORIOS = [
        "R2_ENDPOINT_URL",
        "R2_ACCESS_KEY_ID",
        "R2_SECRET_ACCESS_KEY",
        "R2_BUCKET_NAME",
        "BQ_PROJECT_ID",
        "BQ_DATASET_ID",
        "BQ_SERVICE_ACCOUNT_JSON",
        "LGPD_SALT",
        "DISCORD_SUCESSO",
        "DISCORD_ERRO",
    ]

    def test_secrets_presentes_no_ambiente(self):
        faltando = [s for s in self.SECRETS_OBRIGATORIOS if not os.environ.get(s)]
        if faltando:
            pytest.skip(
                f"Secrets não disponíveis em ambiente local (normal): {faltando}"
            )

    def test_lgpd_salt_nao_vazio_se_presente(self):
        salt = os.environ.get("LGPD_SALT", "")
        if salt:
            assert len(salt.strip()) >= 16, \
                "LGPD_SALT muito curto — mínimo 16 caracteres"

    def test_bq_project_nao_hardcoded(self):
        """Garante que nenhum valor hardcoded de projeto existe no código fonte."""
        caminho = os.path.join(
            os.path.dirname(__file__), "..", "autobot", "motor_analise_preditiva.py"
        )
        if os.path.exists(caminho):
            with open(caminho, "r", encoding="utf-8") as f:
                conteudo = f.read()
            assert "BQ_PROJECT_FIXO" not in conteudo, \
                "BQ_PROJECT_FIXO hardcoded encontrado no código — usar só o secret"


# ══════════════════════════════════════════════════════════════════════════════
# 8. PESOS E CONFIGURAÇÕES
# ══════════════════════════════════════════════════════════════════════════════

class TestPesos:
    def test_todos_pesos_positivos(self):
        for crime, peso in PESO_PENAL_BASE.items():
            assert peso > 0, f"Peso negativo/zero para: {crime}"

    def test_crimes_graves_peso_maximo(self):
        assert PESO_PENAL_BASE["HOMICIDIO DOLOSO"] == 10.0
        assert PESO_PENAL_BASE["LATROCINIO"]       == 10.0

    def test_furto_menor_que_roubo(self):
        assert PESO_PENAL_BASE["FURTO"] < PESO_PENAL_BASE["ROUBO"]

    def test_todos_multiplicadores_positivos(self):
        for perfil, crimes in MULTIPLICADOR_PERFIL.items():
            for crime, mult in crimes.items():
                assert mult > 0, \
                    f"Multiplicador inválido para {perfil}/{crime}: {mult}"

    def test_fator_madrugada_maior_que_noite(self):
        assert FATOR_PERIODO["MADRUGADA"] > FATOR_PERIODO["NOITE"]

    def test_manha_tarde_neutros(self):
        assert FATOR_PERIODO["MANHA"]  == 1.0
        assert FATOR_PERIODO["TARDE"]  == 1.0

    def test_perfis_conhecidos(self):
        perfis = set(MULTIPLICADOR_PERFIL.keys())
        assert "MOTORISTA"    in perfis
        assert "MOTOCICLISTA" in perfis
        assert "PEDESTRE"     in perfis
        assert "CICLISTA"     in perfis
