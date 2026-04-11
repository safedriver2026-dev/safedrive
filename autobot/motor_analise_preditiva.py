import sys, os, json, hashlib
import pytest
import pandas as pd
import numpy as np
import polars as pl
from unittest.mock import patch, MagicMock
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from autobot.motor_analise_preditiva import (
    NOME_SISTEMA,
    PERFIS,
    PESO_PENAL_BASE,
    MULTIPLICADOR_PERFIL,
    FATOR_PERIODO,
    COLUNAS_CRITICAS,
    SINONIMOS,
    SP_LAT_MIN, SP_LAT_MAX,
    SP_LON_MIN, SP_LON_MAX,
    SSP_URL_TEMPLATE,
    SSP_MAX_TENTATIVAS,
    calcular_escore,
    calcular_escores_todos_perfis,
    classificar_periodo,
    fator_periodo,
    anonimizar_campo,
    normalizar_texto,
    renomear_sinonimos,
    baixar_ssp,
    TrackingSSP,
    SafeDriver,
)

@pytest.fixture
def salt_teste():
    return "salt_unitario_safedriver_2024"

@pytest.fixture
def df_prata_valido():
    n = 100
    return pl.DataFrame({
        "H3_INDEX":              [f"8a2a100d{i:04x}fff" for i in range(n)],
        "LATITUDE_F":            np.random.uniform(SP_LAT_MIN, SP_LAT_MAX, n),
        "LONGITUDE_F":           np.random.uniform(SP_LON_MIN, SP_LON_MAX, n),
        "RUBRICA":               np.random.choice(["ROUBO", "FURTO", "HOMICIDIO DOLOSO"], n),
        "PERIODO_DIA":           np.random.choice(["MANHA", "TARDE", "NOITE", "MADRUGADA"], n),
        "IS_NOITE_MADRUGADA":    np.random.randint(0, 2, n),
        "IS_PATRIMONIO":         np.random.randint(0, 2, n),
        "IS_VIOLENCIA_PESSOA":   np.random.randint(0, 2, n),
        "NOME_MUNICIPIO":        np.random.choice(["SAO PAULO", "CAMPINAS"], n),
        "DATA_OCORRENCIA_BO":    pd.to_datetime(pd.date_range("2023-01-01", periods=n, freq="D")),
        "ANO":                   np.random.randint(2022, 2025, n),
        "ESCORE_MOTORISTA":      np.random.uniform(1.0, 15.0, n),
        "ESCORE_MOTOCICLISTA":   np.random.uniform(1.0, 15.0, n),
        "ESCORE_PEDESTRE":       np.random.uniform(1.0, 15.0, n),
        "ESCORE_CICLISTA":       np.random.uniform(1.0, 15.0, n),
        "LOGRADOURO_HASH":       [hashlib.sha256(f"RUA {i}-salt".encode()).hexdigest() for i in range(n)],
        "NUMERO_LOGRADOURO_HASH":[hashlib.sha256(f"{i}-salt".encode()).hexdigest() for i in range(n)],
        "BAIRRO_HASH":           [hashlib.sha256(f"BAIRRO {i}-salt".encode()).hexdigest() for i in range(n)],
    })

@pytest.fixture
def df_ouro_valido():
    n = 300
    return pd.DataFrame({
        "H3_INDEX":                [f"8a2a100d{i:04x}fff" for i in range(n)],
        "QTD_CRIMES":              np.random.randint(1, 200, n),
        "ESCORE_TOTAL":            np.random.uniform(10.0, 500.0, n),
        "ESCORE_MEDIO":            np.random.uniform(1.0, 10.0, n),
        "ESCORE_GRAVIDADE_MAX":    np.random.uniform(5.0, 15.0, n),
        "ESCORE_MOTORISTA":        np.random.uniform(1.0, 15.0, n),
        "ESCORE_MOTOCICLISTA":     np.random.uniform(1.0, 15.0, n),
        "ESCORE_PEDESTRE":         np.random.uniform(1.0, 15.0, n),
        "ESCORE_CICLISTA":         np.random.uniform(1.0, 15.0, n),
        "LATITUDE_MEDIA":          np.random.uniform(SP_LAT_MIN, SP_LAT_MAX, n),
        "LONGITUDE_MEDIA":         np.random.uniform(SP_LON_MIN, SP_LON_MAX, n),
        "PROP_NOITE_MADRUGADA":    np.random.uniform(0.0, 1.0, n),
        "PROP_PATRIMONIO":         np.random.uniform(0.0, 1.0, n),
        "PROP_VIOLENCIA_PESSOA":   np.random.uniform(0.0, 1.0, n),
        "ESCORE_LAG2":             np.random.uniform(5.0, 400.0, n),
        "QTD_LAG2":                np.random.randint(1, 150, n),
        "ESCORE_VIZ_1":            np.random.uniform(5.0, 400.0, n),
        "QTD_CRIMES_VIZ":          np.random.randint(1, 150, n),
        "ESCORE_PREDITO":          np.random.uniform(10.0, 500.0, n),
        "MUNICIPIO_DOMINANTE":     np.random.choice(["SAO PAULO", "CAMPINAS"], n),
        "IS_FERIADO":              np.random.randint(0, 2, n),
        "ANO":                     np.random.randint(2022, 2025, n),
        "DATA_REFERENCIA":         pd.to_datetime(pd.date_range("2023-01-01", periods=n, freq="D")),
        "DIA_SEMANA":              np.random.randint(0, 7, n),
        "MES":                     np.random.randint(1, 13, n),
        "DIA_MES":                 np.random.randint(1, 32, n),
    })

class TestIdentidade:
    def test_nome_sistema(self):
        assert NOME_SISTEMA == "SafeDriver_Motor"

    def test_perfis_completos(self):
        assert set(PERFIS) == {"MOTORISTA", "MOTOCICLISTA", "PEDESTRE", "CICLISTA"}

    def test_anos_disponiveis(self):
        assert 2022 in ANOS_DISPONIVEIS
        assert min(ANOS_DISPONIVEIS) == 2022

    def test_ssp_url_template(self):
        url = SSP_URL_TEMPLATE.format(ano=2024)
        assert "2024" in url
        assert url.startswith("https://www.ssp.sp.gov.br")
        assert url.endswith(".xlsx")

class TestLGPD:
    def test_anonimizar_retorna_sha256(self, salt_teste):
        resultado = anonimizar_campo("Rua das Flores, 123", salt_teste)
        assert len(resultado) == 64
        assert all(c in "0123456789abcdef" for c in resultado)

    def test_anonimizar_determinista(self, salt_teste):
        r1 = anonimizar_campo("Av Paulista", salt_teste)
        r2 = anonimizar_campo("Av Paulista", salt_teste)
        assert r1 == r2

    def test_anonimizar_salts_diferentes(self, salt_teste):
        r1 = anonimizar_campo("Rua A", salt_teste)
        r2 = anonimizar_campo("Rua A", "outro_salt")
        assert r1 != r2

    def test_anonimizar_valores_diferentes(self, salt_teste):
        r1 = anonimizar_campo("Rua A", salt_teste)
        r2 = anonimizar_campo("Rua B", salt_teste)
        assert r1 != r2

    def test_anonimizar_campo_vazio(self, salt_teste):
        resultado = anonimizar_campo("", salt_teste)
        assert len(resultado) == 64

    def test_anonimizar_nao_expoe_original(self, salt_teste):
        original  = "Rua Coronel Melo de Oliveira"
        resultado = anonimizar_campo(original, salt_teste)
        assert original.upper() not in resultado
        assert "RUA" not in resultado

class TestPeriodoDia:
    @pytest.mark.parametrize("hora,esperado", [
        ("00:00", "MADRUGADA"), ("03:30", "MADRUGADA"), ("05:59", "MADRUGADA"),
        ("06:00", "MANHA"),     ("11:59", "MANHA"),
        ("12:00", "TARDE"),     ("17:59", "TARDE"),
        ("18:00", "NOITE"),     ("23:59", "NOITE"),
    ])
    def test_classificar_periodo(self, hora, esperado):
        assert classificar_periodo(hora) == esperado

    def test_periodo_hora_invalida(self):
        assert classificar_periodo("XX:YY") == "MANHA"

    @pytest.mark.parametrize("hora,fator", [
        ("02:00", 1.4), ("08:00", 1.0), ("15:00", 1.0), ("20:00", 1.3),
    ])
    def test_fator_periodo(self, hora, fator):
        assert fator_periodo(hora) == fator

    def test_manha_tarde_nao_sao_agravantes(self):
        assert fator_periodo("09:00") == 1.0
        assert fator_periodo("14:00") == 1.0

    def test_noite_madrugada_sao_agravantes(self):
        assert fator_periodo("20:00") > 1.0
        assert fator_periodo("03:00") > 1.0

    def test_madrugada_mais_agravante_que_noite(self):
        assert fator_periodo("03:00") > fator_periodo("20:00")

class TestEscore:
    def test_escore_positivo(self):
        assert calcular_escore("ROUBO", "20:00", "PEDESTRE") > 0

    def test_escore_noite_maior_que_manha(self):
        assert calcular_escore("ROUBO", "20:00", "PEDESTRE") > \
               calcular_escore("ROUBO", "09:00", "PEDESTRE")

    def test_escore_madrugada_maior_que_noite(self):
        assert calcular_escore("FURTO", "03:00", "MOTORISTA") > \
               calcular_escore("FURTO", "20:00", "MOTORISTA")

    def test_escore_manha_igual_tarde(self):
        assert calcular_escore("ROUBO", "09:00", "PEDESTRE") == \
               calcular_escore("ROUBO", "14:00", "PEDESTRE")

    def test_escore_homicidio_maior_que_furto(self):
        assert calcular_escore("HOMICIDIO DOLOSO", "12:00", "PEDESTRE") > \
               calcular_escore("FURTO", "12:00", "PEDESTRE")

    def test_escore_latrocinio_maximo(self):
        e = calcular_escore("LATROCINIO", "12:00", "MOTORISTA")
        assert e >= 10.0

    def test_escore_crime_desconhecido(self):
        assert calcular_escore("CRIME_INEXISTENTE", "12:00", "MOTORISTA") > 0

    def test_escore_perfil_motociclista(self):
        assert calcular_escore("ROUBO DE MOTOCICLETA", "20:00", "MOTOCICLISTA") > \
               calcular_escore("ROUBO DE MOTOCICLETA", "20:00", "PEDESTRE")

    def test_escore_retorna_float(self):
        assert isinstance(calcular_escore("FURTO", "10:00", "CICLISTA"), float)

    def test_escore_arredondado_4_casas(self):
        e = calcular_escore("ROUBO", "20:00", "MOTORISTA")
        assert round(e, 4) == e

    def test_escore_todos_perfis_retorna_dict(self):
        resultado = calcular_escores_todos_perfis("ROUBO DE VEICULO", "20:00")
        assert set(resultado.keys()) == {f"ESCORE_{p.upper()}" for p in PERFIS}
        assert all(v > 0 for v in resultado.values())

    def test_escore_motorista_maior_roubo_veiculo(self):
        r = calcular_escores_todos_perfis("ROUBO DE VEICULO", "20:00")
        assert r["ESCORE_MOTORISTA"] >= r["ESCORE_PEDESTRE"]

    def test_escore_motociclista_maior_roubo_moto(self):
        r = calcular_escores_todos_perfis("ROUBO DE MOTOCICLETA", "12:00")
        assert r["ESCORE_MOTOCICLISTA"] >= r["ESCORE_MOTORISTA"]

    def test_escore_ciclista_maior_atropelamento(self):
        r = calcular_escores_todos_perfis("ATROPELAMENTO", "12:00")
        assert r["ESCORE_CICLISTA"] >= r["ESCORE_PEDESTRE"]

class TestContratosPrata:
    COLUNAS_OBRIGATORIAS = [
        "H3_INDEX", "LATITUDE_F", "LONGITUDE_F", "RUBRICA", "PERIODO_DIA",
        "IS_NOITE_MADRUGADA", "IS_PATRIMONIO", "IS_VIOLENCIA_PESSOA",
        "NOME_MUNICIPIO", "DATA_OCORRENCIA_BO", "ANO",
        "ESCORE_MOTORISTA", "ESCORE_MOTOCICLISTA", "ESCORE_PEDESTRE", "ESCORE_CICLISTA",
        "LOGRADOURO_HASH", "NUMERO_LOGRADOURO_HASH", "BAIRRO_HASH"
    ]

    def test_colunas_presentes(self, df_prata_valido):
        for col in self.COLUNAS_OBRIGATORIAS:
            assert col in df_prata_valido.columns, f"Coluna ausente na prata: {col}"

    def test_escores_perfil_positivos(self, df_prata_valido):
        for col in ["ESCORE_MOTORISTA", "ESCORE_MOTOCICLISTA",
                    "ESCORE_PEDESTRE", "ESCORE_CICLISTA"]:
            assert (df_prata_valido[col] > 0).all(), f"{col} tem valores negativos/zero"

    def test_h3_nao_nulo(self, df_prata_valido):
        assert df_prata_valido["H3_INDEX"].is_not_null().all()

    def test_periodo_valores_validos(self, df_prata_valido):
        validos = {"MANHA", "TARDE", "NOITE", "MADRUGADA"}
        assert set(df_prata_valido["PERIODO_DIA"].unique()).issubset(validos)

    def test_coordenadas_sp(self, df_prata_valido):
        assert df_prata_valido["LATITUDE_F"].is_between(SP_LAT_MIN, SP_LAT_MAX).all()
        assert df_prata_valido["LONGITUDE_F"].is_between(SP_LON_MIN, SP_LON_MAX).all()

    def test_is_noite_madrugada_binario(self, df_prata_valido):
        assert set(df_prata_valido["IS_NOITE_MADRUGADA"].unique()).issubset({0, 1})

class TestContratosOuro:
    COLUNAS_OBRIGATORIAS = [
        "H3_INDEX", "QTD_CRIMES", "ESCORE_TOTAL", "ESCORE_MEDIO",
        "ESCORE_GRAVIDADE_MAX", "ESCORE_MOTORISTA", "ESCORE_MOTOCICLISTA",
        "ESCORE_PEDESTRE", "ESCORE_CICLISTA", "LATITUDE_MEDIA", "LONGITUDE_MEDIA",
        "PROP_NOITE_MADRUGADA", "PROP_PATRIMONIO", "PROP_VIOLENCIA_PESSOA",
        "ESCORE_LAG2", "QTD_LAG2", "ESCORE_VIZ_1", "QTD_CRIMES_VIZ",
        "ESCORE_PREDITO", "MUNICIPIO_DOMINANTE", "IS_FERIADO",
    ]

    def test_colunas_presentes(self, df_ouro_valido):
        for col in self.COLUNAS_OBRIGATORIAS:
            assert col in df_ouro_valido.columns, f"Coluna ausente no ouro: {col}"

    def test_escores_perfil_positivos(self, df_ouro_valido):
        for col in ["ESCORE_MOTORISTA", "ESCORE_MOTOCICLISTA",
                    "ESCORE_PEDESTRE", "ESCORE_CICLISTA", "ESCORE_TOTAL", "ESCORE_PREDITO"]:
            assert (df_ouro_valido[col] > 0).all(), f"{col} tem valores negativos/zero"

    def test_h3_nao_nulo(self, df_ouro_valido):
        assert df_ouro_valido["H3_INDEX"].notna().all()

    def test_latitude_longitude_media_validas(self, df_ouro_valido):
        assert df_ouro_valido["LATITUDE_MEDIA"].between(SP_LAT_MIN, SP_LAT_MAX).all()
        assert df_ouro_valido["LONGITUDE_MEDIA"].between(SP_LON_MIN, SP_LON_MAX).all()

    def test_proporcoes_entre_0_e_1(self, df_ouro_valido):
        for col in ["PROP_NOITE_MADRUGADA", "PROP_PATRIMONIO", "PROP_VIOLENCIA_PESSOA", "IS_FERIADO"]:
            assert df_ouro_valido[col].between(0, 1).all(), f"{col} fora do range [0, 1]"

    def test_schema_polars_compativel(self, df_ouro_valido):
        try:
            pl.from_pandas(df_ouro_valido)
        except Exception as e:
            pytest.fail(f"Erro de compatibilidade Polars: {e}")

class TestNormalizacaoSinonimos:
    def test_normalizar_texto_maiusculas_sem_acentos(self):
        assert normalizar_texto("São Paulo") == "SAO PAULO"
        assert normalizar_texto("Rua Cândido Portinari") == "RUA CANDIDO PORTINARI"

    def test_normalizar_texto_numeros_e_caracteres_especiais(self):
        assert normalizar_texto("Rua 123-B") == "RUA 123-B"

    def test_normalizar_texto_vazio(self):
        assert normalizar_texto("") == ""

    def test_renomear_sinonimos_funciona(self):
        df = pl.DataFrame({"DATA_OCORRENCIA": ["01/01/2023"]})
        df_r = renomear_sinonimos(df)
        assert "DATA_OCORRENCIA_BO" in df_r.columns
        assert "DATA_OCORRENCIA" not in df_r.columns

    def test_renomear_sinonimos_multiplos(self):
        df = pl.DataFrame({"MUN": ["SP"], "DT_OCORRENCIA": ["01/01/2023"]})
        df_r = renomear_sinonimos(df)
        assert "NOME_MUNICIPIO" in df_r.columns
        assert "DATA_OCORRENCIA_BO" in df_r.columns

    def test_renomear_sem_sinonimos_conhecidos(self):
        df = pl.DataFrame({"COLUNA_ESTRANHA": [1, 2, 3]})
        df_r = renomear_sinonimos(df)
        assert "COLUNA_ESTRANHA" in df_r.columns

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
            pytest.skip(f"Secrets não disponíveis localmente (normal): {faltando}")

    def test_lgpd_salt_nao_vazio_se_presente(self):
        salt = os.environ.get("LGPD_SALT", "")
        if salt:
            assert len(salt.strip()) >= 16, "LGPD_SALT muito curto — mínimo 16 caracteres"

    def test_bq_project_nao_hardcoded(self):
        caminho = os.path.join(
            os.path.dirname(__file__), "..", "autobot", "motor_analise_preditiva.py"
        )
        if os.path.exists(caminho):
            with open(caminho, "r", encoding="utf-8") as f:
                conteudo = f.read()
            assert "BQ_PROJECT_FIXO" not in conteudo, \
                "BQ_PROJECT_FIXO hardcoded encontrado — usar só o secret"
            assert "safe-driver-fc3a9" not in conteudo, \
                "Project ID hardcoded no código — usar só o secret"

class TestIntegracaoR2SSP:
    @patch("requests.get")
    @patch("boto3.client")
    def test_baixar_ssp_sucesso(self, mock_boto_client, mock_requests_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        df_dummy = pd.DataFrame({"col1": [1, 2], "col2": ["A", "B"]})
        excel_buffer = BytesIO()
        df_dummy.to_excel(excel_buffer, index=False, engine="openpyxl")
        excel_buffer.seek(0)
        mock_response.content = excel_buffer.getvalue()
        mock_requests_get.return_value = mock_response

        df_baixado = baixar_ssp(2022)
        assert df_baixado is not None
        assert not df_baixado.empty
        assert len(df_baixado) == 2
        mock_requests_get.assert_called_once()

    @patch("requests.get")
    @patch("boto3.client")
    def test_baixar_ssp_404(self, mock_boto_client, mock_requests_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_requests_get.return_value = mock_response

        df_baixado = baixar_ssp(2027)
        assert df_baixado is None
        mock_requests_get.assert_called_once()

    @patch("requests.get")
    @patch("boto3.client")
    def test_baixar_ssp_falha_apos_retries(self, mock_boto_client, mock_requests_get):
        mock_requests_get.side_effect = requests.exceptions.Timeout("Mock Timeout")

        df_baixado = baixar_ssp(2022)
        assert df_baixado is None
        assert mock_requests_get.call_count == SSP_MAX_TENTATIVAS

    @patch("boto3.client")
    def test_tracking_ssp_init_migracao_int_para_dict(self, mock_boto_client):
        mock_s3_obj = MagicMock()
        mock_s3_obj.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps({"2022": 12345}).encode("utf-8"))
        }
        mock_boto_client.return_value = mock_s3_obj

        tracking = TrackingSSP(mock_boto_client, "test-bucket", "test-tracking.json")
        assert tracking.dados["2022"] == {"tamanho_bytes": 12345, "hash_sha256": "legacy_hash"}
        assert not tracking.precisa_processar(2022, "legacy_hash", 12345)
        assert tracking.precisa_processar(2022, "new_hash", 12345)
        assert tracking.precisa_processar(2022, "legacy_hash", 54321)

    @patch("boto3.client")
    def test_tracking_ssp_atualizar_e_salvar(self, mock_boto_client):
        mock_s3_obj = MagicMock()
        mock_s3_obj.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        mock_boto_client.return_value = mock_s3_obj

        tracking = TrackingSSP(mock_boto_client, "test-bucket", "test-tracking.json")
        tracking.atualizar_tracking({"2023": {"tamanho_bytes": 500, "hash_sha256": "abc"}})
        tracking.salvar_tracking()

        mock_s3_obj.put_object.assert_called_once()
        args, kwargs = mock_s3_obj.put_object.call_args
        assert kwargs["Key"] == "test-tracking.json"
        salvo = json.loads(kwargs["Body"].decode("utf-8"))
        assert salvo["2023"]["hash_sha256"] == "abc"

    @patch("autobot.motor_analise_preditiva.baixar_ssp")
    @patch("boto3.client")
    def test_sincronizar_raw_fallback_ssp(self, mock_boto_client, mock_baixar_ssp):
        mock_s3_instance = MagicMock()
        mock_s3_instance.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        mock_boto_client.return_value = mock_s3_instance

        df_ssp_mock = pd.DataFrame({"DATA_OCORRENCIA_BO": ["01/01/2022"], "LATITUDE": [-23.5], "LONGITUDE": [-46.6], "RUBRICA": ["FURTO"]})
        mock_baixar_ssp.return_value = df_ssp_mock

        mock_tracking = MagicMock(spec=TrackingSSP)
        mock_tracking.dados = {}
        mock_tracking.precisa_processar.return_value = True

        with patch.dict(os.environ, {"R2_BUCKET_NAME": "test-bucket", "LGPD_SALT": "a"*16, "R2_ENDPOINT_URL": "http://mock", "R2_ACCESS_KEY_ID": "mock", "R2_SECRET_ACCESS_KEY": "mock"}):
            app = SafeDriver()
            app.s3 = mock_s3_instance
            app.tracking = mock_tracking
            app.ANOS_DISPONIVEIS = [2022]

            app.sincronizar_raw()

            mock_baixar_ssp.assert_called_once_with(2022)
            mock_s3_instance.put_object.assert_called_once()
            assert not app.df_raw.is_empty()
            assert len(app.df_raw) == 1
            mock_tracking.atualizar_tracking.assert_called_once()
            mock_tracking.salvar_tracking.assert_called_once()

    @patch("autobot.motor_analise_preditiva.baixar_ssp")
    @patch("boto3.client")
    def test_sincronizar_raw_r2_existente_e_nao_alterado(self, mock_boto_client, mock_baixar_ssp):
        mock_s3_instance = MagicMock()
        parquet_data = pl.DataFrame({"DATA_OCORRENCIA_BO": ["01/01/2022"], "LATITUDE": [-23.5], "LONGITUDE": [-46.6], "RUBRICA": ["FURTO"]}).write_parquet(BytesIO()).getvalue()
        mock_s3_instance.get_object.return_value = {
            "Body": MagicMock(read=lambda: parquet_data),
            "ContentLength": len(parquet_data)
        }
        mock_boto_client.return_value = mock_s3_instance

        mock_tracking = MagicMock(spec=TrackingSSP)
        mock_tracking.dados = {"2022": {"tamanho_bytes": len(parquet_data), "hash_sha256": hashlib.sha256(parquet_data).hexdigest()}}
        mock_tracking.precisa_processar.return_value = False

        with patch.dict(os.environ, {"R2_BUCKET_NAME": "test-bucket", "LGPD_SALT": "a"*16, "R2_ENDPOINT_URL": "http://mock", "R2_ACCESS_KEY_ID": "mock", "R2_SECRET_ACCESS_KEY": "mock"}):
            app = SafeDriver()
            app.s3 = mock_s3_instance
            app.tracking = mock_tracking
            app.ANOS_DISPONIVEIS = [2022]

            app.sincronizar_raw()

            mock_baixar_ssp.assert_not_called()
            mock_s3_instance.put_object.assert_not_called()
            assert app.df_raw.is_empty()
            mock_tracking.atualizar_tracking.assert_not_called()
            mock_tracking.salvar_tracking.assert_not_called()

class TestPesos:
    def test_todos_pesos_positivos(self):
        for crime, peso in PESO_PENAL_BASE.items():
            assert peso > 0, f"Peso inválido para: {crime}"

    def test_crimes_graves_peso_maximo(self):
        assert PESO_PENAL_BASE["HOMICIDIO DOLOSO"] == 10.0
        assert PESO_PENAL_BASE["LATROCINIO"]       == 10.0

    def test_furto_menor_que_roubo(self):
        assert PESO_PENAL_BASE["FURTO"] < PESO_PENAL_BASE["ROUBO"]

    def test_todos_multiplicadores_positivos(self):
        for perfil, crimes in MULTIPLICADOR_PERFIL.items():
            for crime, mult in crimes.items():
                assert mult > 0, f"Multiplicador inválido: {perfil}/{crime}"

    def test_fator_madrugada_maior_que_noite(self):
        assert FATOR_PERIODO["MADRUGADA"] > FATOR_PERIODO["NOITE"]

    def test_manha_tarde_neutros(self):
        assert FATOR_PERIODO["MANHA"] == 1.0
        assert FATOR_PERIODO["TARDE"] == 1.0

    def test_perfis_com_multiplicadores(self):
        assert set(MULTIPLICADOR_PERFIL.keys()) == set(PERFIS)
