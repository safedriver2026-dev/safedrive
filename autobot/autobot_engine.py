import io
import os
import json
import math
import hashlib
import logging
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import pygeohash as gh
import requests
import firebase_admin
import xgboost as xgb

from firebase_admin import credentials, firestore
from prophet import Prophet
from requests.adapters import HTTPAdapter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from urllib3.util.retry import Retry

from autobot.config import (
    CATALOGO_CRIMES,
    TIPOS_LOCAL_PERMITIDOS,
    SUBTIPOS_LOCAL_PERMITIDOS,
    LIMITES_SP,
    ESQUEMA_TRUSTED,
    COLUNAS_REFINED,
    PALAVRAS_CHAVE_PERFIL,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class MotorSafeDriver:
    def __init__(self):
        self.data_referencia = pd.Timestamp(datetime.now().date())
        self.janela_fim = self.data_referencia
        self.janela_inicio = self.janela_fim - pd.Timedelta(days=730)
        self.periodo_historico = range(self.janela_inicio.year, self.janela_fim.year + 1)

        self.precisao_geohash = 7
        self.horizonte_predicao_dias = 7
        self.dias_holdout_teste = 60
        self.versao_modelo = "safedriver_v3_2_0"

        self.banco_nuvem = self._estabelecer_conexao_nuvem()
        self.sessao_web = self._criar_sessao_resiliente()

        self.auditoria = {
            "volume_raw": 0,
            "volume_trusted": 0,
            "volume_refined_eventos": 0,
            "volume_refined_modelagem": 0,
            "falhas_integridade": 0,
            "malha_motorista": 0,
            "malha_motociclista": 0,
            "malha_pedestre": 0,
            "malha_ciclista": 0,
            "documentos_sincronizados": 0,
            "documentos_atualizados": 0,
            "documentos_removidos": 0,
            "novos_dados": False,
            "linhas_treino": 0,
            "linhas_teste": 0,
            "mae_teste": None,
            "rmse_teste": None,
        }

        for pasta in ["raw", "trusted", "refined", "metadata"]:
            os.makedirs(f"datalake/{pasta}", exist_ok=True)

    # =========================================================
    # INFRA
    # =========================================================

    def _estabelecer_conexao_nuvem(self):
        chave_secreta = os.environ.get("FIREBASE_JSON")

        if not chave_secreta:
            raise EnvironmentError("A variavel FIREBASE_JSON nao foi definida.")

        if not firebase_admin._apps:
            credenciais = credentials.Certificate(json.loads(chave_secreta))
            firebase_admin.initialize_app(credenciais)

        return firestore.client()

    def _criar_sessao_resiliente(self):
        sessao = requests.Session()
        retentativas = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=2,
            status_forcelist=[403, 429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"],
        )
        adaptador = HTTPAdapter(max_retries=retentativas)
        sessao.mount("http://", adaptador)
        sessao.mount("https://", adaptador)
        sessao.headers.update({"User-Agent": "Mozilla/5.0 SafeDriver/3.2"})
        return sessao

    def _notificar_sucesso(self):
        endereco_webhook = os.environ.get("DISCORD_SUCESSO")
        if not endereco_webhook:
            return

        status_execucao = (
            "Novos arquivos RAW foram baixados da SSP."
            if self.auditoria["novos_dados"]
            else "Pipeline executado reutilizando o RAW local."
        )

        payload = {
            "embeds": [{
                "title": "Relatorio Semanal SafeDriver",
                "description": status_execucao,
                "color": 3066993,
                "fields": [
                    {
                        "name": "Janela de processamento",
                        "value": f"{self.janela_inicio.date()} ate {self.janela_fim.date()}",
                        "inline": False
                    },
                    {
                        "name": "Camadas",
                        "value": (
                            f"RAW: {self.auditoria['volume_raw']:,}\n"
                            f"TRUSTED: {self.auditoria['volume_trusted']:,}\n"
                            f"REFINED_EVENTOS: {self.auditoria['volume_refined_eventos']:,}\n"
                            f"REFINED_MODELAGEM: {self.auditoria['volume_refined_modelagem']:,}"
                        ),
                        "inline": False
                    },
                    {
                        "name": "Validacao temporal",
                        "value": (
                            f"Treino: {self.auditoria['linhas_treino']:,}\n"
                            f"Teste: {self.auditoria['linhas_teste']:,}\n"
                            f"MAE: {self.auditoria['mae_teste']}\n"
                            f"RMSE: {self.auditoria['rmse_teste']}"
                        ),
                        "inline": False
                    },
                    {
                        "name": "Malha operacional",
                        "value": (
                            f"Motorista: {self.auditoria['malha_motorista']:,}\n"
                            f"Motociclista: {self.auditoria['malha_motociclista']:,}\n"
                            f"Pedestre: {self.auditoria['malha_pedestre']:,}\n"
                            f"Ciclista: {self.auditoria['malha_ciclista']:,}"
                        ),
                        "inline": False
                    },
                    {
                        "name": "Firestore",
                        "value": (
                            f"Avaliados: {self.auditoria['documentos_sincronizados']:,}\n"
                            f"Atualizados: {self.auditoria['documentos_atualizados']:,}\n"
                            f"Removidos: {self.auditoria['documentos_removidos']:,}"
                        ),
                        "inline": False
                    }
                ]
            }]
        }

        try:
            self.sessao_web.post(endereco_webhook, json=payload, timeout=30)
        except Exception as erro:
            logging.warning("Falha ao notificar sucesso no Discord: %s", erro)

    def _notificar_erro(self, diagnostico_falha):
        endereco_webhook = os.environ.get("DISCORD_ERRO")
        if not endereco_webhook:
            return

        payload = {
            "embeds": [{
                "title": "Interrupcao Operacional SafeDriver",
                "color": 15158332,
                "fields": [{
                    "name": "Diagnostico",
                    "value": str(diagnostico_falha)[:1000],
                    "inline": False
                }]
            }]
        }

        try:
            self.sessao_web.post(endereco_webhook, json=payload, timeout=30)
        except Exception as erro:
            logging.warning("Falha ao notificar erro no Discord: %s", erro)

    # =========================================================
    # NORMALIZACAO
    # =========================================================

    def _higienizar_texto(self, texto_bruto):
        if pd.isna(texto_bruto):
            return ""

        if not isinstance(texto_bruto, str):
            texto_bruto = str(texto_bruto)

        texto_normalizado = unicodedata.normalize("NFKD", texto_bruto)
        return "".join(
            c for c in texto_normalizado if not unicodedata.combining(c)
        ).upper().strip()

    def _normalizar_hora(self, valor):
        if pd.isna(valor):
            return np.nan

        texto = str(valor).strip()
        if not texto:
            return np.nan

        formatos = [
            "%H:%M:%S",
            "%H:%M",
            "%H",
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
        ]

        for formato in formatos:
            try:
                return datetime.strptime(texto, formato).hour
            except Exception:
                continue

        try:
            dt = pd.to_datetime(texto, errors="coerce")
            return dt.hour if not pd.isna(dt) else np.nan
        except Exception:
            return np.nan

    def _classificar_turno(self, hora):
        if pd.isna(hora):
            return np.nan

        hora = int(hora)

        if 0 <= hora < 6:
            return "Madrugada"
        if 6 <= hora < 12:
            return "Manha"
        if 12 <= hora < 18:
            return "Tarde"
        return "Noite"

    def _inferir_perfil_contextual(self, linha):
        perfis_identificados = set()

        contexto_textual = (
            f"{linha.get('NATUREZA_APURADA', '')} "
            f"{linha.get('DESCR_CONDUTA', '')} "
            f"{linha.get('DESCR_SUBTIPOLOCAL', '')} "
            f"{linha.get('RUBRICA', '')}"
        ).upper()

        for perfil, palavras in PALAVRAS_CHAVE_PERFIL.items():
            if any(palavra in contexto_textual for palavra in palavras):
                perfis_identificados.add(perfil)

        if not perfis_identificados:
            perfis_base = CATALOGO_CRIMES.get(
                linha.get("NATUREZA_APURADA"),
                {}
            ).get("perfis", [])
            perfis_identificados.update(perfis_base)

        return list(perfis_identificados) if perfis_identificados else ["Indefinido"]

    def _aplicar_janela_historica(self, dataframe):
        df = dataframe.copy()

        df["DATA_OCORRENCIA_BO"] = pd.to_datetime(
            df["DATA_OCORRENCIA_BO"],
            errors="coerce"
        ).dt.normalize()

        mascara = (
            df["DATA_OCORRENCIA_BO"].notna()
            & (df["DATA_OCORRENCIA_BO"] >= self.janela_inicio)
            & (df["DATA_OCORRENCIA_BO"] <= self.janela_fim)
        )

        return df.loc[mascara].copy()

    # =========================================================
    # RAW
    # =========================================================

    def _baixar_raw_ssp(self, ano_referencia):
        endereco_arquivo = (
            f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/"
            f"spDados/SPDadosCriminais_{ano_referencia}.xlsx"
        )
        caminho_raw = f"datalake/raw/ssp_{ano_referencia}.parquet"

        resposta = self.sessao_web.get(endereco_arquivo, timeout=120)
        resposta.raise_for_status()

        leitura_previa = pd.read_excel(
            io.BytesIO(resposta.content),
            nrows=50,
            header=None
        )

        linha_cabecalho = next(
            (
                indice for indice, linha in leitura_previa.iterrows()
                if any(
                    termo in [self._higienizar_texto(str(c)) for c in linha.values]
                    for termo in ["NUM_BO", "LATITUDE", "NATUREZA_APURADA"]
                )
            ),
            None
        )

        if linha_cabecalho is None:
            raise ValueError(f"Cabecalho nao encontrado no arquivo SSP {ano_referencia}.")

        tabela = pd.read_excel(
            io.BytesIO(resposta.content),
            skiprows=linha_cabecalho,
            dtype=str
        )

        tabela.columns = [self._higienizar_texto(c) for c in tabela.columns]

        mapeamento = {
            "NUMERO_BO": "NUM_BO",
            "N_BO": "NUM_BO",
            "LAT": "LATITUDE",
            "LON": "LONGITUDE",
            "DATA_FATO": "DATA_OCORRENCIA_BO",
            "HORA_FATO": "HORA_OCORRENCIA_BO",
        }
        tabela.rename(columns=mapeamento, inplace=True)
        tabela.to_parquet(caminho_raw, index=False)

        self.auditoria["novos_dados"] = True
        return tabela

    def _ler_ou_baixar_raw(self, ano_referencia):
        caminho_raw = f"datalake/raw/ssp_{ano_referencia}.parquet"

        if os.path.exists(caminho_raw):
            logging.info("Usando RAW local: %s", caminho_raw)
            return pd.read_parquet(caminho_raw)

        logging.info("RAW local nao encontrado para %s. Baixando da SSP.", ano_referencia)
        return self._baixar_raw_ssp(ano_referencia)

    # =========================================================
    # TRUSTED
    # =========================================================

    def _processar_trusted(self, dataframe_bruto, ano_referencia):
        df = dataframe_bruto.copy()

        for coluna in ESQUEMA_TRUSTED.keys():
            if coluna not in df.columns and coluna != "ANO_BASE":
                df[coluna] = np.nan

        df["ANO_BASE"] = ano_referencia
        volume_inicial = len(df)

        for coluna, tipo_dado in ESQUEMA_TRUSTED.items():
            if coluna not in df.columns:
                continue

            if tipo_dado == "string":
                df[coluna] = df[coluna].apply(self._higienizar_texto)
            elif tipo_dado == "float":
                df[coluna] = pd.to_numeric(
                    df[coluna].astype(str).str.replace(",", ".", regex=False),
                    errors="coerce"
                )
            elif tipo_dado == "datetime":
                df[coluna] = pd.to_datetime(df[coluna], errors="coerce")
            elif tipo_dado == "int":
                df[coluna] = pd.to_numeric(
                    df[coluna],
                    errors="coerce"
                ).fillna(0).astype(int)

        df = self._aplicar_janela_historica(df)

        if "DESCR_TIPOLOCAL" not in df.columns:
            df["DESCR_TIPOLOCAL"] = np.nan

        mascara_geografica = (
            df["LATITUDE"].notna()
            & df["LONGITUDE"].notna()
            & (df["LATITUDE"] != 0)
            & (df["LONGITUDE"] != 0)
            & df["LATITUDE"].between(LIMITES_SP["lat"][0], LIMITES_SP["lat"][1])
            & df["LONGITUDE"].between(LIMITES_SP["lon"][0], LIMITES_SP["lon"][1])
        )

        trusted = df.loc[mascara_geografica].copy()

        colunas_trusted = list(ESQUEMA_TRUSTED.keys())
        if "DESCR_TIPOLOCAL" not in colunas_trusted:
            trusted["DESCR_TIPOLOCAL"] = trusted.get("DESCR_TIPOLOCAL", np.nan)
            colunas_trusted = colunas_trusted + ["DESCR_TIPOLOCAL"]

        trusted = trusted[colunas_trusted].copy()

        self.auditoria["falhas_integridade"] += max(0, volume_inicial - len(trusted))
        self.auditoria["volume_trusted"] += len(trusted)

        return trusted

    # =========================================================
    # REFINED EVENTOS
    # =========================================================

    def _imputar_turno(self, dataframe):
        df = dataframe.copy()

        moda_global = "Noite"
        moda_serie = df["turno_operacional"].dropna().mode()
        if not moda_serie.empty:
            moda_global = moda_serie.iloc[0]

        mapa_moda = (
            df.dropna(subset=["turno_operacional"])
            .groupby(["codigo_geohash", "perfil"])["turno_operacional"]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else moda_global)
            .to_dict()
        )

        mascara_nula = df["turno_operacional"].isna()
        for idx in df[mascara_nula].index:
            chave = (df.at[idx, "codigo_geohash"], df.at[idx, "perfil"])
            df.at[idx, "turno_operacional"] = mapa_moda.get(chave, moda_global)

        df["turno_operacional"] = df["turno_operacional"].fillna(moda_global)
        return df

    def _processar_refined_eventos(self, trusted):
        df = trusted.copy()

        if "DESCR_TIPOLOCAL" not in df.columns:
            df["DESCR_TIPOLOCAL"] = "VIA PUBLICA"

        if "DESCR_SUBTIPOLOCAL" not in df.columns:
            df["DESCR_SUBTIPOLOCAL"] = ""

        mascara_negocio = (
            df["NATUREZA_APURADA"].isin(CATALOGO_CRIMES.keys())
            & df["DESCR_SUBTIPOLOCAL"].isin(SUBTIPOS_LOCAL_PERMITIDOS)
        )

        if "DESCR_TIPOLOCAL" in df.columns:
            mascara_negocio = mascara_negocio & df["DESCR_TIPOLOCAL"].isin(TIPOS_LOCAL_PERMITIDOS)

        refined = df.loc[mascara_negocio].copy()

        colunas_base = [c for c in COLUNAS_REFINED if c in refined.columns]
        refined = refined[colunas_base].copy()

        if "DESCR_TIPOLOCAL" not in refined.columns:
            refined["DESCR_TIPOLOCAL"] = "VIA PUBLICA"

        refined["perfis_afetados"] = refined.apply(self._inferir_perfil_contextual, axis=1)
        refined = refined.explode("perfis_afetados").dropna(subset=["perfis_afetados"]).copy()
        refined.rename(columns={"perfis_afetados": "perfil"}, inplace=True)

        refined["codigo_geohash"] = [
            gh.encode(lat, lon, precision=self.precisao_geohash)
            for lat, lon in zip(refined["LATITUDE"], refined["LONGITUDE"])
        ]
        refined["geohash_prefix_4"] = refined["codigo_geohash"].str[:4]
        refined["geohash_prefix_5"] = refined["codigo_geohash"].str[:5]

        refined["hora_normalizada"] = refined["HORA_OCORRENCIA_BO"].apply(self._normalizar_hora)
        refined["turno_operacional"] = refined["hora_normalizada"].apply(self._classificar_turno)
        refined = self._imputar_turno(refined)

        refined["data_evento"] = pd.to_datetime(
            refined["DATA_OCORRENCIA_BO"],
            errors="coerce"
        ).dt.normalize()

        refined["peso_evento"] = refined["NATUREZA_APURADA"].apply(
            lambda x: float(CATALOGO_CRIMES.get(x, {}).get("peso", 1.0))
        )

        refined = refined.dropna(
            subset=["data_evento", "codigo_geohash", "perfil", "turno_operacional"]
        ).copy()

        self.auditoria["volume_refined_eventos"] += len(refined)
        return refined

    # =========================================================
    # REFINED MODELAGEM
    # =========================================================

    def _criar_painel_diario(self, refined_eventos):
        agrupado = (
            refined_eventos.groupby(
                [
                    "codigo_geohash",
                    "geohash_prefix_4",
                    "geohash_prefix_5",
                    "perfil",
                    "turno_operacional",
                    "data_evento"
                ],
                as_index=False
            )
            .agg(
                target_dia=("peso_evento", "sum"),
                ocorrencias_dia=("peso_evento", "size"),
                latitude_media=("LATITUDE", "mean"),
                longitude_media=("LONGITUDE", "mean"),
            )
            .sort_values(
                ["codigo_geohash", "perfil", "turno_operacional", "data_evento"]
            )
        )

        chaves = ["codigo_geohash", "perfil", "turno_operacional"]

        agrupado["lag_1"] = agrupado.groupby(chaves)["target_dia"].shift(1).fillna(0.0)
        agrupado["lag_7"] = agrupado.groupby(chaves)["target_dia"].shift(7).fillna(0.0)
        agrupado["lag_14"] = agrupado.groupby(chaves)["target_dia"].shift(14).fillna(0.0)

        agrupado["media_7d"] = (
            agrupado.groupby(chaves)["target_dia"]
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
            .fillna(0.0)
        )
        agrupado["media_30d"] = (
            agrupado.groupby(chaves)["target_dia"]
            .transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())
            .fillna(0.0)
        )
        agrupado["ocorrencias_7d"] = (
            agrupado.groupby(chaves)["ocorrencias_dia"]
            .transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum())
            .fillna(0.0)
        )
        agrupado["ocorrencias_30d"] = (
            agrupado.groupby(chaves)["ocorrencias_dia"]
            .transform(lambda s: s.shift(1).rolling(30, min_periods=1).sum())
            .fillna(0.0)
        )

        agrupado["dias_desde_ultimo_evento"] = (
            agrupado.groupby(chaves)["data_evento"]
            .diff()
            .dt.days
            .fillna(999)
            .clip(lower=0, upper=999)
        )

        agrupado["dia_semana"] = agrupado["data_evento"].dt.dayofweek
        agrupado["mes"] = agrupado["data_evento"].dt.month
        agrupado["fim_semana"] = agrupado["dia_semana"].isin([5, 6]).astype(int)
        agrupado["tendencia_7_30"] = agrupado["media_7d"] - agrupado["media_30d"]

        self.auditoria["volume_refined_modelagem"] += len(agrupado)
        return agrupado

    def _criar_target_futuro(self, painel):
        df = painel.copy()
        chaves = ["codigo_geohash", "perfil", "turno_operacional"]

        def somar_janela_futura(serie):
            valores = serie.to_numpy(dtype=float)
            resultado = np.zeros(len(valores), dtype=float)

            for i in range(len(valores)):
                inicio = i + 1
                fim = i + 1 + self.horizonte_predicao_dias
                resultado[i] = valores[inicio:fim].sum()

            return pd.Series(resultado, index=serie.index)

        df["target_futuro_7d"] = (
            df.groupby(chaves)["target_dia"]
            .transform(somar_janela_futura)
        )

        return df

    def _criar_fator_prophet(self, base_treino, datas_consulta):
        serie = (
            base_treino.groupby("data_evento", as_index=False)
            .agg(y=("target_dia", "sum"))
            .rename(columns={"data_evento": "ds"})
            .sort_values("ds")
        )

        if serie.empty or len(serie) < 14:
            return pd.DataFrame({
                "data_evento": pd.to_datetime(datas_consulta).normalize(),
                "fator_prophet": 1.0
            })

        calendario = pd.DataFrame({
            "ds": pd.date_range(serie["ds"].min(), max(pd.to_datetime(datas_consulta)), freq="D")
        })

        serie = calendario.merge(serie, on="ds", how="left").fillna({"y": 0.0})

        try:
            modelo = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_mode="additive"
            )
            modelo.fit(serie)

            forecast = modelo.predict(calendario)[["ds", "yhat"]]
            forecast["yhat"] = forecast["yhat"].clip(lower=0.0)

            media_yhat = forecast["yhat"].mean()
            if pd.isna(media_yhat) or media_yhat == 0:
                forecast["fator_prophet"] = 1.0
            else:
                forecast["fator_prophet"] = (forecast["yhat"] / media_yhat).clip(0.25, 4.0)

            return forecast.rename(columns={"ds": "data_evento"})[
                ["data_evento", "fator_prophet"]
            ]
        except Exception as erro:
            logging.warning("Falha no Prophet. Usando fator neutro. Motivo: %s", erro)
            return pd.DataFrame({
                "data_evento": pd.to_datetime(datas_consulta).normalize(),
                "fator_prophet": 1.0
            })

    def _montar_base_supervisionada(self, refined_eventos):
        painel = self._criar_painel_diario(refined_eventos)
        painel = self._criar_target_futuro(painel)

        cutoff = painel["data_evento"].max() - pd.Timedelta(days=self.dias_holdout_teste)
        treino_temporal = painel[painel["data_evento"] < cutoff].copy()

        fator_prophet = self._criar_fator_prophet(
            treino_temporal,
            painel["data_evento"].unique()
        )

        painel = painel.merge(fator_prophet, on="data_evento", how="left")
        painel["fator_prophet"] = painel["fator_prophet"].fillna(1.0)

        painel = painel.dropna(subset=["target_futuro_7d"]).copy()
        return painel

    # =========================================================
    # MODELAGEM
    # =========================================================

    def _split_temporal(self, base_modelagem):
        data_max = base_modelagem["data_evento"].max()
        cutoff = data_max - pd.Timedelta(days=self.dias_holdout_teste)

        treino = base_modelagem[base_modelagem["data_evento"] < cutoff].copy()
        teste = base_modelagem[base_modelagem["data_evento"] >= cutoff].copy()

        if treino.empty or teste.empty:
            raise ValueError("Split temporal invalido. Ajuste a janela ou o holdout.")

        self.auditoria["linhas_treino"] = len(treino)
        self.auditoria["linhas_teste"] = len(teste)

        return treino, teste

    def _aplicar_encoders(self, treino, teste):
        treino = treino.copy()
        teste = teste.copy()

        encoder_turno = LabelEncoder()
        encoder_perfil = LabelEncoder()

        treino["turno_enc"] = encoder_turno.fit_transform(treino["turno_operacional"])
        treino["perfil_enc"] = encoder_perfil.fit_transform(treino["perfil"])

        mapa_turno = {valor: idx for idx, valor in enumerate(encoder_turno.classes_)}
        mapa_perfil = {valor: idx for idx, valor in enumerate(encoder_perfil.classes_)}

        teste["turno_enc"] = teste["turno_operacional"].map(mapa_turno).fillna(0).astype(int)
        teste["perfil_enc"] = teste["perfil"].map(mapa_perfil).fillna(0).astype(int)

        return treino, teste, encoder_turno, encoder_perfil

    def _treinar_modelo(self, treino):
        colunas_features = [
            "latitude_media",
            "longitude_media",
            "perfil_enc",
            "turno_enc",
            "lag_1",
            "lag_7",
            "lag_14",
            "media_7d",
            "media_30d",
            "ocorrencias_7d",
            "ocorrencias_30d",
            "dias_desde_ultimo_evento",
            "dia_semana",
            "mes",
            "fim_semana",
            "tendencia_7_30",
            "fator_prophet",
        ]

        X = treino[colunas_features]
        y = treino["target_futuro_7d"]

        modelo = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=350,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=4,
        )

        modelo.fit(X, y)
        return modelo, colunas_features

    def _avaliar_modelo(self, modelo, teste, colunas_features):
        X_teste = teste[colunas_features]
        y_teste = teste["target_futuro_7d"]

        previsoes = modelo.predict(X_teste)
        previsoes = np.clip(previsoes, 0.0, None)

        mae = mean_absolute_error(y_teste, previsoes)
        rmse = math.sqrt(mean_squared_error(y_teste, previsoes))

        self.auditoria["mae_teste"] = round(float(mae), 4)
        self.auditoria["rmse_teste"] = round(float(rmse), 4)

    def _treinar_modelo_final(self, base_modelagem, encoder_turno, encoder_perfil):
        base = base_modelagem.copy()

        mapa_turno = {valor: idx for idx, valor in enumerate(encoder_turno.classes_)}
        mapa_perfil = {valor: idx for idx, valor in enumerate(encoder_perfil.classes_)}

        base["turno_enc"] = base["turno_operacional"].map(mapa_turno).fillna(0).astype(int)
        base["perfil_enc"] = base["perfil"].map(mapa_perfil).fillna(0).astype(int)

        modelo_final, colunas_features = self._treinar_modelo(base)
        return modelo_final, colunas_features, base

    def _classificar_faixa_risco(self, score):
        if score < 3.5:
            return "baixo"
        if score < 6.5:
            return "medio"
        if score < 8.5:
            return "alto"
        return "critico"

    def _gerar_malha_semanal(self, base_modelagem, modelo_final, colunas_features):
        base = base_modelagem.copy()

        base["score_bruto"] = np.clip(
            modelo_final.predict(base[colunas_features]),
            0.0,
            None
        )

        snapshot = (
            base.sort_values("data_evento")
            .groupby(
                ["codigo_geohash", "geohash_prefix_4", "geohash_prefix_5", "perfil", "turno_operacional"],
                as_index=False
            )
            .tail(1)
            .copy()
        )

        escala = snapshot["score_bruto"].quantile(0.95)
        if pd.isna(escala) or escala <= 0:
            escala = max(snapshot["score_bruto"].max(), 1.0)

        snapshot["score"] = ((snapshot["score_bruto"] / escala) * 10.0).clip(0.5, 10.0).round(2)
        snapshot["risk_band"] = snapshot["score"].apply(self._classificar_faixa_risco)

        semana_inicio = (self.janela_fim + pd.Timedelta(days=1)).normalize()
        semana_fim = (semana_inicio + pd.Timedelta(days=self.horizonte_predicao_dias - 1)).normalize()

        snapshot["semana_referencia_inicio"] = semana_inicio
        snapshot["semana_referencia_fim"] = semana_fim

        self.auditoria["malha_motorista"] = int((snapshot["perfil"] == "Motorista").sum())
        self.auditoria["malha_motociclista"] = int((snapshot["perfil"] == "Motociclista").sum())
        self.auditoria["malha_pedestre"] = int((snapshot["perfil"] == "Pedestre").sum())
        self.auditoria["malha_ciclista"] = int((snapshot["perfil"] == "Ciclista").sum())

        return snapshot

    def _salvar_metadata_modelo(self, modelo, colunas_features, encoder_turno, encoder_perfil):
        modelo.save_model("datalake/metadata/modelo_xgb.json")

        metadata = {
            "versao_modelo": self.versao_modelo,
            "janela_inicio": str(self.janela_inicio.date()),
            "janela_fim": str(self.janela_fim.date()),
            "horizonte_predicao_dias": self.horizonte_predicao_dias,
            "dias_holdout_teste": self.dias_holdout_teste,
            "mae_teste": self.auditoria["mae_teste"],
            "rmse_teste": self.auditoria["rmse_teste"],
            "features": colunas_features,
            "classes_turno": encoder_turno.classes_.tolist(),
            "classes_perfil": encoder_perfil.classes_.tolist(),
            "gerado_em": datetime.now().isoformat(),
        }

        with open("datalake/metadata/modelo_metadata.json", "w", encoding="utf-8") as arquivo:
            json.dump(metadata, arquivo, ensure_ascii=False, indent=2)

    # =========================================================
    # FIRESTORE
    # =========================================================

    def _hash_registro(self, registro):
        payload = json.dumps(registro, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _sincronizacao_delta_firestore(self, malha_final):
        colecao = self.banco_nuvem.collection("niveis_risco")

        documentos_atuais = {}
        for doc in colecao.stream():
            dados = doc.to_dict()
            documentos_atuais[doc.id] = dados.get("hash_registro")

        lote = self.banco_nuvem.batch()
        operacoes = 0
        ids_novos = set()

        self.auditoria["documentos_sincronizados"] = len(malha_final)

        for _, linha in malha_final.iterrows():
            doc_id = f"{linha['codigo_geohash']}_{linha['perfil']}_{linha['turno_operacional']}"
            ids_novos.add(doc_id)

            payload = {
                "geohash": linha["codigo_geohash"],
                "geohash_prefix_4": linha["geohash_prefix_4"],
                "geohash_prefix_5": linha["geohash_prefix_5"],
                "perfil": linha["perfil"],
                "turno": linha["turno_operacional"],
                "periodo": linha["turno_operacional"],
                "score": round(float(linha["score"]), 2),
                "risk_band": linha["risk_band"],
                "modelo": "xgb_prophet",
                "versao_modelo": self.versao_modelo,
                "janela_inicio": str(self.janela_inicio.date()),
                "janela_fim": str(self.janela_fim.date()),
                "semana_referencia_inicio": str(pd.Timestamp(linha["semana_referencia_inicio"]).date()),
                "semana_referencia_fim": str(pd.Timestamp(linha["semana_referencia_fim"]).date()),
                "horizonte_predicao_dias": self.horizonte_predicao_dias,
                "data_base_modelo": str(pd.Timestamp(linha["data_evento"]).date()),
            }

            hash_registro = self._hash_registro(payload)
            payload["hash_registro"] = hash_registro
            payload["ultima_atualizacao"] = firestore.SERVER_TIMESTAMP

            if documentos_atuais.get(doc_id) != hash_registro:
                ref = colecao.document(doc_id)
                lote.set(ref, payload, merge=True)
                operacoes += 1
                self.auditoria["documentos_atualizados"] += 1

                if operacoes >= 450:
                    lote.commit()
                    lote = self.banco_nuvem.batch()
                    operacoes = 0

        ids_obsoletos = set(documentos_atuais.keys()) - ids_novos
        for doc_id in ids_obsoletos:
            ref = colecao.document(doc_id)
            lote.delete(ref)
            operacoes += 1
            self.auditoria["documentos_removidos"] += 1

            if operacoes >= 450:
                lote.commit()
                lote = self.banco_nuvem.batch()
                operacoes = 0

        if operacoes > 0:
            lote.commit()

    # =========================================================
    # PIPELINE PRINCIPAL
    # =========================================================

    def executar_pipeline_completo(self):
        bases_refined_eventos = []

        try:
            logging.info(
                "Iniciando pipeline SafeDriver | janela %s ate %s",
                self.janela_inicio.date(),
                self.janela_fim.date(),
            )

            for ano_alvo in self.periodo_historico:
                raw = self._ler_ou_baixar_raw(ano_alvo)
                self.auditoria["volume_raw"] += len(raw)

                trusted = self._processar_trusted(raw, ano_alvo)
                trusted.to_parquet(
                    f"datalake/trusted/ssp_trusted_{ano_alvo}.parquet",
                    index=False
                )

                if trusted.empty:
                    continue

                refined_eventos = self._processar_refined_eventos(trusted)
                refined_eventos.to_parquet(
                    f"datalake/refined/ssp_refined_eventos_{ano_alvo}.parquet",
                    index=False
                )

                if not refined_eventos.empty:
                    bases_refined_eventos.append(refined_eventos)

            if not bases_refined_eventos:
                raise ValueError("A camada refined_eventos ficou vazia apos o processamento.")

            refined_eventos_total = pd.concat(bases_refined_eventos, ignore_index=True)
            refined_eventos_total.to_parquet(
                "datalake/refined/refined_eventos_total.parquet",
                index=False
            )

            base_modelagem = self._montar_base_supervisionada(refined_eventos_total)
            base_modelagem.to_parquet(
                "datalake/refined/refined_modelagem.parquet",
                index=False
            )

            treino, teste = self._split_temporal(base_modelagem)
            treino, teste, encoder_turno, encoder_perfil = self._aplicar_encoders(treino, teste)

            modelo_validacao, colunas_features = self._treinar_modelo(treino)
            self._avaliar_modelo(modelo_validacao, teste, colunas_features)

            modelo_final, colunas_features, base_modelagem_final = self._treinar_modelo_final(
                base_modelagem,
                encoder_turno,
                encoder_perfil
            )

            self._salvar_metadata_modelo(
                modelo_final,
                colunas_features,
                encoder_turno,
                encoder_perfil
            )

            malha_final = self._gerar_malha_semanal(
                base_modelagem_final,
                modelo_final,
                colunas_features
            )

            malha_final.to_parquet(
                "datalake/refined/malha_risco_atual.parquet",
                index=False
            )

            if self.banco_nuvem:
                self._sincronizacao_delta_firestore(malha_final)

            self._notificar_sucesso()
            logging.info("Pipeline concluido com sucesso.")

        except Exception as erro_critico:
            logging.exception("Falha critica no pipeline.")
            self._notificar_erro(str(erro_critico))
            raise


if __name__ == "__main__":
    MotorSafeDriver().executar_pipeline_completo()
