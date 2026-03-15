"""
Motor Preditivo SafeDriver (Autobot Engine) - Enterprise Edition
================================================================
Pipeline autossustentável para modelagem preditiva de segurança viária.
Otimizado para roteamento de grafos (Safe & Fast Routing) via OpenMapTiles.
"""

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
from urllib3.util.retry import Retry
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from autobot.config import (
    CATALOGO_CRIMES, TIPOS_LOCAL_PERMITIDOS, SUBTIPOS_LOCAL_PERMITIDOS, LIMITES_SP,
    ESQUEMA_RAW_CANONICO, ESQUEMA_TRUSTED, COLUNAS_REFINED_EVENTOS,
    PALAVRAS_CHAVE_PERFIL, MAPA_SEMANTICO_COLUNAS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class MotorSafeDriver:
    def __init__(self, habilitar_firestore=True, forcar_recarga=False):
        self.data_referencia = pd.Timestamp(datetime.now().date())
        self.janela_fim = self.data_referencia
        
        # Janela Dinâmica de 2 anos: Garante autossustentabilidade temporal
        self.janela_inicio = self.janela_fim - pd.Timedelta(days=730)
        self.periodo_historico = range(self.janela_inicio.year, self.janela_fim.year + 1)
        self.forcar_recarga = forcar_recarga 

        self.precisao_geohash = 7
        self.horizonte_predicao_dias = 7
        self.dias_holdout_teste = 60
        self.versao_modelo = "safedriver_v7_routing"

        self.sessao_web = self._criar_sessao_resiliente()
        self.banco_nuvem = self._estabelecer_conexao_nuvem() if habilitar_firestore else None

        self.auditoria = {
            "volume_raw": 0, "volume_trusted": 0, "volume_refined_eventos": 0,
            "volume_refined_modelagem": 0, "falhas_integridade": 0,
            "malha_motorista": 0, "malha_motociclista": 0, "malha_pedestre": 0, "malha_ciclista": 0,
            "documentos_sincronizados": 0, "documentos_atualizados": 0, "documentos_removidos": 0,
            "novos_dados": False, "linhas_treino": 0, "linhas_teste": 0,
            "mae_teste": None, "rmse_teste": None,
            "anos_validos_modelagem": [], "anos_invalidos_raw": [], "anos_redownload": [],
        }

        for pasta in ["raw", "trusted", "refined", "metadata"]:
            os.makedirs(f"datalake/{pasta}", exist_ok=True)

    # =========================================================
    # INFRA & CONECTIVIDADE
    # =========================================================

    def _estabelecer_conexao_nuvem(self):
        chave_secreta = os.environ.get("FIREBASE_JSON")
        if not chave_secreta: raise EnvironmentError("FIREBASE_JSON ausente.")
        if not firebase_admin._apps:
            credenciais = credentials.Certificate(json.loads(chave_secreta))
            firebase_admin.initialize_app(credenciais)
        return firestore.client()

    def _criar_sessao_resiliente(self):
        sessao = requests.Session()
        retentativas = Retry(
            total=5, connect=5, read=5, backoff_factor=2,
            status_forcelist=[403, 429, 500, 502, 503, 504], allowed_methods=["HEAD", "GET"]
        )
        adaptador = HTTPAdapter(max_retries=retentativas)
        sessao.mount("http://", adaptador)
        sessao.mount("https://", adaptador)
        sessao.headers.update({"User-Agent": "Mozilla/5.0 SafeDriver/7.0"})
        return sessao

    def _notificar_sucesso(self):
        endereco = os.environ.get("DISCORD_SUCESSO")
        if not endereco: return
        payload = {"embeds": [{"title": "SafeDriver Pipeline OK", "color": 3066993, "description": f"Modelagem atualizada. Janela: {self.janela_inicio.date()} a {self.janela_fim.date()}"}]}
        try: self.sessao_web.post(endereco, json=payload, timeout=30)
        except: pass

    def _notificar_erro(self, erro):
        endereco = os.environ.get("DISCORD_ERRO")
        if not endereco: return
        payload = {"embeds": [{"title": "Erro Crítico SafeDriver", "color": 15158332, "description": str(erro)[:1000]}]}
        try: self.sessao_web.post(endereco, json=payload, timeout=30)
        except: pass

    # =========================================================
    # SMART CACHING & SCHEMA EVOLUTION (RAW LAYER)
    # =========================================================

    def _higienizar_texto(self, texto_bruto):
        if pd.isna(texto_bruto): return ""
        texto_norm = unicodedata.normalize("NFKD", str(texto_bruto))
        return "".join(c for c in texto_norm if not unicodedata.combining(c)).upper().strip()

    def _verificar_atualizacao_remota(self, ano_referencia, endereco_arquivo):
        """Impede downloads redundantes lendo os bytes reais no servidor da SSP via HEAD."""
        caminho_meta = f"datalake/metadata/tamanho_{ano_referencia}.json"
        try:
            head = self.sessao_web.head(endereco_arquivo, timeout=30, allow_redirects=True)
            tamanho_nuvem = int(head.headers.get("Content-Length", 0))
            
            if os.path.exists(caminho_meta) and not self.forcar_recarga:
                with open(caminho_meta, "r") as f:
                    meta_local = json.load(f)
                    if meta_local.get("tamanho_bytes") == tamanho_nuvem:
                        return False, tamanho_nuvem 
            return True, tamanho_nuvem
        except Exception as e:
            logging.warning(f"HEAD Check falhou para {ano_referencia}: {e}. Forçando download.")
            return True, 0

    def _coalescer_colunas_equivalentes(self, df, nome_canonico, aliases):
        """Funde nomenclaturas diferentes num único padrão canônico."""
        aliases_norm = [self._higienizar_texto(a) for a in aliases]
        colunas_existentes = [col for col in aliases_norm if col in df.columns]

        if not colunas_existentes: return df
        serie_final = pd.Series(np.nan, index=df.index)
        if nome_canonico in df.columns: serie_final = serie_final.combine_first(df[nome_canonico])
        for coluna in colunas_existentes: serie_final = serie_final.combine_first(df[coluna])
        df[nome_canonico] = serie_final
        return df

    def _construir_raw_operacional(self, dataframe_raw, ano_referencia):
        """Normalização rigorosa contra quebras de formato da SSP."""
        df = dataframe_raw.copy()
        df.columns = [self._higienizar_texto(c) for c in df.columns]

        for nome_canonico, aliases in MAPA_SEMANTICO_COLUNAS.items():
            df = self._coalescer_colunas_equivalentes(df, nome_canonico, aliases)

        for coluna in ESQUEMA_RAW_CANONICO.keys():
            if coluna not in df.columns and coluna != "ANO_BASE":
                df[coluna] = np.nan

        df["DESCR_SUBTIPOLOCAL"] = df["DESCR_SUBTIPOLOCAL"].combine_first(df["DESCR_TIPOLOCAL"])
        subtipo_norm = df["DESCR_SUBTIPOLOCAL"].fillna("").astype(str).map(self._higienizar_texto)
        mascara_subtipo = subtipo_norm.isin(SUBTIPOS_LOCAL_PERMITIDOS)
        df.loc[df["DESCR_TIPOLOCAL"].isna() & mascara_subtipo, "DESCR_TIPOLOCAL"] = "VIA PUBLICA"

        df = df[list(ESQUEMA_RAW_CANONICO.keys() - {"ANO_BASE"})].copy()
        df.to_parquet(f"datalake/raw/ssp_{ano_referencia}.parquet", index=False)
        return df

    def _ler_ou_baixar_raw(self, ano_referencia):
        endereco = f"https://www.ssp.sp.gov.br/assets/estatistica/transparencia/spDados/SPDadosCriminais_{ano_referencia}.xlsx"
        caminho_raw = f"datalake/raw/ssp_{ano_referencia}.parquet"
        
        precisa_baixar, tamanho_nuvem = self._verificar_atualizacao_remota(ano_referencia, endereco)

        if precisa_baixar or not os.path.exists(caminho_raw):
            logging.info(f"Download necessário p/ {ano_referencia} ({tamanho_nuvem} bytes).")
            res = self.sessao_web.get(endereco, timeout=120)
            res.raise_for_status()

            leitura_previa = pd.read_excel(io.BytesIO(res.content), nrows=60, header=None)
            termos = ["NUM_BO", "NUMERO_BO", "NATUREZA", "LATITUDE", "LAT", "DATA_FATO"]
            linha_cabecalho = next((i for i, row in leitura_previa.iterrows() if any(t in [self._higienizar_texto(str(c)) for c in row.values] for t in termos)), None)

            if linha_cabecalho is None: raise ValueError(f"Cabeçalho não encontrado: {ano_referencia}.")

            tabela = pd.read_excel(io.BytesIO(res.content), skiprows=linha_cabecalho, dtype=str)
            df_padronizado = self._construir_raw_operacional(tabela, ano_referencia)
            
            with open(f"datalake/metadata/tamanho_{ano_referencia}.json", "w") as f:
                json.dump({"tamanho_bytes": tamanho_nuvem, "ultima_att": str(datetime.now())}, f)

            self.auditoria["novos_dados"] = True
            self.auditoria["anos_redownload"].append(ano_referencia)
            return df_padronizado

        logging.info(f"RAW {ano_referencia} inalterada. Usando cache local.")
        return self._construir_raw_operacional(pd.read_parquet(caminho_raw), ano_referencia)

    def _analisar_raw(self, dataframe_raw, ano_referencia):
        obrigatorias = {"NATUREZA_APURADA", "LATITUDE", "LONGITUDE", "DATA_OCORRENCIA_BO"}
        self.raw_is_valid = obrigatorias.issubset(set(dataframe_raw.columns))

    # =========================================================
    # CAMADA TRUSTED & REFINED
    # =========================================================

    def _processar_trusted(self, dataframe_bruto, ano_referencia):
        df = dataframe_bruto.copy()
        for col in ESQUEMA_TRUSTED.keys():
            if col not in df.columns and col != "ANO_BASE": df[col] = np.nan
        df["ANO_BASE"] = ano_referencia
        vol_inicial = len(df)

        for col, tipo in ESQUEMA_TRUSTED.items():
            if col not in df.columns: continue
            if tipo == "string": df[col] = df[col].apply(self._higienizar_texto)
            elif tipo == "float": df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", "."), errors="coerce")
            elif tipo == "datetime": df[col] = pd.to_datetime(df[col], errors="coerce")
            elif tipo == "int": df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

        df["DATA_OCORRENCIA_BO"] = pd.to_datetime(df["DATA_OCORRENCIA_BO"], errors="coerce").dt.normalize()
        mascara_janela = df["DATA_OCORRENCIA_BO"].between(self.janela_inicio, self.janela_fim)

        mascara_geografica = (
            df["LATITUDE"].notna() & df["LONGITUDE"].notna() &
            (df["LATITUDE"] != 0) & (df["LONGITUDE"] != 0) &
            df["LATITUDE"].between(LIMITES_SP["lat"][0], LIMITES_SP["lat"][1]) &
            df["LONGITUDE"].between(LIMITES_SP["lon"][0], LIMITES_SP["lon"][1])
        )

        trusted = df.loc[mascara_janela & mascara_geografica, list(ESQUEMA_TRUSTED.keys())].copy()
        self.auditoria["falhas_integridade"] += max(0, vol_inicial - len(trusted))
        self.auditoria["volume_trusted"] += len(trusted)
        return trusted

    def _normalizar_hora(self, valor):
        if pd.isna(valor) or not str(valor).strip(): return np.nan
        texto = str(valor).strip()
        for fmt in ["%H:%M:%S", "%H:%M", "%H", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M:%S"]:
            try: return datetime.strptime(texto, fmt).hour
            except: continue
        try: return pd.to_datetime(texto, errors="coerce").hour
        except: return np.nan

    def _classificar_turno(self, hora):
        if pd.isna(hora): return np.nan
        h = int(hora)
        if 0 <= h < 6: return "Madrugada"
        if 6 <= h < 12: return "Manha"
        if 12 <= h < 18: return "Tarde"
        return "Noite"

    def _inferir_perfil_contextual(self, linha):
        perfis = set()
        ctx = f"{linha.get('NATUREZA_APURADA','')} {linha.get('DESCR_CONDUTA','')} {linha.get('DESCR_SUBTIPOLOCAL','')} {linha.get('RUBRICA','')}".upper()
        for perfil, palavras in PALAVRAS_CHAVE_PERFIL.items():
            if any(p in ctx for p in palavras): perfis.add(perfil)
        if not perfis: perfis.update(CATALOGO_CRIMES.get(linha.get("NATUREZA_APURADA"), {}).get("perfis", []))
        return list(perfis) if perfis else ["Indefinido"]

    def _imputar_turno(self, dataframe):
        df = dataframe.copy().reset_index(drop=True)
        moda_global = "Noite"
        base_moda = df.dropna(subset=["turno_operacional"])
        if base_moda.empty:
            df["turno_operacional"] = moda_global
            return df

        mapa_moda = base_moda.groupby(["codigo_geohash", "perfil"])["turno_operacional"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else moda_global).to_dict()
        df["chave_turno"] = list(zip(df["codigo_geohash"], df["perfil"]))
        mascara_nula = df["turno_operacional"].isna()
        df.loc[mascara_nula, "turno_operacional"] = df.loc[mascara_nula, "chave_turno"].map(mapa_moda)
        df["turno_operacional"] = df["turno_operacional"].fillna(moda_global)
        return df.drop(columns=["chave_turno"])

    def _gerar_geohash_seguro(self, lat, lon):
        try:
            if pd.isna(lat) or pd.isna(lon) or not np.isfinite(lat) or not np.isfinite(lon): return np.nan
            return str(gh.encode(float(lat), float(lon), precision=self.precisao_geohash))
        except: return np.nan

    def _processar_refined_eventos(self, trusted):
        df = trusted.copy()
        for col in ["NATUREZA_APURADA", "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL", "DESCR_CONDUTA", "RUBRICA"]:
            if col not in df.columns: df[col] = ""
            df[col] = df[col].fillna("").astype(str).map(self._higienizar_texto)

        df = df[df["NATUREZA_APURADA"].isin(CATALOGO_CRIMES.keys())].copy()
        df["DESCR_TIPOLOCAL"] = df["DESCR_TIPOLOCAL"].replace("", "VIA PUBLICA")
        df["DESCR_SUBTIPOLOCAL"] = df["DESCR_SUBTIPOLOCAL"].replace("", df["DESCR_TIPOLOCAL"])

        mascara = df["DESCR_TIPOLOCAL"].isin(TIPOS_LOCAL_PERMITIDOS) & (df["DESCR_SUBTIPOLOCAL"].isin(SUBTIPOS_LOCAL_PERMITIDOS) | (df["DESCR_SUBTIPOLOCAL"] == ""))
        df_filtrado = df[mascara].copy()
        if df_filtrado.empty: df_filtrado = df.copy()

        refined = df_filtrado[COLUNAS_REFINED_EVENTOS].dropna(subset=["LATITUDE", "LONGITUDE", "DATA_OCORRENCIA_BO", "NATUREZA_APURADA"]).copy()
        refined["perfis_afetados"] = refined.apply(self._inferir_perfil_contextual, axis=1)
        refined = refined.explode("perfis_afetados").dropna(subset=["perfis_afetados"])
        refined.rename(columns={"perfis_afetados": "perfil"}, inplace=True)

        refined["codigo_geohash"] = refined.apply(lambda l: self._gerar_geohash_seguro(l["LATITUDE"], l["LONGITUDE"]), axis=1)
        refined = refined.dropna(subset=["codigo_geohash"])
        refined = refined[refined["codigo_geohash"].str.len() == self.precisao_geohash].copy()

        refined["geohash_prefix_4"] = refined["codigo_geohash"].str[:4]
        refined["geohash_prefix_5"] = refined["codigo_geohash"].str[:5]
        refined["turno_operacional"] = refined["HORA_OCORRENCIA_BO"].apply(self._normalizar_hora).apply(self._classificar_turno)
        refined = self._imputar_turno(refined)
        refined["data_evento"] = pd.to_datetime(refined["DATA_OCORRENCIA_BO"]).dt.normalize()
        refined["peso_evento"] = refined["NATUREZA_APURADA"].apply(lambda x: float(CATALOGO_CRIMES.get(x, {}).get("peso", 1.0)))

        self.auditoria["volume_refined_eventos"] += len(refined)
        return refined

    # =========================================================
    # MODELAGEM PREDITIVA (SAFE & FAST ROUTING)
    # =========================================================

    def _criar_painel_diario(self, refined_eventos):
        chaves = ["codigo_geohash", "perfil", "turno_operacional"]
        agrupado = refined_eventos.groupby(chaves + ["geohash_prefix_4", "geohash_prefix_5", "data_evento"], as_index=False).agg(
            target_dia=("peso_evento", "sum"), ocorrencias_dia=("peso_evento", "size"),
            latitude_media=("LATITUDE", "mean"), longitude_media=("LONGITUDE", "mean")
        ).sort_values(chaves + ["data_evento"])

        agrupado["lag_1"] = agrupado.groupby(chaves)["target_dia"].shift(1).fillna(0.0)
        agrupado["lag_7"] = agrupado.groupby(chaves)["target_dia"].shift(7).fillna(0.0)
        agrupado["lag_14"] = agrupado.groupby(chaves)["target_dia"].shift(14).fillna(0.0)
        agrupado["media_7d"] = agrupado.groupby(chaves)["target_dia"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean()).fillna(0.0)
        agrupado["media_30d"] = agrupado.groupby(chaves)["target_dia"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean()).fillna(0.0)
        agrupado["ocorrencias_7d"] = agrupado.groupby(chaves)["ocorrencias_dia"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).sum()).fillna(0.0)
        agrupado["ocorrencias_30d"] = agrupado.groupby(chaves)["ocorrencias_dia"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).sum()).fillna(0.0)

        agrupado["dias_desde_ultimo_evento"] = agrupado.groupby(chaves)["data_evento"].diff().dt.days.fillna(999).clip(lower=0, upper=999)
        agrupado["dia_semana"] = agrupado["data_evento"].dt.dayofweek
        agrupado["mes"] = agrupado["data_evento"].dt.month
        agrupado["tendencia_7_30"] = agrupado["media_7d"] - agrupado["media_30d"]
        agrupado["intensidade_recente"] = agrupado["ocorrencias_7d"] / (agrupado["ocorrencias_30d"] + 1.0)
        return agrupado

    def _criar_target_futuro(self, painel):
        df = painel.copy()
        def somar_janela(serie):
            v = serie.to_numpy(dtype=float)
            res = np.zeros(len(v), dtype=float)
            for i in range(len(v)): res[i] = v[i+1 : i+1+self.horizonte_predicao_dias].sum()
            return pd.Series(res, index=serie.index)
        df["target_futuro_7d"] = df.groupby(["codigo_geohash", "perfil", "turno_operacional"])["target_dia"].transform(somar_janela)
        return df

    def _criar_fator_prophet(self, base_treino, datas_consulta):
        """Predição temporal realística projetando a série para o futuro."""
        serie = base_treino.groupby("data_evento", as_index=False).agg(y=("target_dia", "sum")).rename(columns={"data_evento": "ds"}).sort_values("ds")
        if serie.empty or len(serie) < 21:
            return pd.DataFrame({"data_evento": pd.to_datetime(datas_consulta).normalize(), "fator_prophet": 1.0})

        limite_previsao = max(pd.to_datetime(datas_consulta).max(), serie["ds"].max() + pd.Timedelta(days=self.horizonte_predicao_dias))
        calendario = pd.DataFrame({"ds": pd.date_range(serie["ds"].min(), limite_previsao, freq="D")})

        try:
            modelo = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            modelo.fit(serie)
            forecast = modelo.predict(calendario)[["ds", "yhat"]]
            media_yhat = max(forecast["yhat"].mean(), 1.0)
            forecast["fator_prophet"] = (forecast["yhat"].clip(lower=0.0) / media_yhat).clip(0.25, 4.0)
            return forecast.rename(columns={"ds": "data_evento"})[["data_evento", "fator_prophet"]]
        except:
            return pd.DataFrame({"data_evento": pd.to_datetime(datas_consulta).normalize(), "fator_prophet": 1.0})

    def _split_temporal(self, base_modelagem):
        cutoff = base_modelagem["data_evento"].max() - pd.Timedelta(days=self.dias_holdout_teste)
        return base_modelagem[base_modelagem["data_evento"] < cutoff].copy(), base_modelagem[base_modelagem["data_evento"] >= cutoff].copy()

    def _aplicar_encoders(self, treino, teste):
        enc_turno, enc_perfil = LabelEncoder(), LabelEncoder()
        treino["turno_enc"] = enc_turno.fit_transform(treino["turno_operacional"])
        treino["perfil_enc"] = enc_perfil.fit_transform(treino["perfil"])
        mapa_t = {v: i for i, v in enumerate(enc_turno.classes_)}
        mapa_p = {v: i for i, v in enumerate(enc_perfil.classes_)}
        teste["turno_enc"] = teste["turno_operacional"].map(mapa_t).fillna(0).astype(int)
        teste["perfil_enc"] = teste["perfil"].map(mapa_p).fillna(0).astype(int)
        return treino, teste, enc_turno, enc_perfil

    def _treinar_modelo(self, treino):
        features = ["latitude_media", "longitude_media", "perfil_enc", "turno_enc", "lag_1", "lag_7", "lag_14", "media_7d", "media_30d", "ocorrencias_7d", "ocorrencias_30d", "dias_desde_ultimo_evento", "dia_semana", "mes", "tendencia_7_30", "intensidade_recente", "fator_prophet"]
        modelo = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=500, max_depth=6, learning_rate=0.04, subsample=0.9, random_state=42, n_jobs=4)
        modelo.fit(treino[features], treino["target_futuro_7d"])
        return modelo, features

    def _avaliar_modelo(self, modelo, teste, features):
        previsoes = np.clip(modelo.predict(teste[features]), 0.0, None)
        self.auditoria["mae_teste"] = round(float(mean_absolute_error(teste["target_futuro_7d"], previsoes)), 4)
        self.auditoria["rmse_teste"] = round(float(math.sqrt(mean_squared_error(teste["target_futuro_7d"], previsoes))), 4)

    def _treinar_modelo_final(self, base, enc_turno, enc_perfil):
        mapa_t = {v: i for i, v in enumerate(enc_turno.classes_)}
        mapa_p = {v: i for i, v in enumerate(enc_perfil.classes_)}
        base["turno_enc"] = base["turno_operacional"].map(mapa_t).fillna(0).astype(int)
        base["perfil_enc"] = base["perfil"].map(mapa_p).fillna(0).astype(int)
        modelo, features = self._treinar_modelo(base)
        return modelo, features, base

    def _classificar_faixa_risco(self, score):
        if score < 3.5: return "baixo"
        if score < 6.5: return "medio"
        if score < 8.5: return "alto"
        return "critico"

    def _gerar_malha_semanal(self, base_modelagem, modelo_final, features):
        base = base_modelagem.copy()
        base["score_bruto"] = np.clip(modelo_final.predict(base[features]), 0.0, None)
        snapshot = base.sort_values("data_evento").groupby(["codigo_geohash", "geohash_prefix_4", "geohash_prefix_5", "perfil", "turno_operacional"], as_index=False).tail(1).copy()
        
        escala = snapshot["score_bruto"].quantile(0.95)
        if pd.isna(escala) or escala <= 0: escala = max(snapshot["score_bruto"].max(), 1.0)
        snapshot["score"] = ((snapshot["score_bruto"] / escala) * 10.0).clip(0.5, 10.0).round(2)
        snapshot["risk_band"] = snapshot["score"].apply(self._classificar_faixa_risco)
        
        # FEATURE: Routing Penalty para grafos de roteamento Mobile
        snapshot["routing_penalty"] = (1.0 + (snapshot["score"] * 0.15)).round(2)

        snapshot["semana_referencia_inicio"] = (self.janela_fim + pd.Timedelta(days=1)).normalize()
        snapshot["semana_referencia_fim"] = (snapshot["semana_referencia_inicio"] + pd.Timedelta(days=self.horizonte_predicao_dias - 1)).normalize()
        return snapshot

    # =========================================================
    # FIRESTORE DELTA SYNC
    # =========================================================

    def _hash_registro(self, registro):
        return hashlib.sha256(json.dumps(registro, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _sincronizacao_delta_firestore(self, malha_final):
        colecao = self.banco_nuvem.collection("niveis_risco")
        documentos_atuais = {doc.id: doc.to_dict() for doc in colecao.stream()}
        lote = self.banco_nuvem.batch()
        operacoes, ids_novos = 0, set()

        for _, linha in malha_final.iterrows():
            doc_id = f"{linha['codigo_geohash']}_{linha['perfil']}_{linha['turno_operacional']}"
            ids_novos.add(doc_id)
            
            payload = {
                "geohash": linha["codigo_geohash"], "perfil": linha["perfil"], "turno": linha["turno_operacional"],
                "periodo": linha["turno_operacional"], "score": round(float(linha["score"]), 2),
                "routing_penalty": float(linha["routing_penalty"]), "risk_band": linha["risk_band"],
                "geohash_prefix_4": linha["geohash_prefix_4"], "geohash_prefix_5": linha["geohash_prefix_5"]
            }
            payload["hash_registro"] = self._hash_registro(payload)
            payload["ultima_atualizacao"] = firestore.SERVER_TIMESTAMP

            atual = documentos_atuais.get(doc_id, {})
            if atual.get("hash_registro") != payload["hash_registro"]:
                lote.set(colecao.document(doc_id), payload, merge=True)
                operacoes += 1
                if operacoes >= 450: lote.commit(); lote = self.banco_nuvem.batch(); operacoes = 0

        # Autolimpeza de Geohashes frios
        for doc_id in set(documentos_atuais.keys()) - ids_novos:
            lote.delete(colecao.document(doc_id))
            operacoes += 1
            if operacoes >= 450: lote.commit(); lote = self.banco_nuvem.batch(); operacoes = 0

        if operacoes > 0: lote.commit()

    # =========================================================
    # ORQUESTRAÇÃO DO PIPELINE
    # =========================================================

    def executar_pipeline_completo(self):
        bases_refined_eventos = []
        try:
            for ano_alvo in self.periodo_historico:
                raw = self._ler_ou_baixar_raw(ano_alvo)
                self._analisar_raw(raw, ano_alvo)
                if not getattr(self, "raw_is_valid", False): continue

                trusted = self._processar_trusted(raw, ano_alvo)
                if trusted.empty: continue
                
                refined = self._processar_refined_eventos(trusted)
                if not refined.empty: bases_refined_eventos.append(refined)

            if not bases_refined_eventos: raise ValueError("Camada Refined vazia.")
            
            base_modelagem = self._montar_base_supervisionada(pd.concat(bases_refined_eventos, ignore_index=True))
            treino, teste = self._split_temporal(base_modelagem)
            treino, teste, enc_turno, enc_perfil = self._aplicar_encoders(treino, teste)
            
            modelo_val, features = self._treinar_modelo(treino)
            self._avaliar_modelo(modelo_val, teste, features)
            
            modelo_final, features, base_final = self._treinar_modelo_final(base_modelagem, enc_turno, enc_perfil)
            malha = self._gerar_malha_semanal(base_final, modelo_final, features)
            
            if self.banco_nuvem: self._sincronizacao_delta_firestore(malha)
            self._notificar_sucesso()
        except Exception as e:
            self._notificar_erro(str(e))
            raise

if __name__ == "__main__": MotorSafeDriver().executar_pipeline_completo()
