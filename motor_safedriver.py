import os
import io
import json
import logging
import unicodedata
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union

import numpy as np
import pandas as pd
import pygeohash as gh
import requests
import xgboost as xgb
import holidays

import firebase_admin
from firebase_admin import credentials, firestore


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
AnyStr = Union[str, bytes]


LAKE_DIR = Path(os.environ.get("LAKE_DIR", "data_lake"))

RAW_SSP_DIR = LAKE_DIR / "raw" / "ssp"
REF_EVENTS_DIR = LAKE_DIR / "refined" / "events"
REF_GRID_DIR = LAKE_DIR / "refined" / "grid"
REF_FORECAST_DIR = LAKE_DIR / "refined" / "forecast" 

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "modelos"))
STATE_DIR = Path(os.environ.get("STATE_DIR", "state"))

RAW_SSP_DIR.mkdir(parents=True, exist_ok=True)
REF_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
REF_GRID_DIR.mkdir(parents=True, exist_ok=True)
REF_FORECAST_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_ENGINE = os.environ.get("PARQUET_ENGINE", "pyarrow")
PARQUET_COMPRESSION = os.environ.get("PARQUET_COMPRESSION", "zstd") 

SSP_FORCE_REFRESH = os.environ.get("SSP_FORCE_REFRESH", "0").strip().lower() in ("1", "true", "yes", "y")
GEOHASH_PRECISION_APP = int(os.environ.get("GEOHASH_PRECISION_APP", "7"))


MIN_NOVOS_BO_TREINO = int(os.environ.get("MIN_NOVOS_BO_TREINO", "200"))
TREINO_COOLDOWN_H = int(os.environ.get("TREINO_COOLDOWN_H", "24"))
TRAIN_WINDOW_MONTHS = int(os.environ.get("TRAIN_WINDOW_MONTHS", "24"))

XGB_ALPHA = float(os.environ.get("XGB_ALPHA", "0.6"))
XGB_ALPHA = min(max(XGB_ALPHA, 0.0), 1.0)

XGB_MAX_ROWS = int(os.environ.get("XGB_MAX_ROWS", "50000"))
XGB_MAX_DEPTH = int(os.environ.get("XGB_MAX_DEPTH", "4"))
XGB_MAX_BIN = int(os.environ.get("XGB_MAX_BIN", "128"))
XGB_NUM_BOOST_ROUND = int(os.environ.get("XGB_NUM_BOOST_ROUND", "1200"))
XGB_EARLY_STOP = int(os.environ.get("XGB_EARLY_STOP", "50"))

XGB_MODEL_PATH = Path(os.environ.get("XGB_MODEL_PATH", str(MODEL_DIR / "modelo_safedriver_xgb.json")))
TRAIN_STATE_PATH = STATE_DIR / "train_state.json"
SSP_STATE_PATH = STATE_DIR / "ssp_state.json"

FIRESTORE_COLLECTION_RISK = os.environ.get("FIRESTORE_COLLECTION_RISK", "risk_cells_v2")


COLS_SSP_MIN = [
    "NUM_BO",
    "DATA_OCORRENCIA_BO",
    "HORA_OCORRENCIA_BO",
    "LATITUDE",
    "LONGITUDE",
    "NATUREZA_APURADA",
    "NOME_MUNICIPIO",
    "DESCR_TIPOLOCAL",
    "DESCR_SUBTIPOLOCAL",
]


@dataclass(frozen=True)
class CrimeInfo:
    artigo: int
    classe: str
    peso: float


class SafeDriverEngine:
    """
Pipeline SafeDriver - ETL e geração de mapa de criminalidade urbana
    """

    def __init__(self) -> None:
        self.db = self._iniciar_persistencia()
        self.modelo_xgb: Optional[xgb.Booster] = None

        self.crime_map: Dict[str, CrimeInfo] = {
            "FURTO DE VEICULO": CrimeInfo(155, "MEDIO", 1.0),
            "FURTO DE CARGA": CrimeInfo(155, "MEDIO", 1.2),
            "ROUBO DE VEICULO": CrimeInfo(157, "GRAVE", 2.5),
            "ROUBO DE CARGA": CrimeInfo(157, "GRAVE", 3.0),
            "LATROCINIO": CrimeInfo(157, "GRAVISSIMO", 5.0),
            "EXTORSAO MEDIANTE A SEQUESTRO": CrimeInfo(159, "GRAVISSIMO", 5.0),
        }

        self.tipos_local_validos = [
            self._limpar_texto("Via Pública"),
            self._limpar_texto("Rodovia/Estrada"),
        ]

        self.subtipos_local_validos = [
            "VIA PUBLICA",
            "TRANSEUNTE",
            "ACOSTAMENTO",
            "AREA DE DESCANSO",
            "BALANCA",
            "CICLOFAIXA",
            "DE FRENTE A RESIDENCIA DA VITIMA",
            "FEIRA LIVRE",
            "INTERIOR DE VEICULO DE CARGA",
            "INTERIOR DE VEICULO DE PARTICULAR",
            "POSTO DE AUXILIO",
            "POSTO DE FISCALIZACAO",
            "POSTO POLICIAL",
            "PRACA",
            "PRACA DE PEDAGIO",
            "SEMAFORO",
            "TUNEL/VIADUTO/PONTE",
            "VEICULO EM MOVIMENTO",
        ]

        self.peso_periodo = {
            "MADRUGADA": 1.3,
            "MANHA": 0.9,
            "TARDE": 1.0,
            "NOITE": 1.2,
            "INDEFINIDO": 1.0,
        }

        self.calendario_feriados = holidays.Brazil()

  
    def executar_pipeline(self, ano_inicio: int = 2022, ano_fim: Optional[int] = None) -> None:
        inicio_execucao = datetime.now()
        referencia = inicio_execucao

        if ano_fim is None:
            ano_fim = referencia.year
        ano_atual = referencia.year

        logging.info("SafeDriver: iniciando (%s–%s). Ano atual=%s", ano_inicio, ano_fim, ano_atual)

   
        df_hist = self._carregar_historico(ano_inicio, ano_fim)
        total_brutos = len(df_hist)
        logging.info("Ingestão SSP (raw parquet): %d registros.", total_brutos)

    
        if "NUM_BO" not in df_hist.columns:
            df_hist["NUM_BO"] = None
        df_hist["NUM_BO"] = df_hist["NUM_BO"].astype(str).map(self._limpar_texto)

     
        df_ano_atual = df_hist[df_hist["ANO_BASE"] == ano_atual].copy() if "ANO_BASE" in df_hist.columns else df_hist.copy()
        index_bo = self._carregar_index_bo_refined(ano_atual)

        df_ano_atual["BO_CONHECIDO"] = df_ano_atual["NUM_BO"].isin(index_bo)
        df_novos = df_ano_atual[~df_ano_atual["BO_CONHECIDO"]].copy()
        qtd_novos_bo = df_novos["NUM_BO"].dropna().nunique()

        logging.info("BOs novos (ano atual %s): %d", ano_atual, qtd_novos_bo)
        if qtd_novos_bo == 0:
            logging.info("Sem BO novo. Encerrando (sem recalcular malha).")
            return

        # Processamento: saneamento + filtros + pesos
        df_geo = self._sanear_geo(df_hist)
        df_norm = self._normalizar_campos_texto(df_geo)
        df_filtrado = self._filtrar_conteudo(df_norm)

        df_pesos, total_qualificados, aproveitamento = self._aplicar_pesos(df_filtrado, referencia, total_brutos)
        if total_qualificados == 0:
            logging.info("Nenhum registro qualificado após filtros.")
            return

       
        if TRAIN_WINDOW_MONTHS > 0:
            cutoff = pd.Timestamp(referencia) - pd.DateOffset(months=TRAIN_WINDOW_MONTHS)
            df_pesos = df_pesos[df_pesos["DATA_OCORRENCIA_BO"].notna() & (df_pesos["DATA_OCORRENCIA_BO"] >= cutoff)].copy()

     
        df_eventos, grid = self._preparar_grid(df_pesos)

   
        if self._deve_treinar(qtd_novos_bo, referencia):
            self.modelo_xgb = self._treinar_xgb(grid)
            self._salvar_estado_treino({"last_train_at": referencia.isoformat(), "last_novos_bo": qtd_novos_bo})
        else:
            self.modelo_xgb = self._carregar_modelo_xgb_se_existir()

        grid = self._aplicar_modelo_xgb(grid)

        
        self._exportar_events_refined_parquet(df_eventos, referencia)
        self._exportar_grid_refined_parquet(grid, referencia)

      
        self._persistir_grid_app_firestore(grid)

        logging.info(
            "Concluído: brutos=%d qualificados=%d (%.1f%%) grid=%d novos_bo=%d tempo=%ss",
            total_brutos,
            total_qualificados,
            aproveitamento,
            len(grid),
            qtd_novos_bo,
            (datetime.now() - inicio_execucao).seconds,
        )


    def _limpar_texto(self, texto: AnyStr) -> str:
        if not isinstance(texto, str):
            texto = str(texto)
        nfkd = unicodedata.normalize("NFKD", texto)
        return "".join(c for c in nfkd if not unicodedata.combining(c)).upper().strip()

    def _iniciar_persistencia(self):
        secret_json = os.environ.get("FIREBASE_JSON")
        if not secret_json:
            logging.info("FIREBASE_JSON não definido. Firestore desativado.")
            return None
        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(secret_json))
            firebase_admin.initialize_app(cred)
        logging.info("Firestore conectado.")
        return firestore.client()


    def _load_json(self, path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _save_json(self, path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _carregar_estado_treino(self) -> dict:
        return self._load_json(TRAIN_STATE_PATH)

    def _salvar_estado_treino(self, payload: dict) -> None:
        self._save_json(TRAIN_STATE_PATH, payload)

    def _carregar_estado_ssp(self) -> dict:
        return self._load_json(SSP_STATE_PATH)

    def _salvar_estado_ssp(self, payload: dict) -> None:
        self._save_json(SSP_STATE_PATH, payload)


    def _deve_treinar(self, novos_bo: int, referencia: datetime) -> bool:
        if not XGB_MODEL_PATH.exists():
            return True
        if novos_bo >= MIN_NOVOS_BO_TREINO:
            return True

        estado = self._carregar_estado_treino()
        last = estado.get("last_train_at")
        if not last:
            return True

        try:
            last_dt = datetime.fromisoformat(last)
        except Exception:
            return True

        return (referencia - last_dt) >= timedelta(hours=TREINO_COOLDOWN_H)

    def _carregar_modelo_xgb_se_existir(self) -> Optional[xgb.Booster]:
        if not XGB_MODEL_PATH.exists():
            return None
        try:
            booster = xgb.Booster()
            booster.load_model(str(XGB_MODEL_PATH))
            logging.info("Modelo XGB carregado (%s).", XGB_MODEL_PATH)
            return booster
        except Exception as e:
            logging.warning("Falha ao carregar modelo XGB (%s).", e)
            return None

   
    def _ssp_url(self, ano: int) -> str:
        return (
            "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/"
            f"spDados/SPDadosCriminais_{ano}.xlsx"
        )

    def _raw_parquet_path(self, ano: int) -> Path:
        return RAW_SSP_DIR / f"ano={ano}" / "ssp_raw.parquet"


    def _parse_ssp_xlsx_to_df(self, xlsx_bytes: bytes, ano: int) -> pd.DataFrame:
        excel = pd.ExcelFile(io.BytesIO(xlsx_bytes))
        df_bruto = pd.DataFrame()

        for aba in excel.sheet_names:
            df_temp = excel.parse(aba, header=None)
            indice_cabecalho = 0
            for i, linha in df_temp.head(25).iterrows():
                linha_upper = [self._limpar_texto(c) for c in linha.values]
                if "LATITUDE" in linha_upper:
                    indice_cabecalho = i
                    break

            df_corrigido = excel.parse(aba, skiprows=indice_cabecalho)
            df_corrigido.columns = [self._limpar_texto(str(c)) for c in df_corrigido.columns]

            cols_presentes = [c for c in COLS_SSP_MIN if c in df_corrigido.columns]
            df_corrigido = df_corrigido[cols_presentes].copy()

            df_corrigido["ANO_BASE"] = ano
            df_bruto = pd.concat([df_bruto, df_corrigido], ignore_index=True)

        return df_bruto

    def _carregar_dados_ssp(self, ano: int) -> pd.DataFrame:
        """
        CONTRATO mantido. Otimização:
        - anos passados: usa RAW parquet
        - ano atual: valida atualização (HEAD/ETag/Last-Modified + GET condicional + sha256)
        """
        out_path = self._raw_parquet_path(ano)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        ano_atual = datetime.now().year

       
        if ano != ano_atual and out_path.exists() and not SSP_FORCE_REFRESH:
            logging.info("SSP %s: usando RAW parquet cache (ano passado).", ano)
            return pd.read_parquet(out_path)

        url = self._ssp_url(ano)
        state = self._carregar_estado_ssp()
        meta = state.get(str(ano), {})

       
        if out_path.exists() and not SSP_FORCE_REFRESH:
            try:
                h = requests.head(url, timeout=30, allow_redirects=True)
                if h.ok:
                    etag = h.headers.get("ETag")
                    last_mod = h.headers.get("Last-Modified")
                    content_len = h.headers.get("Content-Length")

                    if etag and meta.get("etag") and etag == meta["etag"]:
                        logging.info("SSP %s: sem update (ETag igual).", ano)
                        return pd.read_parquet(out_path)
                    if last_mod and meta.get("last_modified") and last_mod == meta["last_modified"]:
                        logging.info("SSP %s: sem update (Last-Modified igual).", ano)
                        return pd.read_parquet(out_path)
                    if content_len and meta.get("content_length") and content_len == meta["content_length"]:
                        logging.info("SSP %s: sem update provável (Content-Length igual).", ano)
                        return pd.read_parquet(out_path)
            except Exception:
                pass

       
        headers = {}
        if not SSP_FORCE_REFRESH:
            if meta.get("etag"):
                headers["If-None-Match"] = meta["etag"]
            if meta.get("last_modified"):
                headers["If-Modified-Since"] = meta["last_modified"]

        logging.info("SSP %s: consultando online (GET condicional).", ano)
        resp = requests.get(url, timeout=120, headers=headers, allow_redirects=True)

        if resp.status_code == 304 and out_path.exists() and not SSP_FORCE_REFRESH:
            logging.info("SSP %s: 304 Not Modified. Usando RAW parquet cache.", ano)
            return pd.read_parquet(out_path)

        resp.raise_for_status()
        content = resp.content
        sha = hashlib.sha256(content).hexdigest()

       
        if out_path.exists() and not SSP_FORCE_REFRESH and meta.get("sha256") == sha:
            logging.info("SSP %s: conteúdo idêntico (sha256). Usando RAW parquet cache.", ano)
            return pd.read_parquet(out_path)

        # salva parquet
        df = self._parse_ssp_xlsx_to_df(content, ano)
        df.to_parquet(out_path, index=False, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION)

        state[str(ano)] = {
            "etag": resp.headers.get("ETag"),
            "last_modified": resp.headers.get("Last-Modified"),
            "content_length": resp.headers.get("Content-Length"),
            "sha256": sha,
            "cached_at": datetime.utcnow().isoformat(),
            "rows": int(len(df)),
        }
        self._salvar_estado_ssp(state)

        logging.info("SSP %s: atualizado (RAW parquet) linhas=%d", ano, len(df))
        return df

    def _carregar_historico(self, ano_inicio: int = 2022, ano_fim: Optional[int] = None) -> pd.DataFrame:
        if ano_fim is None:
            ano_fim = datetime.now().year

        dfs: List[pd.DataFrame] = []
        for ano in range(ano_inicio, ano_fim + 1):
            try:
                dfs.append(self._carregar_dados_ssp(ano))
            except Exception as e:
                logging.warning("Ano %s falhou (%s). Pulando.", ano, e)

        if not dfs:
            raise RuntimeError("Nenhum ano carregado na ingestão.")
        return pd.concat(dfs, ignore_index=True)

  
    def _carregar_index_bo_refined(self, ano: int) -> set[str]:
        part_dir = REF_EVENTS_DIR / f"ano={ano}"
        if not part_dir.exists():
            return set()
        paths = list(part_dir.rglob("*.parquet"))
        if not paths:
            return set()

        frames = []
        for p in paths:
            try:
                frames.append(pd.read_parquet(p, columns=["NUM_BO"]))
            except Exception:
                continue
        if not frames:
            return set()

        df = pd.concat(frames, ignore_index=True)
        s = df["NUM_BO"].dropna().astype(str).map(self._limpar_texto)
        return set(s.unique().tolist())


    def _sanear_geo(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        df["LATITUDE"] = pd.to_numeric(df.get("LATITUDE"), errors="coerce")
        df["LONGITUDE"] = pd.to_numeric(df.get("LONGITUDE"), errors="coerce")
        df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
        df = df[df["LATITUDE"] != 0]
        return df

    def _normalizar_campos_texto(self, df_geo: pd.DataFrame) -> pd.DataFrame:
        df = df_geo.copy()
        for col in ["NATUREZA_APURADA", "NOME_MUNICIPIO", "DESCR_TIPOLOCAL", "DESCR_SUBTIPOLOCAL"]:
            if col in df.columns:
                df[col] = df[col].astype(str).map(self._limpar_texto)
        return df

    def _filtrar_conteudo(self, df_norm: pd.DataFrame) -> pd.DataFrame:
        df = df_norm.copy()
        df = df[df["NATUREZA_APURADA"].isin(list(self.crime_map.keys()))]
        df = df[df["DESCR_TIPOLOCAL"].isin(self.tipos_local_validos)]
        df = df[df["DESCR_SUBTIPOLOCAL"].isin(self.subtipos_local_validos)]
        return df

 
    def _definir_periodo(self, hora: AnyStr) -> str:
        try:
            h = int(str(hora).split(":")[0])
            if 0 <= h < 6:
                return "MADRUGADA"
            if 6 <= h < 12:
                return "MANHA"
            if 12 <= h < 18:
                return "TARDE"
            return "NOITE"
        except Exception:
            return "INDEFINIDO"

    def _classificar_qualificacao(self, natureza_limpa: str) -> Optional[str]:
        info = self.crime_map.get(natureza_limpa)
        if not info:
            return None
        return "ALERTA" if info.classe in ("GRAVE", "GRAVISSIMO") else "QUALIFICADO"

    def _peso_crime(self, natureza_limpa: str) -> float:
        info = self.crime_map.get(natureza_limpa)
        return float(info.peso) if info else 0.5

    def _calcular_peso_recencia(self, dias: Any) -> float:
        if pd.isna(dias) or dias < 0:
            return 1.0
        if dias <= 30:
            return 1.5
        if dias <= 90:
            return 1.3
        if dias <= 365:
            return 1.1
        return 1.0

    def _peso_local(self, tipo: AnyStr, subtipo: AnyStr) -> float:
        t = self._limpar_texto(tipo) if isinstance(tipo, str) else ""
        s = self._limpar_texto(subtipo) if isinstance(subtipo, str) else ""
        return 1.2 if (t in self.tipos_local_validos and s in self.subtipos_local_validos) else 1.0

    def _aplicar_pesos(
        self,
        df_filtrado_conteudo: pd.DataFrame,
        referencia: datetime,
        total_registros_brutos: int,
    ) -> Tuple[pd.DataFrame, int, float]:
        df = df_filtrado_conteudo.copy()

        df["QUALIFICACAO"] = df["NATUREZA_APURADA"].apply(self._classificar_qualificacao)
        df = df[df["QUALIFICACAO"].notna()].copy()

        total_qualificados = len(df)
        aproveitamento_final = (total_qualificados / total_registros_brutos) * 100 if total_registros_brutos else 0.0

        df["DATA_OCORRENCIA_BO"] = pd.to_datetime(df.get("DATA_OCORRENCIA_BO"), errors="coerce", dayfirst=True)
        df["DIAS_DESDE_OCORRENCIA"] = (pd.Timestamp(referencia).normalize() - df["DATA_OCORRENCIA_BO"]).dt.days

        data_col = df["DATA_OCORRENCIA_BO"]
        data_date = data_col.dt.date

        df["IS_FERIADO"] = data_date.isin(self.calendario_feriados).astype(int)
        df["IS_FIM_SEMANA"] = data_col.dt.dayofweek.isin([5, 6]).astype(int)
        df["IS_VESPERA_FERIADO"] = ((data_col + pd.Timedelta(days=1)).dt.date.isin(self.calendario_feriados)).astype(int)

        df["PESO_RECENCIA"] = df["DIAS_DESDE_OCORRENCIA"].apply(self._calcular_peso_recencia)
        df["PESO_CRIME"] = df["NATUREZA_APURADA"].apply(self._peso_crime)

        hora_col = df.get("HORA_OCORRENCIA_BO")
        if hora_col is not None:
            df["PERIODO"] = hora_col.apply(self._definir_periodo)
        else:
            df["PERIODO"] = "INDEFINIDO"

        df["PESO_PERIODO"] = df["PERIODO"].map(self.peso_periodo).fillna(1.0)
        df["PESO_LOCAL"] = df.apply(
            lambda r: self._peso_local(r.get("DESCR_TIPOLOCAL"), r.get("DESCR_SUBTIPOLOCAL")),
            axis=1,
        )

        df["PESO_RISCO"] = df["PESO_CRIME"] * df["PESO_RECENCIA"] * df["PESO_PERIODO"] * df["PESO_LOCAL"]
        return df, total_qualificados, aproveitamento_final

   
    def _preparar_grid(self, df_com_pesos: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df_com_pesos.copy()

        df["perfil"] = df["QUALIFICACAO"].apply(
            lambda x: ["Pedestre", "Ciclista", "Motorista"] if x == "ALERTA" else ["Pedestre", "Ciclista"]
        )
        df_eventos = df.explode("perfil").copy()

        df_eventos["geohash"] = [
            gh.encode(la, lo, precision=GEOHASH_PRECISION_APP)
            for la, lo in zip(df_eventos["LATITUDE"], df_eventos["LONGITUDE"])
        ]

        keys = ["geohash", "perfil", "PERIODO"]

        grid = (
            df_eventos.groupby(keys)
            .agg(
                frequencia=("PESO_RISCO", "size"),
                risco_total=("PESO_RISCO", "sum"),
                risco_medio=("PESO_RISCO", "mean"),
                dias_medio=("DIAS_DESDE_OCORRENCIA", "mean"),
                feriado_pct=("IS_FERIADO", "mean"),
                fim_semana_pct=("IS_FIM_SEMANA", "mean"),
                vespera_feriado_pct=("IS_VESPERA_FERIADO", "mean"),
            )
            .reset_index()
        )

        q95 = grid["risco_total"].quantile(0.95)
        if pd.isna(q95) or q95 <= 0:
            q95 = 1.0
        grid["score"] = ((grid["risco_total"] / q95) * 10.0).clip(0.5, 10.0).round(2)
        return df_eventos, grid

    def _montar_features_xgb(self, df_grid: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df = df_grid.copy()

        perfil_map = {"Pedestre": 0, "Ciclista": 1, "Motorista": 2}
        periodo_map = {"MADRUGADA": 0, "MANHA": 1, "TARDE": 2, "NOITE": 3, "INDEFINIDO": 4}

        df["perfil_id"] = df["perfil"].map(perfil_map).fillna(-1).astype(np.float32)
        df["periodo_id"] = df["PERIODO"].map(periodo_map).fillna(-1).astype(np.float32)
        df["score_heuristico"] = df["score"].astype(np.float32)

        feature_cols = [
            "frequencia",
            "risco_total",
            "risco_medio",
            "dias_medio",
            "feriado_pct",
            "fim_semana_pct",
            "vespera_feriado_pct",
            "perfil_id",
            "periodo_id",
            "score_heuristico",
        ]
        X = df[feature_cols].astype(np.float32)
        return X, feature_cols

    def _treinar_xgb(self, resumo_grid: pd.DataFrame) -> xgb.Booster:
        X, feature_cols = self._montar_features_xgb(resumo_grid)

        q75 = resumo_grid["risco_total"].quantile(0.75)
        y = (resumo_grid["risco_total"] >= q75).astype(int).values

        idx_pos = np.where(y == 1)[0]
        idx_neg = np.where(y == 0)[0]

        rng = np.random.default_rng(42)
        neg_keep = min(len(idx_neg), max(len(idx_pos) * 3, 2000))
        if len(idx_neg) > neg_keep:
            idx_neg = rng.choice(idx_neg, size=neg_keep, replace=False)

        idx = np.concatenate([idx_pos, idx_neg])
        rng.shuffle(idx)

        if len(idx) > XGB_MAX_ROWS:
            idx = idx[:XGB_MAX_ROWS]

        Xs = X.iloc[idx].reset_index(drop=True)
        ys = y[idx]

        n = len(Xs)
        n_train = int(n * 0.8)
        X_train, X_val = Xs.iloc[:n_train], Xs.iloc[n_train:]
        y_train, y_val = ys[:n_train], ys[n_train:]

        pos = max(int((y_train == 1).sum()), 1)
        neg = max(int((y_train == 0).sum()), 1)
        spw = neg / pos

        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_cols)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_cols)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": 0.07,
            "max_depth": XGB_MAX_DEPTH,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1.0,
            "tree_method": "hist",
            "max_bin": XGB_MAX_BIN,
            "scale_pos_weight": float(spw),
        }

        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=XGB_NUM_BOOST_ROUND,
            evals=[(dval, "val")],
            early_stopping_rounds=XGB_EARLY_STOP,
            verbose_eval=False,
        )

        XGB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(XGB_MODEL_PATH))
        logging.info("XGB treinado: best_iter=%s, salvo=%s", booster.best_iteration, XGB_MODEL_PATH)
        return booster

    def _aplicar_modelo_xgb(self, resumo_grid: pd.DataFrame) -> pd.DataFrame:
        df = resumo_grid.copy()
        if not self.modelo_xgb:
            df["score_final"] = df["score"]
            return df

        X, feature_cols = self._montar_features_xgb(df)
        dmat = xgb.DMatrix(X, feature_names=feature_cols)

        try:
            proba = self.modelo_xgb.predict(dmat)
        except Exception as e:
            logging.warning("Inferência XGB falhou (%s). Usando heurístico.", e)
            df["score_final"] = df["score"]
            return df

        df["proba_modelo"] = proba
        df["score_modelo"] = (df["proba_modelo"] * 10.0).clip(0.5, 10.0)
        df["score_final"] = (XGB_ALPHA * df["score_modelo"] + (1 - XGB_ALPHA) * df["score"]).round(2)
        return df

   
    def _exportar_events_refined_parquet(self, df_eventos: pd.DataFrame, referencia: datetime) -> None:
        df = df_eventos.copy()
        df["TIMESTAMP_ATUALIZACAO"] = pd.Timestamp(referencia)

        df["DATA_OCORRENCIA_BO"] = pd.to_datetime(df.get("DATA_OCORRENCIA_BO"), errors="coerce", dayfirst=True)
        df["ANO_PART"] = df["DATA_OCORRENCIA_BO"].dt.year.fillna(df.get("ANO_BASE", referencia.year)).astype(int)
        df["MES_PART"] = df["DATA_OCORRENCIA_BO"].dt.month.fillna(referencia.month).astype(int)
        df["NUM_BO"] = df["NUM_BO"].astype(str).map(self._limpar_texto)

        ts = referencia.strftime("%Y%m%dT%H%M%S")

        part_keys = df[["ANO_PART", "MES_PART"]].drop_duplicates().values.tolist()
        for ano, mes in part_keys:
            out_dir = REF_EVENTS_DIR / f"ano={int(ano)}" / f"mes={int(mes):02d}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"part-{ts}.parquet"

            df_part = df[(df["ANO_PART"] == ano) & (df["MES_PART"] == mes)].copy()
            df_part.drop(columns=["ANO_PART", "MES_PART"], inplace=True, errors="ignore")

            df_part.to_parquet(out_path, index=False, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION)
            logging.info("REF events salvo: %s (linhas=%d)", out_path, len(df_part))

    def _exportar_grid_refined_parquet(self, grid: pd.DataFrame, referencia: datetime) -> None:
        df = grid.copy()
        df["TIMESTAMP_ATUALIZACAO"] = pd.Timestamp(referencia)
        ts = referencia.strftime("%Y%m%dT%H%M%S")
        out_path = REF_GRID_DIR / f"grid-{ts}.parquet"
        df.to_parquet(out_path, index=False, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION)
        logging.info("REF grid salvo: %s (linhas=%d)", out_path, len(df))


    def _geohash_bbox_center(self, geohash: str) -> Tuple[float, float, float, float, float, float]:
        lat, lon, lat_err, lon_err = gh.decode_exactly(geohash)
        min_lat = lat - lat_err
        max_lat = lat + lat_err
        min_lon = lon - lon_err
        max_lon = lon + lon_err
        return min_lat, min_lon, max_lat, max_lon, lat, lon

    def _persistir_grid_app_firestore(self, grid: pd.DataFrame) -> None:
        if not self.db:
            logging.info("Firestore desativado. Pulando persistência app.")
            return

        col = self.db.collection(FIRESTORE_COLLECTION_RISK)
        self._limpar_colecao(col)

        batch = self.db.batch()
        for i, row in grid.iterrows():
            geohash = str(row["geohash"])
            perfil = str(row["perfil"])
            periodo = str(row["PERIODO"])

            doc_id = f"{geohash}|{perfil}|{periodo}"
            ref = col.document(doc_id)

            min_lat, min_lon, max_lat, max_lon, c_lat, c_lon = self._geohash_bbox_center(geohash)
            p5 = geohash[:5] if len(geohash) >= 5 else geohash
            p6 = geohash[:6] if len(geohash) >= 6 else geohash
            p7 = geohash[:7] if len(geohash) >= 7 else geohash

            payload = {
                "geohash": geohash,
                "perfil": perfil,
                "periodo": periodo,
                "frequencia": int(row.get("frequencia", 0)),
                "risco_total": float(row.get("risco_total", 0.0)),
                "score": float(row.get("score_final", row.get("score", 0.0))),
                "score_heuristico": float(row.get("score", 0.0)),
                "center_lat": float(c_lat),
                "center_lon": float(c_lon),
                "min_lat": float(min_lat),
                "min_lon": float(min_lon),
                "max_lat": float(max_lat),
                "max_lon": float(max_lon),
                "gh5": p5,
                "gh6": p6,
                "gh7": p7,
                "atualizado_em": firestore.SERVER_TIMESTAMP,
            }

            batch.set(ref, payload)

            if (i + 1) % 400 == 0:
                batch.commit()
                batch = self.db.batch()

        batch.commit()
        logging.info("Firestore grid atualizado (%s): %d docs", FIRESTORE_COLLECTION_RISK, len(grid))

    def _limpar_colecao(self, collection_ref) -> None:
        docs = collection_ref.limit(500).stream()
        while True:
            batch = self.db.batch()
            count = 0
            for doc in docs:
                batch.delete(doc.reference)
                count += 1
            if count == 0:
                break
            batch.commit()
            docs = collection_ref.limit(500).stream()


if __name__ == "__main__":
    engine = SafeDriverEngine()
    engine.executar_pipeline(ano_inicio=2022)
