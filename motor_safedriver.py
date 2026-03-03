import os
import io
import json
import logging
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List, Union

import numpy as np
import pandas as pd
import pygeohash as gh
import requests
import xgboost as xgb
import firebase_admin
from firebase_admin import credentials, firestore
import holidays


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

AnyStr = Union[str, bytes]


@dataclass(frozen=True)
class CrimeInfo:
    artigo: int
    classe: str   
    peso: float


class SafeDriverEngine:
    """
    Motor de risco SafeDrive

    Fluxo:
    - Lê histórico da SSP (anos configuráveis)
    - Usa NUM_BO para identificar BOs novos
    - Se não houver BO novo:
        - não recalcula malha, não treina modelo
        - apenas envia log informativo
    - Se houver BO novo:
        - baixa + trata + aplica filtros
        - calcula pesos / recência / feriados
        - monta grid geohash + perfil + período
        - treina XGBoost em cima do grid
        - aplica score_final (blend modelo + heurística)
        - atualiza Firestore + CSV analítico
        - registra BOs novos em bo_index
    """

    def __init__(self) -> None:
        self.db = self._iniciar_persistencia()

      
        self.modelo_xgb: Optional[xgb.Booster] = None

        # Mapa de crimes relevantes
        self.crime_map: Dict[str, CrimeInfo] = {
            "FURTO DE VEICULO": CrimeInfo(155, "MEDIO", 1.0),
            "FURTO DE CARGA": CrimeInfo(155, "MEDIO", 1.2),
            "ROUBO DE VEICULO": CrimeInfo(157, "GRAVE", 2.5),
            "ROUBO DE CARGA": CrimeInfo(157, "GRAVE", 3.0),
            "LATROCINIO": CrimeInfo(157, "GRAVISSIMO", 5.0),
            "EXTORSAO MEDIANTE A SEQUESTRO": CrimeInfo(159, "GRAVISSIMO", 5.0),
        }

        # Locais macro
        self.tipos_local_validos = [
            self._limpar_texto("Via Pública"),
            self._limpar_texto("Rodovia/Estrada"),
        ]

        # Subtipos de interesse
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

        # Peso por período do dia
        self.peso_periodo = {
            "MADRUGADA": 1.3,
            "MANHA": 0.9,
            "TARDE": 1.0,
            "NOITE": 1.2,
            "INDEFINIDO": 1.0,
        }

        # Feriados nacionais
        self.calendario_feriados = holidays.Brazil()

   
    def executar_pipeline(
        self,
        ano_inicio: int = 2022,
        ano_fim: Optional[int] = None,
    ) -> None:
        # Referência temporal para recência e logs
        inicio_execucao = datetime.now()
        referencia = inicio_execucao

        if ano_fim is None:
            ano_fim = referencia.year

        logging.info(
            "Iniciando pipeline SafeDriver (histórico %s–%s).",
            ano_inicio,
            ano_fim,
        )

        # Ingestão histórica SSP
        try:
            df_hist = self._carregar_historico(ano_inicio, ano_fim)
        except Exception as e:
            logging.error("Falha na ingestão histórica SSP: %s", e)
            self._enviar_relatorio_consolidado({"Erro de Processamento": str(e)}, "erro")
            return

        total_brutos = len(df_hist)
        logging.info("Total de registros brutos na ingestão: %d", total_brutos)

        # NUM_BO normalizado
        if "NUM_BO" not in df_hist.columns:
            logging.warning("Coluna NUM_BO não encontrada. Dedupe por BO não será aplicado.")
            df_hist["NUM_BO"] = None
        else:
            df_hist["NUM_BO"] = df_hist["NUM_BO"].astype(str).map(self._limpar_texto)

        #  Carrega índice de BOs já processados
        index_bo = self._carregar_index_bo()
        logging.info("Índice BO carregado com %d boletins já conhecidos.", len(index_bo))

        # BOs novos
        df_hist["BO_CONHECIDO"] = df_hist["NUM_BO"].isin(index_bo)
        df_novos = df_hist[~df_hist["BO_CONHECIDO"]].copy()
        qtd_novos_bo = df_novos["NUM_BO"].nunique() if "NUM_BO" in df_novos.columns else 0

        logging.info("Boletins novos detectados nesta execução: %d", qtd_novos_bo)

       
        if qtd_novos_bo == 0:
            logs = {
                "Sincronização": "Sem novos boletins",
                "Total Registros Ingestão": f"{total_brutos:,}",
                "Boletins Novos": "0",
                "Observação": "Pipeline não recalculou a malha por ausência de novos NUM_BO.",
                "Tempo de Resposta": f"{(datetime.now() - inicio_execucao).seconds}s",
                "Concluído em": referencia.strftime("%d/%m/%Y às %H:%M"),
            }
            self._enviar_relatorio_consolidado(logs, status="neutro")
            return

        logging.info("Novos BOs detectados. Recalculando malha e treinando XGBoost com todo o histórico.")

        # Pipeline de risco completo

        #  Saneamento geo
        df_geo = self._sanear_geo(df_hist)

        # Normalização texto
        df_norm = self._normalizar_campos_texto(df_geo)

        # Filtros de conteúdo
        df_filtrado_conteudo = self._filtrar_conteudo(df_norm)

        # Pesos 
        df_pesos, total_qualificados, aproveitamento_final = self._aplicar_pesos(
            df_filtrado_conteudo,
            referencia,
            total_brutos,
        )

        if total_qualificados == 0:
            logs = {
                "Sincronização": "Finalizada",
                "Observação": "Não foram detectados registros qualificados após os filtros.",
                "Total Registros Ingestão": f"{total_brutos:,}",
                "Boletins Novos": f"{qtd_novos_bo:,}",
                "Tempo de Resposta": f"{(datetime.now() - inicio_execucao).seconds}s",
                "Concluído em": referencia.strftime("%d/%m/%Y às %H:%M"),
            }
            self._enviar_relatorio_consolidado(logs, "neutro")
            return

        # Grid + features
        df_final, resumo_grid = self._preparar_grid(df_pesos)

        # Treino XGBoost com grid histórico
        self.modelo_xgb = self._treinar_xgb(resumo_grid)

        # Aplicação do modelo 
        resumo_grid = self._aplicar_modelo_xgb(resumo_grid)

        # Observabilidade básica
        self._log_observabilidade(resumo_grid)

       
        docs_higienizados = self._persistir_risco(resumo_grid)
        self._exportar_bi(df_final, referencia)

     
        novos_numeros_bo = (
            df_novos["NUM_BO"]
            .dropna()
            .astype(str)
            .map(self._limpar_texto)
            .unique()
            .tolist()
        )
        self._atualizar_index_bo(novos_numeros_bo)

        
        logs = self._gerar_logs(
            df_final,
            resumo_grid,
            total_brutos,
            total_qualificados,
            aproveitamento_final,
            docs_higienizados,
        )
        logs["Boletins Novos"] = f"{qtd_novos_bo:,}"
        logs["Tempo de Resposta"] = f"{(datetime.now() - inicio_execucao).seconds}s"
        logs["Concluído em"] = referencia.strftime("%d/%m/%Y às %H:%M")

        self._enviar_relatorio_consolidado(logs, "sucesso")
        logging.info("Pipeline SafeDriver concluída com sucesso.")

    
    def _limpar_texto(self, texto: AnyStr) -> str:
        if not isinstance(texto, str):
            texto = str(texto)
        nfkd = unicodedata.normalize("NFKD", texto)
        return "".join(c for c in nfkd if not unicodedata.combining(c)).upper().strip()

    def _iniciar_persistencia(self):
        secret_json = os.environ.get("FIREBASE_JSON")
        if not secret_json:
            logging.info("FIREBASE_JSON não definido. Execução apenas local (CSV).")
            return None

        if not firebase_admin._apps:
            cred = credentials.Certificate(json.loads(secret_json))
            firebase_admin.initialize_app(cred)

        logging.info("Conexão com Firestore inicializada.")
        return firestore.client()


    def _carregar_index_bo(self) -> set[str]:
        if not self.db:
            return set()

        try:
            colecao = self.db.collection("bo_index")
            docs = colecao.stream()
            return {doc.id for doc in docs}
        except Exception as e:
            logging.warning("Falha ao carregar índice de BO (%s). Seguindo sem dedupe.", e)
            return set()

    def _atualizar_index_bo(self, novos_numeros_bo: List[str]) -> None:
        if not self.db or not novos_numeros_bo:
            return

        try:
            colecao = self.db.collection("bo_index")
            batch = self.db.batch()

            for i, num_bo in enumerate(novos_numeros_bo):
                doc_ref = colecao.document(num_bo)
                batch.set(
                    doc_ref,
                    {
                        "processado_em": firestore.SERVER_TIMESTAMP,
                    },
                )
                if (i + 1) % 400 == 0:
                    batch.commit()
                    batch = self.db.batch()
            batch.commit()
            logging.info("Índice de BO atualizado com %d novos boletins.", len(novos_numeros_bo))
        except Exception as e:
            logging.warning("Falha ao atualizar índice de BO (%s).", e)

  
    def _carregar_dados_ssp(self, ano: int) -> pd.DataFrame:
        url = (
            "https://www.ssp.sp.gov.br/assets/estatistica/transparencia/"
            f"spDados/SPDadosCriminais_{ano}.xlsx"
        )
        logging.info("Baixando arquivo SSP: %s", url)
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()

        excel = pd.ExcelFile(io.BytesIO(resp.content))
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
            df_corrigido.columns = [
                self._limpar_texto(str(c)) for c in df_corrigido.columns
            ]
            df_corrigido["ANO_BASE"] = ano
            df_bruto = pd.concat([df_bruto, df_corrigido], ignore_index=True)

        logging.info("Ingestão SSP %s concluída: %d registros.", ano, len(df_bruto))
        return df_bruto

    def _carregar_historico(
        self,
        ano_inicio: int = 2022,
        ano_fim: Optional[int] = None,
    ) -> pd.DataFrame:
        if ano_fim is None:
            ano_fim = datetime.now().year

        dfs: List[pd.DataFrame] = []

        for ano in range(ano_inicio, ano_fim + 1):
            try:
                df_ano = self._carregar_dados_ssp(ano)
                dfs.append(df_ano)
            except Exception as e:
                logging.warning(
                    "Falha ao carregar ano %s (%s). Pulando este ano.", ano, e
                )

        if not dfs:
            raise RuntimeError("Nenhum ano pôde ser carregado na ingestão histórica.")

        df_hist = pd.concat(dfs, ignore_index=True)
        logging.info(
            "Ingestão histórica concluída: anos %s–%s, total %d registros.",
            ano_inicio,
            ano_fim,
            len(df_hist),
        )
        return df_hist

   
    def _sanear_geo(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        df = df_raw.copy()
        df["LATITUDE"] = pd.to_numeric(df["LATITUDE"], errors="coerce")
        df["LONGITUDE"] = pd.to_numeric(df["LONGITUDE"], errors="coerce")
        df = df.dropna(subset=["LATITUDE", "LONGITUDE"])
        df = df[df["LATITUDE"] != 0]
        logging.info("Saneamento geo: %d registros com lat/long válidos.", len(df))
        return df

    def _normalizar_campos_texto(self, df_geo: pd.DataFrame) -> pd.DataFrame:
        df = df_geo.copy()
        for col in [
            "NATUREZA_APURADA",
            "NOME_MUNICIPIO",
            "DESCR_TIPOLOCAL",
            "DESCR_SUBTIPOLOCAL",
        ]:
            if col in df.columns:
                df[col] = df[col].astype(str).map(self._limpar_texto)
        return df

  
    def _filtrar_conteudo(self, df_norm: pd.DataFrame) -> pd.DataFrame:
        df = df_norm.copy()

        naturezas_validas = list(self.crime_map.keys())
        df = df[df["NATUREZA_APURADA"].isin(naturezas_validas)]
        df = df[df["DESCR_TIPOLOCAL"].isin(self.tipos_local_validos)]
        df = df[df["DESCR_SUBTIPOLOCAL"].isin(self.subtipos_local_validos)]

        logging.info("Filtros de conteúdo: %d registros após filtros.", len(df))
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
        if info.classe in ("GRAVE", "GRAVISSIMO"):
            return "ALERTA"
        return "QUALIFICADO"

    def _peso_crime(self, natureza_limpa: str) -> float:
        info = self.crime_map.get(natureza_limpa)
        if not info:
            return 0.5
        return float(info.peso)

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
        if t in self.tipos_local_validos and s in self.subtipos_local_validos:
            return 1.2
        return 1.0

    def _aplicar_pesos(
        self,
        df_filtrado_conteudo: pd.DataFrame,
        referencia: datetime,
        total_registros_brutos: int,
    ) -> Tuple[pd.DataFrame, int, float]:
        df = df_filtrado_conteudo.copy()

        df["QUALIFICACAO"] = df["NATUREZA_APURADA"].apply(
            self._classificar_qualificacao
        )
        df = df[df["QUALIFICACAO"].notna()].copy()
        total_qualificados = len(df)

        if total_registros_brutos > 0:
            aproveitamento_final = (total_qualificados / total_registros_brutos) * 100
        else:
            aproveitamento_final = 0.0

      
        if "DATA_OCORRENCIA_BO" in df.columns:
            df["DATA_OCORRENCIA_BO"] = pd.to_datetime(
                df["DATA_OCORRENCIA_BO"],
                errors="coerce",
                dayfirst=True,
            )
    
            df["DIAS_DESDE_OCORRENCIA"] = (
                pd.Timestamp(referencia).normalize() - df["DATA_OCORRENCIA_BO"]
            ).dt.days
        else:
            df["DIAS_DESDE_OCORRENCIA"] = np.nan

     
        if "DATA_OCORRENCIA_BO" in df.columns:
            data_col = df["DATA_OCORRENCIA_BO"]
            data_date = data_col.dt.date

            df["IS_FERIADO"] = data_date.isin(self.calendario_feriados).astype(int)
            df["IS_FIM_SEMANA"] = data_col.dt.dayofweek.isin([5, 6]).astype(int)
            df["IS_VESPERA_FERIADO"] = (
                (data_col + pd.Timedelta(days=1))
                .dt.date.isin(self.calendario_feriados)
                .astype(int)
            )
        else:
            df["IS_FERIADO"] = 0
            df["IS_FIM_SEMANA"] = 0
            df["IS_VESPERA_FERIADO"] = 0

        df["PESO_RECENCIA"] = df["DIAS_DESDE_OCORRENCIA"].apply(
            self._calcular_peso_recencia
        )
        df["PESO_CRIME"] = df["NATUREZA_APURADA"].apply(self._peso_crime)
        df["PERIODO"] = df["HORA_OCORRENCIA_BO"].apply(self._definir_periodo)
        df["PESO_PERIODO"] = df["PERIODO"].map(self.peso_periodo).fillna(1.0)
        df["PESO_LOCAL"] = df.apply(
            lambda r: self._peso_local(
                r.get("DESCR_TIPOLOCAL"), r.get("DESCR_SUBTIPOLOCAL")
            ),
            axis=1,
        )

        df["PESO_RISCO"] = (
            df["PESO_CRIME"]
            * df["PESO_RECENCIA"]
            * df["PESO_PERIODO"]
            * df["PESO_LOCAL"]
        )

        logging.info(
            "Pesos aplicados: %d registros qualificados (%.1f%%).",
            total_qualificados,
            aproveitamento_final,
        )

        return df, total_qualificados, aproveitamento_final


    def _preparar_grid(
        self,
        df_com_pesos: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = df_com_pesos.copy()

        # Perfis por qualificação
        df["perfil"] = df["QUALIFICACAO"].apply(
            lambda x: ["Pedestre", "Ciclista", "Motorista"]
            if x == "ALERTA"
            else ["Pedestre", "Ciclista"]
        )
        df_final = df.explode("perfil")

        # Geohash
        df_final["geohash"] = [
            gh.encode(la, lo, precision=7)
            for la, lo in zip(df_final["LATITUDE"], df_final["LONGITUDE"])
        ]

        group_keys = ["geohash", "perfil", "PERIODO"]

        # Agregado principal
        base = (
            df_final.groupby(group_keys)
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

        # Janelas temporais
        for limite, sufixo in [(30, "30"), (90, "90"), (365, "365")]:
            subset = df_final[df_final["DIAS_DESDE_OCORRENCIA"] <= limite]
            if subset.empty:
                base[f"freq_{sufixo}"] = 0.0
                base[f"risco_{sufixo}"] = 0.0
                continue

            stats = (
                subset.groupby(group_keys)
                .agg(
                    **{
                        f"freq_{sufixo}": ("PESO_RISCO", "size"),
                        f"risco_{sufixo}": ("PESO_RISCO", "sum"),
                    }
                )
                .reset_index()
            )
            base = base.merge(stats, on=group_keys, how="left")
            base[f"freq_{sufixo}"] = base[f"freq_{sufixo}"].fillna(0).astype(float)
            base[f"risco_{sufixo}"] = base[f"risco_{sufixo}"].fillna(0.0).astype(float)

        # Score (0.5–10)
        q95 = base["risco_total"].quantile(0.95)
        if pd.isna(q95) or q95 <= 0:
            q95 = 1.0

        base["score"] = (
            (base["risco_total"] / q95) * 10.0
        ).clip(0.5, 10.0).round(2)

        logging.info("Grid preparado: %d células de risco.", len(base))
        return df_final, base

    def _montar_features_xgb(self, df_grid: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        df = df_grid.copy()

        perfil_map = {"Pedestre": 0, "Ciclista": 1, "Motorista": 2}
        periodo_map = {
            "MADRUGADA": 0,
            "MANHA": 1,
            "TARDE": 2,
            "NOITE": 3,
            "INDEFINIDO": 4,
        }

        df["perfil_id"] = df["perfil"].map(perfil_map).fillna(-1).astype(float)
        df["periodo_id"] = df["PERIODO"].map(periodo_map).fillna(-1).astype(float)

        df["score_heuristico"] = df["score"].astype(float)

        feature_cols = [
            "frequencia",
            "risco_total",
            "risco_medio",
            "dias_medio",
            "freq_30",
            "risco_30",
            "freq_90",
            "risco_90",
            "freq_365",
            "risco_365",
            "feriado_pct",
            "fim_semana_pct",
            "vespera_feriado_pct",
            "perfil_id",
            "periodo_id",
            "score_heuristico",
        ]

        X = df[feature_cols].astype(float)
        return X, feature_cols

   
    def _treinar_xgb(self, resumo_grid: pd.DataFrame) -> xgb.Booster:
        X, feature_cols = self._montar_features_xgb(resumo_grid)

       
        q75 = resumo_grid["risco_total"].quantile(0.75)
        y = (resumo_grid["risco_total"] >= q75).astype(int)

        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_cols)

        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "eta": 0.1,
            "max_depth": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "lambda": 1.0,
            "alpha": 0.0,
        }

        logging.info(
            "Treinando modelo XGBoost (amostras=%d, alta=%d, baixa=%d)...",
            len(X),
            int(y.sum()),
            int((y == 0).sum()),
        )
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=150,
        )

       
        caminho = os.environ.get(
            "XGB_MODEL_PATH",
            os.path.join("modelos", "modelo_safedriver_xgb.json"),
        )
        try:
            os.makedirs(os.path.dirname(caminho), exist_ok=True)
            booster.save_model(caminho)
            logging.info("Modelo XGBoost salvo em %s.", caminho)
        except Exception as e:
            logging.warning("Falha ao salvar modelo XGBoost (%s).", e)

        return booster

   
    def _aplicar_modelo_xgb(self, resumo_grid: pd.DataFrame) -> pd.DataFrame:
        df = resumo_grid.copy()

        if not self.modelo_xgb:
            df["score_final"] = df["score"]
            return df

        X, feature_cols = self._montar_features_xgb(df)
        dmat = xgb.DMatrix(X, feature_names=feature_cols)

        try:
            y_pred = self.modelo_xgb.predict(dmat)
        except Exception as e:
            logging.warning(
                "Falha na inferência XGBoost (%s). Mantendo apenas heurístico.", e
            )
            df["score_final"] = df["score"]
            return df

        df["proba_modelo"] = y_pred
        df["score_modelo"] = (df["proba_modelo"] * 10.0).clip(0.5, 10.0)

        alpha = float(os.environ.get("XGB_ALPHA", "0.6"))
        alpha = min(max(alpha, 0.0), 1.0)

        df["score_final"] = (
            alpha * df["score_modelo"] + (1 - alpha) * df["score"]
        ).round(2)

        logging.info("Modelo XGBoost aplicado. alpha=%.2f", alpha)
        return df


    def _log_observabilidade(self, resumo_grid: pd.DataFrame) -> None:
        if resumo_grid.empty:
            logging.warning(
                "Observabilidade: resumo_grid vazio, nada para inspecionar."
            )
            return

        try:
            agrupado = (
                resumo_grid.groupby(["perfil", "PERIODO"])["geohash"]
                .nunique()
                .reset_index(name="qtd_geohash")
            )

            logging.info("Distribuição de geohashes por perfil/período:")
            for _, row in agrupado.iterrows():
                logging.info(
                    "  - perfil=%s | periodo=%s | geohashes=%d",
                    row["perfil"],
                    row["PERIODO"],
                    row["qtd_geohash"],
                )
        except Exception as e:
            logging.warning("Falha ao gerar distribuição de geohashes: %s", e)

        coluna_score = "score_final" if "score_final" in resumo_grid.columns else "score"
        try:
            top5 = (
                resumo_grid.sort_values(coluna_score, ascending=False)
                .head(5)
                .copy()
            )
            logging.info("Top 5 células mais críticas (%s):", coluna_score)
            for _, row in top5.iterrows():
                logging.info(
                    "  - geohash=%s | perfil=%s | periodo=%s | score=%.2f | freq=%d",
                    row["geohash"],
                    row["perfil"],
                    row["PERIODO"],
                    float(row[coluna_score]),
                    int(row.get("frequencia", 0)),
                )
        except Exception as e:
            logging.warning("Falha ao gerar top 5 de risco: %s", e)

        if "score_final" in resumo_grid.columns and "score" in resumo_grid.columns:
            try:
                media_heuristico = float(resumo_grid["score"].mean())
                media_final = float(resumo_grid["score_final"].mean())
                logging.info(
                    "Resumo scores: média heurístico=%.2f | média final=%.2f",
                    media_heuristico,
                    media_final,
                )
            except Exception as e:
                logging.warning("Falha ao calcular médias de score: %s", e)
        else:
            logging.info("Resumo scores: usando apenas score heurístico (sem score_final).")

  
    def _higienizar_ambiente_cloud(self) -> int:
        if not self.db:
            return 0

        colecao_ref = self.db.collection("niveis_risco")
        docs = colecao_ref.limit(500).stream()
        deletados = 0

        while True:
            batch = self.db.batch()
            count = 0
            for doc in docs:
                batch.delete(doc.reference)
                count += 1
            if count == 0:
                break
            batch.commit()
            deletados += count
            docs = colecao_ref.limit(500).stream()

        logging.info("Higienização cloud: %d documentos removidos.", deletados)
        return deletados

    def _persistir_risco(self, resumo_grid: pd.DataFrame) -> int:
        if not self.db:
            logging.info("Sem Firestore: pulando persistência cloud.")
            return 0

        docs_higienizados = self._higienizar_ambiente_cloud()
        batch = self.db.batch()

        for i, row in resumo_grid.iterrows():
            doc_id = f"{row['geohash']}_{row['perfil']}_{row['PERIODO']}"
            doc_ref = self.db.collection("niveis_risco").document(doc_id)
            batch.set(
                doc_ref,
                {
                    "geohash": row["geohash"],
                    "perfil": row["perfil"],
                    "periodo": row["PERIODO"],
                    "frequencia": int(row["frequencia"]),
                    "risco_total": float(row["risco_total"]),
                    "score": float(row["score_final"]),
                    "score_heuristico": float(row["score"]),
                    "atualizado_em": firestore.SERVER_TIMESTAMP,
                },
            )
            if (i + 1) % 400 == 0:
                batch.commit()
                batch = self.db.batch()
        batch.commit()

        logging.info("Persistência cloud concluída.")
        return docs_higienizados

    def _exportar_bi(self, df_final: pd.DataFrame, referencia: datetime) -> None:
        df_export = df_final.copy()
        df_export["TIMESTAMP_ATUALIZACAO"] = referencia
        df_export.to_csv(
            "analise_consolidada_safedriver.csv",
            index=False,
            encoding="utf-8-sig",
        )
        logging.info("Exportação CSV concluída (analise_consolidada_safedriver.csv).")


    def _gerar_logs(
        self,
        df_final: pd.DataFrame,
        resumo_grid: pd.DataFrame,
        total_registros_brutos: int,
        total_qualificados: int,
        aproveitamento_final: float,
        docs_higienizados: int,
    ) -> Dict[str, str]:
        distribuicao = df_final["perfil"].value_counts(normalize=True) * 100

        logs = {
            "📂 Ingestão SSP (histórico)": f"{total_registros_brutos:,} registros",
            "💎 Aproveitamento": f"{total_qualificados:,} úteis ({aproveitamento_final:.1f}%)",
            "🧹 Saneamento Cloud": f"{docs_higienizados:,} removidos",
            "🚶 Pedestres": f"{distribuicao.get('Pedestre', 0):.1f}%",
            "🚲 Ciclistas": f"{distribuicao.get('Ciclista', 0):.1f}%",
            "🚗 Motoristas": f"{distribuicao.get('Motorista', 0):.1f}%",
            "🧱 Células de Risco": f"{len(resumo_grid):,} geohashes",
            "🚀 Status Cloud": "Firestore & BI Atualizados"
            if self.db
            else "Exportação CSV concluída",
        }
        return logs

    def _enviar_relatorio_consolidado(
        self,
        campos: Dict[str, Any],
        status: str = "sucesso",
    ) -> None:
        webhook_sucesso = os.environ.get("DISCORD_SUCESSO")
        webhook_erro = os.environ.get("DISCORD_ERRO")

        if status == "erro" and webhook_erro:
            webhook_url = webhook_erro
        else:
            webhook_url = webhook_sucesso

        if not webhook_url:
            return

        cores = {"sucesso": 0x27AE60, "neutro": 0x3498DB, "erro": 0xE74C3C}

        fields = []
        for k, v in campos.items():
            v_str = str(v)
            inline = "%" in v_str or "s" in v_str
            fields.append(
                {
                    "name": k,
                    "value": v_str,
                    "inline": inline,
                }
            )

        titulo = {
            "sucesso": "🛡️ Relatório de Operações – Sucesso",
            "neutro": "🛡️ Relatório de Operações – Informativo",
            "erro": "🛡️ Relatório de Operações – Erro",
        }.get(status, "🛡️ Relatório de Operações")

        embed = {
            "username": "SafeDriver Autobot",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2082/2082805.png",
            "embeds": [
                {
                    "title": titulo,
                    "description": "Atualização da malha de risco e base analítica efetuada.",
                    "color": cores.get(status, 0x3498DB),
                    "fields": fields,
                    "footer": {
                        "text": "Autobot Infrastructure • Monitoramento Consolidado"
                    },
                }
            ],
        }

        try:
            requests.post(webhook_url, json=embed, timeout=15)
        except Exception as e:
            logging.warning("Falha ao enviar relatório para Discord: %s", e)


if __name__ == "__main__":
    engine = SafeDriverEngine()
    engine.executar_pipeline(ano_inicio=2022)
