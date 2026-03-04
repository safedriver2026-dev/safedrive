import os
import logging
from datetime import timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple

import pandas as pd
import requests
from prophet import Prophet
import holidays


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LAKE_DIR = Path(os.environ.get("LAKE_DIR", "data_lake"))
REF_EVENTS_DIR = LAKE_DIR / "refined" / "events"
FORECAST_DIR = LAKE_DIR / "refined" / "forecast"
METRICS_DIR = LAKE_DIR / "refined" / "metrics"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_ENGINE = os.environ.get("PARQUET_ENGINE", "pyarrow")
PARQUET_COMPRESSION = os.environ.get("PARQUET_COMPRESSION", "zstd")

DISCORD_SUCESSO = os.environ.get("DISCORD_SUCESSO", "")
DISCORD_ERRO = os.environ.get("DISCORD_ERRO", "")

PROPHET_HORIZON_DIAS = int(os.environ.get("PROPHET_HORIZON_DIAS", "7"))
PROPHET_TOP_GEOHASH = int(os.environ.get("PROPHET_TOP_GEOHASH", "20"))
PROPHET_MIN_POINTS = int(os.environ.get("PROPHET_MIN_POINTS", "14"))  


def carregar_events_refined() -> pd.DataFrame:
    if not REF_EVENTS_DIR.exists():
        return pd.DataFrame()

    paths = list(REF_EVENTS_DIR.rglob("*.parquet"))
    if not paths:
        return pd.DataFrame()

    frames = []
    for p in paths:
        try:
            frames.append(pd.read_parquet(p, columns=["DATA_OCORRENCIA_BO", "geohash"]))
        except Exception:
            continue

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["DATA_OCORRENCIA_BO"] = pd.to_datetime(df["DATA_OCORRENCIA_BO"], errors="coerce")
    df = df.dropna(subset=["DATA_OCORRENCIA_BO", "geohash"])
    return df


def preparar_series_diaria(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.assign(ds=df["DATA_OCORRENCIA_BO"].dt.floor("D"))
          .groupby(["ds", "geohash"])
          .size()
          .reset_index(name="y")
    )


def gerar_previsoes_prophet(series: pd.DataFrame, horizonte_dias: int, top_k: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if series.empty:
        return pd.DataFrame(), pd.DataFrame()

    data_min = series["ds"].min().date()
    data_max = series["ds"].max().date() + timedelta(days=horizonte_dias)

    calendario = holidays.Brazil()
    lista_feriados = [
        {"ds": pd.to_datetime(d), "holiday": "feriado_nacional"}
        for d in calendario
        if data_min <= d <= data_max
    ]
    holidays_df = pd.DataFrame(lista_feriados) if lista_feriados else None

    soma_geo = series.groupby("geohash")["y"].sum().reset_index()
    soma_geo = soma_geo.sort_values("y", ascending=False).head(top_k)

    forecasts = []
    geo_metrics_rows: List[Dict[str, Any]] = []

    for geo in soma_geo["geohash"].tolist():
        df_geo = series[series["geohash"] == geo].copy()
        pontos = int(len(df_geo))
        soma = float(df_geo["y"].sum())

        if pontos < PROPHET_MIN_POINTS:
            geo_metrics_rows.append({
                "geohash": geo,
                "status": "pulado_poucos_pontos",
                "pontos": pontos,
                "soma_y": soma,
            })
            continue

        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            holidays=holidays_df,
        )

        try:
            model.fit(df_geo[["ds", "y"]])
        except Exception as e:
            geo_metrics_rows.append({
                "geohash": geo,
                "status": f"falha_fit:{type(e).__name__}",
                "pontos": pontos,
                "soma_y": soma,
            })
            continue

        futuro = model.make_future_dataframe(periods=horizonte_dias)
        fc = model.predict(futuro)

        out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        out["geohash"] = geo
        forecasts.append(out)

        geo_metrics_rows.append({
            "geohash": geo,
            "status": "treinado",
            "pontos": pontos,
            "soma_y": soma,
        })

    df_forecast = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()
    df_geo_metrics = pd.DataFrame(geo_metrics_rows)
    return df_forecast, df_geo_metrics


def exportar_parquets(df_forecast: pd.DataFrame, df_geo_metrics: pd.DataFrame, series: pd.DataFrame) -> Dict[str, Any]:
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
    info: Dict[str, Any] = {"ts": ts, "forecast_path": "", "forecast_rows": 0}

    if not df_forecast.empty:
        out_path = FORECAST_DIR / f"forecast-{ts}.parquet"
        df_forecast.to_parquet(out_path, index=False, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION)
        info["forecast_path"] = str(out_path)
        info["forecast_rows"] = int(len(df_forecast))

  
    global_metrics = {
        "timestamp_execucao_utc": pd.Timestamp.utcnow(),
        "horizonte_dias": int(PROPHET_HORIZON_DIAS),
        "top_k": int(PROPHET_TOP_GEOHASH),
        "min_points": int(PROPHET_MIN_POINTS),
        "series_rows": int(len(series)),
        "series_geohashes": int(series["geohash"].nunique()) if not series.empty else 0,
        "series_ds_min": series["ds"].min() if not series.empty else pd.NaT,
        "series_ds_max": series["ds"].max() if not series.empty else pd.NaT,
        "treinados": int((df_geo_metrics["status"] == "treinado").sum()) if not df_geo_metrics.empty else 0,
        "pulados": int((df_geo_metrics["status"] != "treinado").sum()) if not df_geo_metrics.empty else 0,
    }

    out_global = METRICS_DIR / f"prophet_metrics_global-{ts}.parquet"
    pd.DataFrame([global_metrics]).to_parquet(out_global, index=False, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION)

    out_geo = METRICS_DIR / f"prophet_metrics_geo-{ts}.parquet"
    df_geo_metrics.to_parquet(out_geo, index=False, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION)

    info["metrics_global_path"] = str(out_global)
    info["metrics_geo_path"] = str(out_geo)
    info.update(global_metrics)
    return info


def enviar_resumo_discord_prophet(campos: Dict[str, Any], status: str = "sucesso") -> None:
    if status == "erro":
        webhook = DISCORD_ERRO or DISCORD_SUCESSO
    elif status == "neutro":
        webhook = DISCORD_SUCESSO or DISCORD_ERRO
    else:
        webhook = DISCORD_SUCESSO or DISCORD_ERRO

    if not webhook:
        return

    cores = {"sucesso": 0x8E44AD, "neutro": 0x3498DB, "erro": 0xE74C3C}
    titulo = {
        "sucesso": "📈 Autobot SafeDriver — BI | Execução concluída",
        "neutro": "📈 Autobot SafeDriver — BI | Execução informativa",
        "erro": "📈 Autobot SafeDriver — BI | Falha detectada",
    }.get(status, "📈 Autobot SafeDriver — BI")

    fields = []
    for k, v in campos.items():
        fields.append({"name": str(k), "value": str(v)[:1024], "inline": False})

    payload = {
        "username": "Autobot SafeDriver",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/2082/2082805.png",
        "embeds": [{
            "title": titulo,
            "color": cores.get(status, 0x3498DB),
            "fields": fields,
            "footer": {"text": "Autobot SafeDriver • Prophet hotspots • Forecast Parquet"},
        }]
    }

    try:
        requests.post(webhook, json=payload, timeout=15)
    except Exception as e:
        logging.warning("Autobot SafeDriver: falha ao enviar resumo Prophet ao Discord (%s).", e)


def gerar_resumo_executivo_prophet(info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "🤖 Autobot SafeDriver": "Resumo Executivo — BI",
        "📥 Fonte de dados": "Refined events (Parquet) — data_lake/refined/events",
        "🧮 Série diária (linhas)": f"{info.get('series_rows', 0):,}",
        "🗺️ Geohashes na série": f"{info.get('series_geohashes', 0):,}",
        "🏷️ Top-K solicitado": str(info.get("top_k", "")),
        "✅ Geohashes treinados": str(info.get("treinados", "")),
        "⏭️ Geohashes pulados": str(info.get("pulados", "")),
        "📆 ds mínimo": str(info.get("series_ds_min", "")),
        "📆 ds máximo": str(info.get("series_ds_max", "")),
        "🔮 Horizonte": f"{info.get('horizonte_dias', '')} dias",
        "📤 Forecast (linhas)": f"{info.get('forecast_rows', 0):,}",
        "📦 Forecast (arquivo)": info.get("forecast_path", "Nenhum (sem previsões)"),
        "🧾 Métricas (global)": info.get("metrics_global_path", ""),
        "🧾 Métricas (por geohash)": info.get("metrics_geo_path", ""),
    }


def main() -> None:
    inicio = datetime.now()

    try:
        df = carregar_events_refined()
        if df.empty:
            campos = {
                "🤖 Autobot SafeDriver": "Resumo Executivo — BI",
                "Status": "Base refined indisponível ou vazia. Prophet não executou.",
            }
            enviar_resumo_discord_prophet(campos, status="neutro")
            return

        series = preparar_series_diaria(df)
        if series.empty:
            campos = {
                "🤖 Autobot SafeDriver": "Resumo Executivo — Prophet",
                "Status": "Série diária vazia. Prophet não executou.",
            }
            enviar_resumo_discord_prophet(campos, status="neutro")
            return

        df_forecast, df_geo_metrics = gerar_previsoes_prophet(series, PROPHET_HORIZON_DIAS, PROPHET_TOP_GEOHASH)
        info = exportar_parquets(df_forecast, df_geo_metrics, series)

        campos = gerar_resumo_executivo_prophet(info)
        status = "sucesso" if info.get("forecast_rows", 0) > 0 else "neutro"
        enviar_resumo_discord_prophet(campos, status=status)

        logging.info("Autobot SafeDriver: Prophet concluído em %ss.", (datetime.now() - inicio).seconds)

    except Exception as e:
        campos = {
            "🤖 Autobot SafeDriver": "Resumo Executivo — BI",
            "📌 Erro": f"{type(e).__name__}: {e}",
            "⏱️ Tempo até falha": f"{(datetime.now()-inicio).seconds}s",
        }
        enviar_resumo_discord_prophet(campos, status="erro")
        raise


if __name__ == "__main__":
    from datetime import datetime
    main()
