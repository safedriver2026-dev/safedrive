import logging
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
from prophet import Prophet
import holidays


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

LAKE_DIR = Path(os.environ.get("LAKE_DIR", "data_lake"))
REF_EVENTS_DIR = LAKE_DIR / "refined" / "events"
FORECAST_DIR = LAKE_DIR / "refined" / "forecast"
FORECAST_DIR.mkdir(parents=True, exist_ok=True)

PARQUET_ENGINE = os.environ.get("PARQUET_ENGINE", "pyarrow")
PARQUET_COMPRESSION = os.environ.get("PARQUET_COMPRESSION", "zstd")

PROPHET_HORIZON_DIAS = int(os.environ.get("PROPHET_HORIZON_DIAS", "7"))
PROPHET_TOP_GEOHASH = int(os.environ.get("PROPHET_TOP_GEOHASH", "20"))


def carregar_events_refined() -> pd.DataFrame:
    if not REF_EVENTS_DIR.exists():
        logging.info("Não encontrado Refined (%s).", REF_EVENTS_DIR)
        return pd.DataFrame()

    paths = list(REF_EVENTS_DIR.rglob("*.parquet"))
    if not paths:
        logging.info("Sem parquets em refined/events.")
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


def gerar_previsoes_prophet(df: pd.DataFrame, horizonte_dias: int, top_k: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    series = (
        df.assign(ds=df["DATA_OCORRENCIA_BO"].dt.floor("D"))
          .groupby(["ds", "geohash"])
          .size()
          .reset_index(name="y")
    )
    if series.empty:
        return pd.DataFrame()

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
    logging.info("Prophet: top_k=%d, horizonte=%d dias", top_k, horizonte_dias)

    for geo in soma_geo["geohash"].tolist():
        df_geo = series[series["geohash"] == geo].copy()
        if len(df_geo) < 14:
            continue

        model = Prophet(
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
            holidays=holidays_df,
        )

        try:
            model.fit(df_geo[["ds", "y"]])
        except Exception:
            continue

        futuro = model.make_future_dataframe(periods=horizonte_dias)
        fc = model.predict(futuro)

        out = fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        out["geohash"] = geo
        forecasts.append(out)

    if not forecasts:
        return pd.DataFrame()

    return pd.concat(forecasts, ignore_index=True)


def main() -> None:
    df = carregar_events_refined()
    if df.empty:
        logging.info("Sem base para Prophet. Encerrando.")
        return

    df_forecast = gerar_previsoes_prophet(df, PROPHET_HORIZON_DIAS, PROPHET_TOP_GEOHASH)
    if df_forecast.empty:
        logging.info("Sem previsões geradas.")
        return

    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = FORECAST_DIR / f"forecast-{ts}.parquet"
    df_forecast.to_parquet(out_path, index=False, engine=PARQUET_ENGINE, compression=PARQUET_COMPRESSION)
    logging.info("Forecast salvo: %s", out_path)


if __name__ == "__main__":
    main()
